"""Labeled evaluation: ingest a document folder into a throwaway account, ask each
labeled question, and score retrieval, citations, abstention, and answer correctness.

Question file: a JSON list of objects with
  id, question, category, expected_abstain (bool),
  expected_documents (file names that should be retrieved and cited),
  key_facts (strings that must all appear in a correct answer),
  expected (free-text reference answer, used by --judge),
  history (optional list of earlier user messages, asked first in the same session).
"""

import argparse
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from .config import Settings, settings
from .providers import make_providers
from .service import answer, ingest
from .store import connection, delete_document, init_db

METRICS = ("retrieval_hit", "citation_support", "abstention_correct", "answer_correct")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9:]+", " ", text.lower().replace("–", "-")).strip()


def score_case(case: dict, result: dict, trace: dict, key_to_name: dict, judge_result: dict | None = None) -> dict:
    expected_docs = set(case.get("expected_documents", []))
    expected_abstain = bool(case.get("expected_abstain"))
    retrieved = [key_to_name.get(c["document_key"]) for c in trace.get("candidates", [])]
    cited = {key_to_name.get(r["document_key"]) for r in result["references"]}
    metrics: dict = {"abstained": result["abstained"], "abstain_reason": result.get("abstain_reason"), "retrieved_documents": list(dict.fromkeys(d for d in retrieved if d)), "cited_documents": sorted(d for d in cited if d)}
    metrics["abstention_correct"] = result["abstained"] == expected_abstain
    if expected_docs:
        metrics["retrieval_hit"] = bool(expected_docs & set(retrieved))
        hits = [i for i, d in enumerate(retrieved) if d in expected_docs]
        metrics["first_relevant_rank"] = hits[0] + 1 if hits else None
    else:
        metrics["retrieval_hit"] = None
    if expected_abstain:
        metrics["citation_support"] = None
        metrics["answer_correct"] = None
    else:
        metrics["citation_support"] = bool(cited) and (not expected_docs or cited <= expected_docs)
        answer_text = _norm(result["answer"]) if not result["abstained"] else ""
        metrics["answer_correct"] = bool(answer_text) and all(_norm(f) in answer_text for f in case.get("key_facts", []))
        if judge_result is not None:
            metrics["judge"] = judge_result
            metrics["answer_correct"] = metrics["answer_correct"] or (not result["abstained"] and judge_result.get("correct") is True)
            metrics["citation_support"] = metrics["citation_support"] and judge_result.get("supported") is True
    failures = [m for m in METRICS if metrics.get(m) is False]
    metrics["failures"] = failures
    metrics["passed"] = not failures
    return metrics


def summarize(rows: list[dict]) -> dict:
    summary = {"questions": len(rows), "passed": sum(r["metrics"]["passed"] for r in rows)}
    for metric in METRICS:
        values = [r["metrics"][metric] for r in rows if r["metrics"].get(metric) is not None]
        summary[metric] = {"passed": sum(values), "applicable": len(values)}
    summary["pass_rate"] = round(summary["passed"] / max(len(rows), 1), 3)
    return summary


def _mark(value) -> str:
    return "–" if value is None else ("✓" if value else "✗")


def render_markdown(rows: list[dict], summary: dict, config: dict) -> str:
    lines = ["# Evaluation report", "", f"Run `{config['run_id']}` · embedding `{config['embedding_model']}` · generation `{config['generation_model']}` · rerank `{config['rerank_provider']}` · min score {config['min_retrieval_score']} · git `{config.get('git_sha') or 'unknown'}`", ""]
    lines.append(f"**{summary['passed']}/{summary['questions']} questions passed.** " + " · ".join(f"{m.replace('_', ' ')}: {summary[m]['passed']}/{summary[m]['applicable']}" for m in METRICS))
    lines += ["", "| ID | Category | Retrieval | Citations | Abstention | Answer | Result | Notes |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        m = row["metrics"]
        notes = f"abstained ({m['abstain_reason']})" if m["abstained"] else "cited: " + ", ".join(m["cited_documents"])
        if m["failures"]:
            notes += "; failed: " + ", ".join(m["failures"])
        lines.append(f"| {row['id']} | {row['category']} | {_mark(m['retrieval_hit'])} | {_mark(m['citation_support'])} | {_mark(m['abstention_correct'])} | {_mark(m['answer_correct'])} | {'pass' if m['passed'] else '**fail**'} | {notes} |")
    lines += ["", "## Answers", ""]
    for row in rows:
        lines.append(f"- **{row['id']}** {row['question']}\n  - expected: {row.get('expected') or ('abstain' if row['expected_abstain'] else '')}\n  - answer: {row['answer']}")
    return "\n".join(lines) + "\n"


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_evaluation(conn, embedder, generator, reranker, documents: Path, questions: list[dict], config: Settings, judge: bool = False, log=print) -> tuple[list[dict], dict, dict]:
    account = f"eval-{uuid.uuid4()}"
    run_id = uuid.uuid4()
    key_to_name = {}
    rows = []
    try:
        for path in sorted(documents.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                info = ingest(conn, embedder, path.name, path.read_bytes(), account, config, created_by="evaluation")
                key_to_name[info["document_key"]] = path.name
                log(f"ingested {path.name}: {info['chunks']} chunks" + (f", {len(info['warnings'])} warnings" if info["warnings"] else ""))
        conn.commit()
        for case in questions:
            session_id = None
            for earlier in case.get("history", []):
                prior, _ = answer(conn, embedder, generator, reranker, earlier, account, "evaluation", session_id, config)
                session_id = uuid.UUID(prior["session_id"])
            result, trace = answer(conn, embedder, generator, reranker, case["question"], account, "evaluation", session_id, config)
            judged = None
            if judge and not result["abstained"] and not case.get("expected_abstain"):
                try:
                    judged = generator.judge(case["question"], case.get("expected", ""), result["answer"], "\n\n".join(r["excerpt"] for r in result["references"]))
                except Exception as exc:
                    judged = {"error": type(exc).__name__}
            metrics = score_case(case, result, trace, key_to_name, judged)
            rows.append({"id": case["id"], "category": case.get("category"), "question": case["question"], "expected": case.get("expected"), "expected_abstain": bool(case.get("expected_abstain")), "answer": result["answer"], "references": result["references"], "metrics": metrics})
            log(f"{'PASS' if metrics['passed'] else 'FAIL'} {case['id']} [{case.get('category')}]" + (f" failed: {', '.join(metrics['failures'])}" if metrics["failures"] else ""))
        summary = summarize(rows)
        run_config = {"run_id": str(run_id), "embedding_model": embedder.model_id, "generation_model": generator.model_id, "rerank_provider": reranker.name, "min_retrieval_score": config.min_retrieval_score, "chunk_max_chars": config.chunk_max_chars, "chunk_overlap": config.chunk_overlap, "context_passages": config.context_passages, "judge": judge, "git_sha": git_sha(), "documents": sorted(key_to_name.values())}
        conn.execute("INSERT INTO eval_runs (id, config, summary) VALUES (%s,%s::jsonb,%s::jsonb)", (run_id, json.dumps(run_config), json.dumps(summary)))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO eval_results (run_id,question_id,category,passed,metrics,answer,refs) VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb)", [(run_id, r["id"], r["category"], r["metrics"]["passed"], json.dumps(r["metrics"]), r["answer"], json.dumps(r["references"])) for r in rows])
        conn.commit()
    finally:
        conn.rollback()
        for key in key_to_name:
            delete_document(conn, uuid.UUID(key))
        conn.execute("DELETE FROM sessions WHERE account_id=%s", (account,))
        conn.commit()
    return rows, summary, run_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="write <report>.json and <report>.md")
    parser.add_argument("--judge", action="store_true", help="also grade answers with the generation model")
    parser.add_argument("--min-pass", type=float, default=None, help="exit 1 if the pass rate is below this fraction")
    args = parser.parse_args(argv)
    init_db(settings.database_url, settings.embedding_dimensions, settings.embedding_id)
    embedder, generator, reranker = make_providers(settings)
    questions = json.loads(args.questions.read_text())
    with connection(settings.database_url) as conn:
        rows, summary, run_config = run_evaluation(conn, embedder, generator, reranker, args.documents, questions, settings, args.judge)
    markdown = render_markdown(rows, summary, run_config)
    print("\n" + markdown)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.with_suffix(".json").write_text(json.dumps({"config": run_config, "summary": summary, "results": rows}, indent=2))
        args.report.with_suffix(".md").write_text(markdown)
    if args.min_pass is not None and summary["pass_rate"] < args.min_pass:
        print(f"pass rate {summary['pass_rate']} is below --min-pass {args.min_pass}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
