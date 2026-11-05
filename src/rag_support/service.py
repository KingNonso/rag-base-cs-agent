import hashlib
import json
import logging
import time
import uuid

from .config import Settings
from .parsing import ParsedDocument, chunk_document, parse_document
from .store import accessible_chunks, create_document, owned_session, replace_document, search, session_messages

logger = logging.getLogger("rag_support")

FALLBACK = "I couldn't find a reliable answer in the available documents. Please contact support for help."
CONFLICT = "The available documents give conflicting information about this, so I can't answer reliably. Please contact support for help."
EMBED_BATCH = 32


def embed_text(title: str, chunk) -> str:
    heading = " > ".join(part for part in (title, chunk.section) if part)
    return f"{heading}\n{chunk.text}" if heading else chunk.text


def prepare(embedder, name: str, data: bytes, config: Settings):
    """Parse, chunk and embed a document. Runs before any database transaction."""
    parsed: ParsedDocument = parse_document(name, data, config.ocr_min_chars, config.ocr_min_confidence)
    chunks = chunk_document(parsed, config.chunk_max_chars, config.chunk_overlap)
    if not chunks:
        raise ValueError("The document contains no usable text")
    pairs = [(chunk, embed_text(parsed.title, chunk)) for chunk in chunks]
    vectors = []
    for index in range(0, len(pairs), EMBED_BATCH):
        vectors.extend(embedder.embed([text for _, text in pairs[index:index + EMBED_BATCH]]))
    return parsed, pairs, vectors


def ingest(conn, embedder, name: str, data: bytes, account_id: str | None, config: Settings, *, title=None, version_label=None, created_by=None):
    parsed, pairs, vectors = prepare(embedder, name, data, config)
    key, document_id, version = create_document(
        conn, name=name, title=title or parsed.title, version_label=version_label or parsed.version, account_id=account_id,
        parsed_text=parsed.text, warnings=parsed.warnings, sha256=hashlib.sha256(data).hexdigest(), created_by=created_by, chunks=pairs, vectors=vectors,
    )
    return {"document_key": str(key), "document_id": str(document_id), "version": version, "title": title or parsed.title, "chunks": len(pairs), "warnings": parsed.warnings}


def replace(conn, embedder, document_key, name: str, data: bytes, config: Settings, *, title=None, version_label=None, created_by=None):
    parsed, pairs, vectors = prepare(embedder, name, data, config)
    document_id, version = replace_document(
        conn, document_key, name=name, title=title or parsed.title, version_label=version_label or parsed.version,
        parsed_text=parsed.text, warnings=parsed.warnings, sha256=hashlib.sha256(data).hexdigest(), created_by=created_by, chunks=pairs, vectors=vectors,
    )
    return {"document_key": str(document_key), "document_id": str(document_id), "version": version, "chunks": len(pairs), "warnings": parsed.warnings}


def _search_query(generator, question: str, history_rows, history: str) -> tuple[str, str]:
    if not history_rows:
        return question, "none"
    try:
        query = generator.rewrite(question, history).strip()
        if query:
            return query[:1000], "model"
    except Exception as exc:  # rewriting is best-effort
        logger.warning("query rewrite failed: %s", type(exc).__name__)
    previous = next((content for role, content, *_ in reversed(history_rows) if role == "user"), "")
    if previous and len(question.split()) < 7:
        return f"{previous[:300]} {question}", "heuristic"
    return question, "none"


def _reference(item: dict) -> dict:
    return {
        "chunk_id": item["id"], "document_id": item["document_id"], "document_key": item["document_key"],
        "document": item["title"], "version_label": item["version_label"], "location": item["location"],
        "page": item["page"], "section": item["section"], "start_offset": item["start_offset"], "end_offset": item["end_offset"],
        "excerpt": item["content"][:500], "low_ocr_quality": item["low_ocr_quality"],
    }


def answer(conn, embedder, generator, reranker, question: str, account_id: str, user_id: str, session_id: uuid.UUID | None, config: Settings, request_id: uuid.UUID | None = None):
    """Answer a question for one account. Returns (response, trace).

    ``trace`` holds identifiers, scores, timings and model versions for logging; it
    never contains question, answer, or document text.
    """
    request_id = request_id or uuid.uuid4()
    trace = {"embedding_model": embedder.model_id, "generation_model": generator.model_id, "rerank_provider": reranker.name}
    if session_id:
        if not owned_session(conn, session_id, account_id, user_id):
            raise PermissionError("Session not found")
    else:
        session_id = uuid.uuid4()
        conn.execute("INSERT INTO sessions (id,account_id,user_id) VALUES (%s,%s,%s)", (session_id, account_id, user_id))
    history_rows = session_messages(conn, session_id, config.max_history_turns * 2)
    history = "\n".join(f"{row[0]}: {row[1][:700]}" for row in history_rows)

    started = time.monotonic()
    query, trace["rewrite"] = _search_query(generator, question, history_rows, history)
    vector = embedder.embed([query])[0]
    candidates = reranker.rerank(question if trace["rewrite"] == "none" else query, search(conn, vector, account_id, config.retrieval_candidates))
    trace["retrieval_ms"] = int((time.monotonic() - started) * 1000)
    trace["candidates"] = [{"chunk_id": c["id"], "document_key": c["document_key"], "score": round(c["score"], 4), "rank_score": round(c.get("rank_score", c["score"]), 4)} for c in candidates]
    candidates = candidates[:config.context_passages]

    result = {"answer": FALLBACK, "references": [], "abstained": True, "abstain_reason": None, "session_id": str(session_id)}
    top_score = max((c["score"] for c in candidates), default=0.0)
    if not candidates:
        result["abstain_reason"] = "no_evidence"
    elif top_score < config.min_retrieval_score:
        result["abstain_reason"] = "low_similarity"
    else:
        context = "\n\n".join(f"[{i}] {c['title']} ({c['location']}): {c['content']}" for i, c in enumerate(candidates, 1))
        started = time.monotonic()
        try:
            generated, trace["generation_model_version"] = generator.answer(question, context, history)
        except Exception as exc:
            generated = None
            trace["error_type"] = type(exc).__name__
            result["abstain_reason"] = "generation_error"
            logger.warning("generation failed request_id=%s error_type=%s", request_id, type(exc).__name__)
        trace["generation_ms"] = int((time.monotonic() - started) * 1000)
        if generated is not None:
            status = generated.get("status")
            if status is None and "supported" in generated:  # tolerate the earlier schema
                status = "answered" if generated.get("supported") is True else "insufficient"
            citations = generated.get("citations")
            text = generated.get("answer")
            if status == "conflicting":
                result.update(answer=CONFLICT, abstain_reason="conflicting")
            elif status != "answered":
                result["abstain_reason"] = "insufficient"
            elif not (isinstance(text, str) and text.strip() and isinstance(citations, list) and citations and all(type(i) is int and 1 <= i <= len(candidates) for i in citations)):
                result["abstain_reason"] = "invalid_citations"
            else:
                cited = [candidates[i - 1] for i in dict.fromkeys(citations)]
                visible = accessible_chunks(conn, [c["id"] for c in cited], account_id)
                references = [_reference(c) for c in cited if c["id"] in visible]
                if len(references) < len(cited):
                    trace["dropped_references"] = len(cited) - len(references)
                if references:
                    result.update(answer=text.strip(), references=references, abstained=False)
                else:
                    result["abstain_reason"] = "references_unavailable"
    trace["top_score"] = round(top_score, 4)
    trace["abstain_reason"] = result["abstain_reason"]
    trace["cited_chunks"] = [r["chunk_id"] for r in result["references"]]
    conn.execute(
        "INSERT INTO messages (session_id,role,content,request_id) VALUES (%s,'user',%s,%s)", (session_id, question, request_id)
    )
    conn.execute(
        "INSERT INTO messages (session_id,role,content,request_id,refs,abstain_reason) VALUES (%s,'assistant',%s,%s,%s::jsonb,%s)",
        (session_id, result["answer"], request_id, json.dumps(result["references"]), result["abstain_reason"]),
    )
    return result, trace
