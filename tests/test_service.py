import uuid

import pytest

from rag_support import service
from rag_support.config import Settings
from rag_support.providers import LexicalReranker, parse_json_object
from rag_support.service import CONFLICT, FALLBACK, answer

from conftest import HashEmbedder


class FakeConn:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=()):
        self.queries.append((query, params))
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None


def candidate(content="Basic allows three agents.", score=0.9):
    return {"id": str(uuid.uuid4()), "document_id": str(uuid.uuid4()), "document_key": str(uuid.uuid4()), "title": "Guide", "version_label": None, "name": "guide.md",
            "location": "section Plans", "page": None, "section": "Plans", "start_offset": 0, "end_offset": len(content), "content": content, "low_ocr_quality": False, "score": score}


class Gen:
    model_id = "test:gen"

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def answer(self, *args):
        if self.error:
            raise self.error
        return self.result, "gen-v1"

    def rewrite(self, question, history):
        return question


@pytest.fixture
def run(monkeypatch):
    def go(candidates, generator, visible=None):
        monkeypatch.setattr(service, "search", lambda *a: [dict(c) for c in candidates])
        monkeypatch.setattr(service, "accessible_chunks", lambda conn, ids, account: set(ids) if visible is None else visible)
        monkeypatch.setattr(service, "session_messages", lambda *a: [])
        return answer(FakeConn(), HashEmbedder(), generator, LexicalReranker(), "How many agents on Basic?", "account-a", "user-a", None, Settings(_env_file=None))
    return go


def test_no_evidence_abstains_without_calling_generator(run):
    result, trace = run([], Gen(error=AssertionError("must not run")))
    assert (result["answer"], result["abstained"], result["abstain_reason"]) == (FALLBACK, True, "no_evidence")


def test_low_similarity_abstains(run):
    result, _ = run([candidate(score=0.05)], Gen(error=AssertionError("must not run")))
    assert result["abstain_reason"] == "low_similarity"


def test_answer_with_valid_citation(run):
    item = candidate()
    result, trace = run([item], Gen({"status": "answered", "answer": "Three agents.", "citations": [1]}))
    assert result["abstained"] is False and result["answer"] == "Three agents."
    ref = result["references"][0]
    assert ref["chunk_id"] == item["id"] and ref["document"] == "Guide" and ref["location"] == "section Plans"
    assert trace["generation_model_version"] == "gen-v1" and trace["cited_chunks"] == [item["id"]]


@pytest.mark.parametrize("generated,reason", [
    ({"status": "insufficient", "answer": "", "citations": []}, "insufficient"),
    ({"status": "answered", "answer": "Three", "citations": [2]}, "invalid_citations"),
    ({"status": "answered", "answer": "Three", "citations": []}, "invalid_citations"),
    ({"status": "answered", "answer": " ", "citations": [1]}, "invalid_citations"),
    ({"status": "answered", "answer": "Three", "citations": ["1"]}, "invalid_citations"),
    ({"supported": False, "answer": "x", "citations": [1]}, "insufficient"),
])
def test_unsupported_outputs_abstain(run, generated, reason):
    result, _ = run([candidate()], Gen(generated))
    assert result["abstained"] is True and result["abstain_reason"] == reason and result["references"] == []


def test_conflicting_evidence_abstains_with_conflict_message(run):
    result, _ = run([candidate()], Gen({"status": "conflicting", "answer": "", "citations": [1]}))
    assert result["abstained"] is True and result["answer"] == CONFLICT and result["abstain_reason"] == "conflicting"


@pytest.mark.parametrize("error", [ValueError("bad json"), RuntimeError("http 500"), KeyError("content")])
def test_generation_errors_abstain(run, error):
    result, trace = run([candidate()], Gen(error=error))
    assert result["abstain_reason"] == "generation_error" and trace["error_type"] == type(error).__name__


def test_reference_recheck_drops_inaccessible_chunks(run):
    a, b = candidate("Basic allows three agents."), candidate("Basic agents can export.")
    result, trace = run([a, b], Gen({"status": "answered", "answer": "Three.", "citations": [1, 2]}), visible={a["id"]})
    assert [r["chunk_id"] for r in result["references"]] == [a["id"]] and trace["dropped_references"] == 1
    result, _ = run([a], Gen({"status": "answered", "answer": "Three.", "citations": [1]}), visible=set())
    assert result["abstain_reason"] == "references_unavailable"


def test_trace_has_no_text(run):
    item = candidate("Secret passage text")
    _, trace = run([item], Gen({"status": "answered", "answer": "Answer text", "citations": [1]}))
    assert "Secret passage" not in repr(trace) and "Answer text" not in repr(trace) and "How many agents" not in repr(trace)


def test_followup_uses_rewrite_and_falls_back(monkeypatch):
    seen = []
    monkeypatch.setattr(service, "session_messages", lambda *a: [("user", "Which plans exist?", None, None, None, None), ("assistant", "Basic and Pro.", None, None, None, None)])
    monkeypatch.setattr(service, "owned_session", lambda *a: True)
    monkeypatch.setattr(service, "search", lambda conn, vector, account, limit: seen.append(account) or [])

    class Rewriter(Gen):
        def rewrite(self, question, history):
            assert "Which plans exist?" in history
            return "How many agents does Pro allow?"

    class Broken(Gen):
        def rewrite(self, question, history):
            raise RuntimeError("down")

    _, trace = answer(FakeConn(), HashEmbedder(), Rewriter(), LexicalReranker(), "And the bigger one?", "acct", "u", uuid.uuid4(), Settings(_env_file=None))
    assert trace["rewrite"] == "model" and seen == ["acct"]
    _, trace = answer(FakeConn(), HashEmbedder(), Broken(), LexicalReranker(), "And the bigger one?", "acct", "u", uuid.uuid4(), Settings(_env_file=None))
    assert trace["rewrite"] == "heuristic"


def test_foreign_session_is_rejected(monkeypatch):
    monkeypatch.setattr(service, "owned_session", lambda *a: False)
    with pytest.raises(PermissionError):
        answer(FakeConn(), HashEmbedder(), Gen(), LexicalReranker(), "Hi", "acct", "u", uuid.uuid4(), Settings(_env_file=None))


def test_lexical_rerank_promotes_term_matches():
    items = [candidate("Pro allows twenty agents.", 0.50), candidate("Refunds within thirty days for annual plans.", 0.52)]
    ranked = LexicalReranker().rerank("refunds for annual plans", items)
    assert ranked[0]["content"].startswith("Refunds")
    ranked = LexicalReranker().rerank("how many agents does pro allow", [dict(i) for i in items])
    assert ranked[0]["content"].startswith("Pro")


def test_parse_json_object_accepts_fenced_json():
    assert parse_json_object('```json\n{"status": "answered"}\n```') == {"status": "answered"}
    with pytest.raises(ValueError):
        parse_json_object("[1]")
