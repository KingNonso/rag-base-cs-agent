import uuid

from rag_support.parsing import Passage, chunk_passages, parse_document
from rag_support.service import FALLBACK, answer
from rag_support.config import Settings


def test_parsing_and_chunking():
    passages = parse_document("guide.md", b"# Title\n\nAlpha beta gamma.\n\nDelta epsilon.")
    chunks = chunk_passages(passages, max_chars=25, overlap=5)
    assert len(chunks) >= 2
    assert all(chunk.location == "document" for chunk in chunks)


class FakeConn:
    def __init__(self):
        self.queries = []

    def execute(self, query, params=()):
        self.queries.append((query, params))
        return self

    def fetchall(self):
        return []


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeGenerator:
    def answer(self, question, context, history):
        raise AssertionError("Generator must not run without evidence")


def test_no_evidence_abstains(monkeypatch):
    monkeypatch.setattr("rag_support.service.search", lambda *args: [])
    conn = FakeConn()
    result = answer(conn, FakeEmbedder(), FakeGenerator(), "Unknown?", "account-a", "user-a", None, Settings(),)
    assert result["answer"] == FALLBACK
    assert result["abstained"] is True
    assert result["references"] == []


def test_invalid_citation_abstains(monkeypatch):
    monkeypatch.setattr("rag_support.service.search", lambda *args: [{"id":str(uuid.uuid4()),"document_id":str(uuid.uuid4()),"name":"guide","location":"page 1","content":"Basic allows three agents.","score":0.9}])
    class BadGenerator:
        def answer(self, *args):
            return {"supported": True, "answer": "Three", "citations": [2]}
    result = answer(FakeConn(), FakeEmbedder(), BadGenerator(), "Agents?", "a", "u", None, Settings())
    assert result["abstained"] is True
