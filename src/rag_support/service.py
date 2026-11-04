import re
import uuid

from .config import Settings
from .parsing import chunk_passages, parse_document
from .store import save_document, search


FALLBACK = "I couldn't find a reliable answer in the available documents. Please contact support for help."


def ingest(conn, embedder, name: str, data: bytes, account_id: str | None):
    chunks = chunk_passages(parse_document(name, data))
    if not chunks:
        raise ValueError("The document contains no usable text")
    vectors = []
    for index in range(0, len(chunks), 32):
        vectors.extend(embedder.embed([c.text for c in chunks[index:index + 32]]))
    document_id = save_document(conn, name, account_id, chunks, vectors)
    return {"document_id": str(document_id), "chunks": len(chunks)}


def _rerank(question: str, candidates: list[dict]) -> list[dict]:
    words = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    for item in candidates:
        terms = set(re.findall(r"[a-z0-9]{3,}", item["content"].lower()))
        overlap = len(words & terms) / max(len(words), 1)
        item["rank_score"] = 0.8 * item["score"] + 0.2 * overlap
    return sorted(candidates, key=lambda item: item["rank_score"], reverse=True)


def answer(conn, embedder, generator, question: str, account_id: str, user_id: str, session_id: uuid.UUID | None, config: Settings):
    if session_id:
        session = conn.execute("SELECT id FROM sessions WHERE id=%s AND account_id=%s AND user_id=%s", (session_id, account_id, user_id)).fetchone()
        if not session:
            raise PermissionError("Session not found")
    else:
        session_id = uuid.uuid4()
        conn.execute("INSERT INTO sessions (id,account_id,user_id) VALUES (%s,%s,%s)", (session_id, account_id, user_id))
    history_rows = conn.execute("SELECT role,content FROM messages WHERE session_id=%s ORDER BY id DESC LIMIT %s", (session_id, config.max_history_turns * 2)).fetchall()
    history_rows.reverse()
    history = "\n".join(f"{role}: {content[:700]}" for role, content in history_rows)
    previous_question = next((content for role, content in reversed(history_rows) if role == "user"), "")
    search_query = (previous_question[:300] + " " + question) if previous_question and len(question.split()) < 7 else question
    vector = embedder.embed([search_query])[0]
    candidates = _rerank(question, search(conn, vector, account_id))[:6]
    result = {"answer": FALLBACK, "references": [], "abstained": True, "session_id": str(session_id)}
    if candidates and candidates[0]["score"] >= config.min_retrieval_score:
        context = "\n\n".join(f"[{index}] {item['name']} ({item['location']}): {item['content']}" for index, item in enumerate(candidates, 1))
        generated = generator.answer(question, context, history)
        citation_ids = generated.get("citations", [])
        if generated.get("supported") is True and isinstance(generated.get("answer"), str) and generated["answer"].strip() and isinstance(citation_ids, list) and citation_ids and all(type(i) is int and 1 <= i <= len(candidates) for i in citation_ids):
            references = []
            for i in dict.fromkeys(citation_ids):
                item = candidates[i - 1]
                references.append({"document_id": item["document_id"], "document": item["name"], "location": item["location"], "chunk_id": item["id"], "excerpt": item["content"][:500]})
            result.update(answer=generated["answer"].strip(), references=references, abstained=False)
    conn.execute("INSERT INTO messages (session_id,role,content) VALUES (%s,'user',%s),(%s,'assistant',%s)", (session_id, question, session_id, result["answer"]))
    return result
