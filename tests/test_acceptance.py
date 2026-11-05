"""End-to-end acceptance tests against PostgreSQL + pgvector.

Run with: TEST_DATABASE_URL=postgresql://rag:rag@localhost:5433/rag_test pytest -m integration
The database is wiped at the start of the session.
"""

import io
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

from rag_support.api import create_app
from rag_support.evaluate import run_evaluation
from rag_support.logging_setup import JsonFormatter

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).parent.parent / "fixtures"
HAS_OCR = bool(shutil.which("pdftoppm") and shutil.which("tesseract"))


@pytest.fixture(scope="session")
def database(test_settings):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("DROP SCHEMA public CASCADE")
            conn.execute("CREATE SCHEMA public")
    except psycopg.OperationalError as exc:
        pytest.skip(f"database unavailable: {exc}")
    return url


@pytest.fixture
def client(database, test_settings, providers):
    with TestClient(create_app(test_settings, providers)) as c:
        yield c


@pytest.fixture
def auth(make_token):
    def headers(account="account-a", user="user-a", admin=False):
        return {"Authorization": "Bearer " + make_token(account_id=account, sub=user, roles=["knowledge_admin"] if admin else [])}
    return headers


def upload(client, auth, name, content, account_id=None, admin_account="staff"):
    data = {"account_id": account_id} if account_id else {}
    response = client.post("/v1/documents", files={"file": (name, content)}, data=data, headers=auth(admin_account, "admin", admin=True))
    assert response.status_code == 201, response.text
    return response.json()


def ask(client, auth, message, account="account-a", user="user-a", session_id=None):
    response = client.post("/v1/chat", json={"message": message, "session_id": session_id}, headers=auth(account, user))
    assert response.status_code == 200, response.text
    return response.json()


def make_scan(lines):
    from fpdf import FPDF
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((150, 200 + i * 80), line, fill=0, font=ImageFont.load_default(size=44))
    png = io.BytesIO()
    image.save(png, "PNG")
    pdf = FPDF(unit="pt", format=(612, 792))
    pdf.add_page()
    pdf.image(png, x=0, y=0, w=612, h=792)
    return bytes(pdf.output())


def test_upload_replace_delete_changes_answers(client, auth):
    account = f"acct-{uuid.uuid4()}"
    doc = upload(client, auth, "shipping.txt", b"Shipping policy\n\nWidget orders ship within five business days.", account)
    first = ask(client, auth, "How many business days until widget orders ship?", account)
    assert not first["abstained"] and "five" in first["answer"]
    old_chunk = first["references"][0]["chunk_id"]
    assert first["references"][0]["document_key"] == doc["document_key"]
    assert client.get(f"/v1/references/{old_chunk}", headers=auth(account)).status_code == 200

    replaced = client.put(f"/v1/documents/{doc['document_key']}", files={"file": ("shipping.txt", b"Shipping policy\n\nWidget orders ship within two business days.")}, headers=auth("staff", "admin", admin=True))
    assert replaced.status_code == 200 and replaced.json()["version"] == 2
    second = ask(client, auth, "How many business days until widget orders ship?", account)
    assert not second["abstained"] and "two" in second["answer"] and "five" not in second["answer"]
    assert client.get(f"/v1/references/{old_chunk}", headers=auth(account)).status_code == 404

    failed = client.put(f"/v1/documents/{doc['document_key']}", files={"file": ("shipping.pdf", b"%PDF-1.4 not really a pdf")}, headers=auth("staff", "admin", admin=True))
    assert failed.status_code == 422
    assert "two" in ask(client, auth, "How many business days until widget orders ship?", account)["answer"]

    info = client.get(f"/v1/documents/{doc['document_key']}", headers=auth("staff", "admin", admin=True)).json()
    assert [v["status"] for v in info["versions"]] == ["active", "superseded"]

    assert client.delete(f"/v1/documents/{doc['document_key']}", headers=auth("staff", "admin", admin=True)).json() == {"deleted": True}
    gone = ask(client, auth, "How many business days until widget orders ship?", account)
    assert gone["abstained"] and gone["references"] == []
    assert client.delete(f"/v1/documents/{doc['document_key']}", headers=auth("staff", "admin", admin=True)).status_code == 404
    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
        assert conn.execute("SELECT count(*) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.document_key=%s", (doc["document_key"],)).fetchone()[0] == 0


@pytest.mark.skipif(not HAS_OCR, reason="pdftoppm and tesseract are required for scanned PDFs")
def test_scanned_pdf_upload_replace_delete(client, auth):
    account = f"acct-{uuid.uuid4()}"
    doc = upload(client, auth, "trial-scan.pdf", make_scan(["Trial terms", "The free trial lasts", "21 days for new accounts."]), account)
    assert doc["warnings"] == []
    first = ask(client, auth, "How many days does the free trial last?", account)
    assert not first["abstained"] and "21" in first["answer"] and first["references"][0]["page"] == 1
    client.put(f"/v1/documents/{doc['document_key']}", files={"file": ("trial-scan.pdf", make_scan(["Trial terms", "The free trial lasts", "30 days for new accounts."]))}, headers=auth("staff", "admin", admin=True)).raise_for_status()
    second = ask(client, auth, "How many days does the free trial last?", account)
    assert "30" in second["answer"] and "21" not in second["answer"]
    client.delete(f"/v1/documents/{doc['document_key']}", headers=auth("staff", "admin", admin=True)).raise_for_status()
    assert ask(client, auth, "How many days does the free trial last?", account)["abstained"]


def test_references_resolve_and_unsupported_or_conflicting_abstain(client, auth):
    account = f"acct-{uuid.uuid4()}"
    upload(client, auth, "policy.txt", (FIXTURES / "documents" / "policy.txt").read_bytes(), account)
    upload(client, auth, "legacy-hours.html", (FIXTURES / "documents" / "legacy-hours.html").read_bytes(), account)
    upload(client, auth, "product.md", (FIXTURES / "documents" / "product.md").read_bytes(), account)
    answered = ask(client, auth, "How many support agents does the Basic plan allow?", account)
    assert not answered["abstained"] and answered["references"]
    for ref in answered["references"]:
        resolved = client.get(f"/v1/references/{ref['chunk_id']}", headers=auth(account)).json()
        assert resolved["document_key"] == ref["document_key"] and ref["excerpt"][:100] in resolved["text"]
        assert resolved["start_offset"] is not None and resolved["end_offset"] > resolved["start_offset"]
    conflict = ask(client, auth, "What are the support hours Monday through Friday?", account)
    assert conflict["abstained"] and conflict["abstain_reason"] == "conflicting"
    unsupported = ask(client, auth, "Can I pay with cryptocurrency?", account)
    assert unsupported["abstained"] and unsupported["references"] == []


def test_accounts_are_isolated(client, auth):
    a, b = f"acct-a-{uuid.uuid4()}", f"acct-b-{uuid.uuid4()}"
    private = upload(client, auth, "discount.txt", b"Partner terms\n\nThe partner discount code is ORCHID for this account.", a)
    upload(client, auth, "shared.txt", b"Office locations\n\nThe headquarters office is in Lisbon.")

    mine = ask(client, auth, "What is the partner discount code?", a)
    assert not mine["abstained"] and "ORCHID" in mine["answer"]
    a_chunk, a_session = mine["references"][0]["chunk_id"], mine["session_id"]

    theirs = ask(client, auth, "What is the partner discount code?", b, "user-b")
    assert theirs["abstained"] and "ORCHID" not in theirs["answer"]
    assert all(r["document_key"] != private["document_key"] for r in theirs["references"])

    # shared documents remain visible to both accounts
    assert "Lisbon" in ask(client, auth, "Where is the headquarters office?", b, "user-b")["answer"]

    # supplied IDs belonging to another account are not found
    assert client.post("/v1/chat", json={"message": "and the code?", "session_id": a_session}, headers=auth(b, "user-b")).status_code == 404
    assert client.get(f"/v1/sessions/{a_session}/messages", headers=auth(b, "user-b")).status_code == 404
    assert client.get(f"/v1/references/{a_chunk}", headers=auth(b, "user-b")).status_code == 404
    # the same account but a different user cannot reuse the session either
    assert client.get(f"/v1/sessions/{a_session}/messages", headers=auth(a, "user-a2")).status_code == 404
    own = client.get(f"/v1/sessions/{a_session}/messages", headers=auth(a)).json()
    assert [m["role"] for m in own["messages"]] == ["user", "assistant"]

    # a follow-up in B's own session cannot pull A's evidence
    first = ask(client, auth, "Tell me about partner terms", b, "user-b")
    follow = ask(client, auth, "What is the discount code for it?", b, "user-b", first["session_id"])
    assert "ORCHID" not in follow["answer"] and all(r["document_key"] != private["document_key"] for r in follow["references"])

    # feedback is limited to the caller's own responses
    assert client.post("/v1/feedback", json={"request_id": mine["request_id"], "rating": "up"}, headers=auth(a)).json() == {"recorded": True}
    assert client.post("/v1/feedback", json={"request_id": mine["request_id"], "rating": "down"}, headers=auth(b, "user-b")).status_code == 404


def test_admin_endpoints_require_role_and_token(client, auth):
    assert client.post("/v1/documents", files={"file": ("x.txt", b"hello")}, headers=auth()).status_code == 403
    assert client.get("/v1/documents", headers=auth()).status_code == 403
    assert client.delete(f"/v1/documents/{uuid.uuid4()}", headers=auth()).status_code == 403
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401
    assert client.post("/v1/chat", json={"message": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_logs_have_ids_and_models_but_no_text(client, auth):
    account = f"acct-{uuid.uuid4()}"
    upload(client, auth, "notes.txt", b"Release notes\n\nThe zeppelin dashboard shipped in version nine.", account)
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(JsonFormatter().format(record))
    logger = logging.getLogger("rag_support")
    logger.addHandler(handler)
    try:
        request_id = str(uuid.uuid4())
        response = client.post("/v1/chat", json={"message": "When did the zeppelin dashboard ship?"}, headers={**auth(account), "X-Request-ID": request_id})
    finally:
        logger.removeHandler(handler)
    assert response.headers["X-Request-ID"] == request_id and not response.json()["abstained"]
    events = [json.loads(r) for r in records]
    chat = next(e for e in events if e.get("event") == "chat")
    assert chat["request_id"] == request_id and chat["embedding_model"] == "test:hash-256" and chat["generation_model_version"] == "scripted-1"
    assert chat["candidates"] and "retrieval_ms" in chat
    joined = "\n".join(records)
    assert "zeppelin" not in joined.lower() and "version nine" not in joined


def test_evaluation_produces_per_question_report(database, test_settings, providers):
    embedder, generator, reranker = providers
    questions = json.loads((FIXTURES / "questions.json").read_text())
    documents = FIXTURES / "documents"
    if not HAS_OCR:
        pytest.skip("evaluation fixtures include a scanned PDF")
    with psycopg.connect(database) as conn:
        rows, summary, config = run_evaluation(conn, embedder, generator, reranker, documents, questions, test_settings, log=lambda *_: None)
        assert len(rows) == len(questions) and summary["questions"] == len(questions)
        assert all({"passed", "failures", "retrieval_hit", "abstention_correct"} <= row["metrics"].keys() for row in rows)
        stored = conn.execute("SELECT count(*) FROM eval_results WHERE run_id=%s", (config["run_id"],)).fetchone()[0]
        assert stored == len(questions)
        leftover = conn.execute("SELECT count(*) FROM documents WHERE created_by='evaluation' AND status='active'").fetchone()[0]
        assert leftover == 0


def test_reindex_switches_embedding_model(database, test_settings, monkeypatch, client, auth):
    from conftest import HashEmbedder
    from rag_support import reindex
    from rag_support.store import init_db, search

    account = f"acct-{uuid.uuid4()}"
    upload(client, auth, "colors.txt", b"Brand guide\n\nThe primary brand color is teal.", account)

    def run(dimensions):
        config = test_settings.model_copy(update={"embedding_model": f"hash-{dimensions}", "embedding_dimensions": dimensions})
        monkeypatch.setattr(reindex, "settings", config)
        monkeypatch.setattr(reindex, "embedder_for", lambda *a: HashEmbedder(dimensions))
        monkeypatch.setattr("sys.argv", ["rag-reindex"])
        reindex.main()
        return config

    config = run(128)
    with pytest.raises(RuntimeError, match="rag-reindex"):
        init_db(database, test_settings.embedding_dimensions, test_settings.embedding_id)
    init_db(database, config.embedding_dimensions, config.embedding_id)
    with psycopg.connect(database) as conn:
        hits = search(conn, HashEmbedder(128).embed(["primary brand color"])[0], account)
        assert hits and "teal" in hits[0]["content"]
    run(256)  # restore the configuration the other tests use
    init_db(database, test_settings.embedding_dimensions, test_settings.embedding_id)
