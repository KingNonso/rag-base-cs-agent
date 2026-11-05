import json
import uuid
from contextlib import contextmanager

import psycopg

# Each entry runs once, in order, inside its own transaction. Never edit an applied
# migration; append a new one instead. The `{dimensions}` placeholder is filled from
# the configured embedding size when the chunks table is first created.
MIGRATIONS = [
    # 1: prototype schema (matches databases created before migrations existed)
    """
    CREATE TABLE IF NOT EXISTS documents (id uuid PRIMARY KEY, name text NOT NULL, account_id text, version integer NOT NULL, active boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now());
    CREATE INDEX IF NOT EXISTS documents_scope ON documents (account_id, active);
    CREATE TABLE IF NOT EXISTS chunks (id uuid PRIMARY KEY, document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE, ordinal integer NOT NULL, location text NOT NULL, content text NOT NULL, embedding vector({dimensions}) NOT NULL);
    CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id);
    CREATE TABLE IF NOT EXISTS sessions (id uuid PRIMARY KEY, account_id text NOT NULL, user_id text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
    CREATE TABLE IF NOT EXISTS messages (id bigserial PRIMARY KEY, session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, role text NOT NULL, content text NOT NULL, created_at timestamptz NOT NULL DEFAULT now());
    """,
    # 2: versioned documents, source locations, feedback, evaluation runs
    """
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_key uuid;
    UPDATE documents SET document_key = id WHERE document_key IS NULL;
    ALTER TABLE documents ALTER COLUMN document_key SET NOT NULL;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS title text;
    UPDATE documents SET title = name WHERE title IS NULL;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS version_label text;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS sha256 text;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS status text;
    UPDATE documents SET status = CASE WHEN active THEN 'active' ELSE 'superseded' END WHERE status IS NULL;
    ALTER TABLE documents ALTER COLUMN status SET NOT NULL;
    ALTER TABLE documents ADD CONSTRAINT documents_status_check CHECK (status IN ('active','superseded','deleted'));
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS warnings jsonb NOT NULL DEFAULT '[]';
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS extracted_text text;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_by text;
    ALTER TABLE documents ADD COLUMN IF NOT EXISTS deleted_at timestamptz;
    ALTER TABLE documents DROP COLUMN active;
    DROP INDEX IF EXISTS documents_scope;
    CREATE INDEX documents_scope ON documents (account_id, status);
    CREATE UNIQUE INDEX documents_one_active ON documents (document_key) WHERE status = 'active';
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page integer;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS section text;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_offset integer;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_offset integer;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS low_ocr_quality boolean NOT NULL DEFAULT false;
    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embed_text text;
    UPDATE chunks SET embed_text = content WHERE embed_text IS NULL;
    ALTER TABLE chunks ALTER COLUMN embed_text SET NOT NULL;
    CREATE INDEX IF NOT EXISTS sessions_owner ON sessions (account_id, user_id);
    CREATE INDEX IF NOT EXISTS messages_session ON messages (session_id, id);
    ALTER TABLE messages ADD COLUMN IF NOT EXISTS request_id uuid;
    ALTER TABLE messages ADD COLUMN IF NOT EXISTS refs jsonb;
    ALTER TABLE messages ADD COLUMN IF NOT EXISTS abstain_reason text;
    CREATE TABLE feedback (id bigserial PRIMARY KEY, request_id uuid NOT NULL UNIQUE, session_id uuid, account_id text NOT NULL, user_id text NOT NULL, rating text NOT NULL CHECK (rating IN ('up','down')), reason text, comment text, created_at timestamptz NOT NULL DEFAULT now());
    CREATE TABLE eval_runs (id uuid PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), config jsonb NOT NULL, summary jsonb NOT NULL);
    CREATE TABLE eval_results (run_id uuid NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE, question_id text NOT NULL, category text, passed boolean NOT NULL, metrics jsonb NOT NULL, answer text NOT NULL, refs jsonb NOT NULL, PRIMARY KEY (run_id, question_id));
    DROP TABLE IF EXISTS evaluations;
    """,
]

DOCUMENT_FIELDS = "d.document_key, d.id, d.name, d.title, d.version, d.version_label, d.account_id, d.status, d.warnings, d.created_at, d.created_by"


def vector_literal(values):
    return "[" + ",".join(str(float(value)) for value in values) + "]"


@contextmanager
def connection(url: str):
    with psycopg.connect(url) as conn:
        yield conn


def init_db(url: str, dimensions: int, model_id: str):
    if not 1 <= dimensions <= 2000:
        raise ValueError("pgvector vector dimension must be between 1 and 2000")
    with psycopg.connect(url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(724501)")
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE TABLE IF NOT EXISTS rag_config (id boolean PRIMARY KEY DEFAULT true, embedding_id text NOT NULL, dimensions integer NOT NULL)")
        row = conn.execute("SELECT embedding_id, dimensions FROM rag_config WHERE id=true").fetchone()
        if row and row != (model_id, dimensions):
            raise RuntimeError(f"Embedding configuration changed from {row[0]} ({row[1]}d) to {model_id} ({dimensions}d); run rag-reindex before switching models")
        if not row:
            conn.execute("INSERT INTO rag_config (id,embedding_id,dimensions) VALUES (true,%s,%s)", (model_id, dimensions))
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_version").fetchall()}
        # Databases created by the prototype already have the version-1 tables.
        if not applied and conn.execute("SELECT to_regclass('documents')").fetchone()[0]:
            conn.execute("INSERT INTO schema_version (version) VALUES (1)")
            applied.add(1)
        for version, sql in enumerate(MIGRATIONS, 1):
            if version in applied:
                continue
            with conn.transaction():
                conn.execute(sql.replace("{dimensions}", str(int(dimensions))))
                conn.execute("INSERT INTO schema_version (version) VALUES (%s)", (version,))


def _insert_version(conn, document_key, name, title, version_label, account_id, version, parsed_text, warnings, sha256, created_by, chunks, vectors):
    document_id = uuid.uuid4()
    conn.execute(
        """INSERT INTO documents (id,document_key,name,title,version_label,account_id,version,status,warnings,extracted_text,sha256,created_by)
        VALUES (%s,%s,%s,%s,%s,%s,%s,'active',%s::jsonb,%s,%s,%s)""",
        (document_id, document_key, name, title, version_label, account_id, version, json.dumps(warnings), parsed_text, sha256, created_by),
    )
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id,document_id,ordinal,location,content,embed_text,page,section,start_offset,end_offset,low_ocr_quality,embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
            [(uuid.uuid4(), document_id, i, c.location, c.text, embed_text, c.page, c.section, c.start, c.end, c.low_ocr_quality, vector_literal(v)) for i, ((c, embed_text), v) in enumerate(zip(chunks, vectors))],
        )
    return document_id


def create_document(conn, *, name, title, version_label, account_id, parsed_text, warnings, sha256, created_by, chunks, vectors):
    """Insert a new logical document. ``chunks`` is a list of (Chunk, embed_text)."""
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("No usable content or mismatched embeddings")
    document_key = uuid.uuid4()
    with conn.transaction():
        document_id = _insert_version(conn, document_key, name, title, version_label, account_id, 1, parsed_text, warnings, sha256, created_by, chunks, vectors)
    return document_key, document_id, 1


def replace_document(conn, document_key, *, name, title, version_label, parsed_text, warnings, sha256, created_by, chunks, vectors):
    """Atomically swap in a new version. The old version stays retrievable until commit."""
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("No usable content or mismatched embeddings")
    with conn.transaction():
        old = conn.execute("SELECT id, version, account_id, title, version_label FROM documents WHERE document_key=%s AND status='active' FOR UPDATE", (document_key,)).fetchone()
        if not old:
            raise LookupError("Document not found")
        version = conn.execute("SELECT max(version) FROM documents WHERE document_key=%s", (document_key,)).fetchone()[0] + 1
        conn.execute("UPDATE documents SET status='superseded', extracted_text=NULL WHERE id=%s", (old[0],))
        conn.execute("DELETE FROM chunks WHERE document_id=%s", (old[0],))
        document_id = _insert_version(conn, document_key, name, title or old[3], version_label, old[2], version, parsed_text, warnings, sha256, created_by, chunks, vectors)
    return document_id, version


def delete_document(conn, document_key) -> bool:
    with conn.transaction():
        rows = conn.execute("UPDATE documents SET status='deleted', deleted_at=now(), extracted_text=NULL WHERE document_key=%s AND status='active' RETURNING id", (document_key,)).fetchall()
        if rows:
            conn.execute("DELETE FROM chunks c USING documents d WHERE c.document_id=d.id AND d.document_key=%s", (document_key,))
    return bool(rows)


def _document_row(row):
    keys = ["document_key", "document_id", "name", "title", "version", "version_label", "account_id", "status", "warnings", "created_at", "created_by"]
    item = dict(zip(keys, row))
    item["document_key"], item["document_id"] = str(item["document_key"]), str(item["document_id"])
    item["created_at"] = item["created_at"].isoformat()
    return item


def list_documents(conn, account_id=None, include_shared=True, limit=100, offset=0):
    where, params = ["d.status='active'"], []
    if account_id is not None:
        where.append("(d.account_id=%s OR d.account_id IS NULL)" if include_shared else "d.account_id=%s")
        params.append(account_id)
    rows = conn.execute(f"SELECT {DOCUMENT_FIELDS} FROM documents d WHERE {' AND '.join(where)} ORDER BY d.created_at DESC LIMIT %s OFFSET %s", (*params, limit, offset)).fetchall()
    return [_document_row(r) for r in rows]


def get_document_versions(conn, document_key):
    rows = conn.execute(f"SELECT {DOCUMENT_FIELDS} FROM documents d WHERE d.document_key=%s ORDER BY d.version DESC", (document_key,)).fetchall()
    return [_document_row(r) for r in rows]


def search(conn, vector, account_id, limit=12):
    vector = vector_literal(vector)
    rows = conn.execute(
        """SELECT c.id, d.id, d.document_key, d.title, d.version_label, d.name, c.location, c.page, c.section, c.start_offset, c.end_offset, c.content, c.low_ocr_quality,
                  1-(c.embedding <=> %s::vector) AS score
        FROM chunks c JOIN documents d ON d.id=c.document_id
        WHERE d.status='active' AND (d.account_id IS NULL OR d.account_id=%s)
        ORDER BY c.embedding <=> %s::vector LIMIT %s""",
        (vector, account_id, vector, limit),
    ).fetchall()
    keys = ["id", "document_id", "document_key", "title", "version_label", "name", "location", "page", "section", "start_offset", "end_offset", "content", "low_ocr_quality", "score"]
    results = []
    for row in rows:
        item = dict(zip(keys, row))
        for key in ("id", "document_id", "document_key"):
            item[key] = str(item[key])
        item["score"] = float(item["score"])
        results.append(item)
    return results


def accessible_chunks(conn, chunk_ids, account_id) -> set[str]:
    """Return the subset of chunk IDs that are active and visible to the account."""
    if not chunk_ids:
        return set()
    rows = conn.execute(
        """SELECT c.id FROM chunks c JOIN documents d ON d.id=c.document_id
        WHERE c.id = ANY(%s::uuid[]) AND d.status='active' AND (d.account_id IS NULL OR d.account_id=%s)""",
        (list(chunk_ids), account_id),
    ).fetchall()
    return {str(r[0]) for r in rows}


def get_reference(conn, chunk_id, account_id):
    row = conn.execute(
        """SELECT c.id, d.document_key, d.title, d.version_label, d.version, c.location, c.page, c.section, c.start_offset, c.end_offset, c.content, c.low_ocr_quality
        FROM chunks c JOIN documents d ON d.id=c.document_id
        WHERE c.id=%s AND d.status='active' AND (d.account_id IS NULL OR d.account_id=%s)""",
        (chunk_id, account_id),
    ).fetchone()
    if not row:
        return None
    keys = ["chunk_id", "document_key", "document", "version_label", "version", "location", "page", "section", "start_offset", "end_offset", "text", "low_ocr_quality"]
    item = dict(zip(keys, row))
    item["chunk_id"], item["document_key"] = str(item["chunk_id"]), str(item["document_key"])
    return item


def owned_session(conn, session_id, account_id, user_id) -> bool:
    return conn.execute("SELECT 1 FROM sessions WHERE id=%s AND account_id=%s AND user_id=%s", (session_id, account_id, user_id)).fetchone() is not None


def session_messages(conn, session_id, limit=None):
    query = "SELECT role, content, refs, abstain_reason, request_id, created_at FROM messages WHERE session_id=%s ORDER BY id"
    rows = conn.execute(query + (" DESC LIMIT %s" if limit else ""), (session_id, limit) if limit else (session_id,)).fetchall()
    if limit:
        rows.reverse()
    return rows


def purge_sessions(conn, retention_days: int) -> int:
    with conn.transaction():
        rows = conn.execute(
            """DELETE FROM sessions s WHERE s.created_at < now() - make_interval(days => %s)
            AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id=s.id AND m.created_at >= now() - make_interval(days => %s)) RETURNING id""",
            (retention_days, retention_days),
        ).fetchall()
    return len(rows)
