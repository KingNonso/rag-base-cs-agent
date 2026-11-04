import uuid
from contextlib import contextmanager

import psycopg


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
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE TABLE IF NOT EXISTS rag_config (id boolean PRIMARY KEY DEFAULT true, embedding_id text NOT NULL, dimensions integer NOT NULL)")
        row = conn.execute("SELECT embedding_id, dimensions FROM rag_config WHERE id=true").fetchone()
        if row and row != (model_id, dimensions):
            raise RuntimeError("Embedding configuration changed; reindex in a new database before switching models")
        if not row:
            conn.execute("INSERT INTO rag_config (id,embedding_id,dimensions) VALUES (true,%s,%s)", (model_id, dimensions))
        conn.execute("CREATE TABLE IF NOT EXISTS documents (id uuid PRIMARY KEY, name text NOT NULL, account_id text, version integer NOT NULL, active boolean NOT NULL DEFAULT false, created_at timestamptz NOT NULL DEFAULT now())")
        conn.execute("CREATE INDEX IF NOT EXISTS documents_scope ON documents (account_id,active)")
        conn.execute(f"CREATE TABLE IF NOT EXISTS chunks (id uuid PRIMARY KEY, document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE, ordinal integer NOT NULL, location text NOT NULL, content text NOT NULL, embedding vector({dimensions}) NOT NULL)")
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_document ON chunks (document_id)")
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id uuid PRIMARY KEY, account_id text NOT NULL, user_id text NOT NULL, created_at timestamptz NOT NULL DEFAULT now())")
        conn.execute("CREATE TABLE IF NOT EXISTS messages (id bigserial PRIMARY KEY, session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, role text NOT NULL, content text NOT NULL, created_at timestamptz NOT NULL DEFAULT now())")
        conn.execute("CREATE TABLE IF NOT EXISTS evaluations (id bigserial PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(), question text NOT NULL, expected text, answer text NOT NULL, abstained boolean NOT NULL, citations jsonb NOT NULL)")


def save_document(conn, name, account_id, chunks, vectors):
    if not chunks or len(chunks) != len(vectors):
        raise ValueError("No usable content or mismatched embeddings")
    old = conn.execute("SELECT id,version FROM documents WHERE name=%s AND account_id IS NOT DISTINCT FROM %s AND active=true FOR UPDATE", (name, account_id)).fetchone()
    document_id = uuid.uuid4()
    conn.execute("INSERT INTO documents (id,name,account_id,version,active) VALUES (%s,%s,%s,%s,false)", (document_id, name, account_id, old[1] + 1 if old else 1))
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        conn.execute("INSERT INTO chunks (id,document_id,ordinal,location,content,embedding) VALUES (%s,%s,%s,%s,%s,%s::vector)", (uuid.uuid4(), document_id, index, chunk.location, chunk.text, vector_literal(vector)))
    if old:
        conn.execute("UPDATE documents SET active=false WHERE id=%s", (old[0],))
    conn.execute("UPDATE documents SET active=true WHERE id=%s", (document_id,))
    return document_id


def search(conn, vector, account_id, limit=12):
    vector = vector_literal(vector)
    rows = conn.execute("""SELECT c.id,d.id,d.name,c.location,c.content,1-(c.embedding <=> %s::vector) AS score
        FROM chunks c JOIN documents d ON d.id=c.document_id
        WHERE d.active=true AND (d.account_id IS NULL OR d.account_id=%s)
        ORDER BY c.embedding <=> %s::vector LIMIT %s""", (vector, account_id, vector, limit)).fetchall()
    return [dict(id=str(r[0]), document_id=str(r[1]), name=r[2], location=r[3], content=r[4], score=float(r[5])) for r in rows]
