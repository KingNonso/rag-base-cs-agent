"""Re-embed every stored chunk after an embedding provider, model, or dimension change.

Usage: set the *new* EMBEDDING_* settings in the environment, pause document
ingestion, then run `rag-reindex`. Retrieval keeps using the old vectors until the
final swap transaction commits.
"""

import argparse

import psycopg

from .config import settings
from .providers import embedder_for
from .store import vector_literal


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= settings.embedding_dimensions <= 2000:
        raise SystemExit("pgvector vector dimension must be between 1 and 2000")
    embedder = embedder_for(settings.embedding_provider, settings.embedding_model, settings.embedding_dimensions, settings.openai_api_key)
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        row = conn.execute("SELECT embedding_id, dimensions FROM rag_config WHERE id=true").fetchone()
        if not row:
            raise SystemExit("Database is not initialized; start the API once first")
        if row == (settings.embedding_id, settings.embedding_dimensions):
            raise SystemExit("Stored embeddings already match the configured model; nothing to do")
        print(f"reindexing {row[0]} ({row[1]}d) -> {settings.embedding_id} ({settings.embedding_dimensions}d)")
        conn.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_next")
        conn.execute(f"ALTER TABLE chunks ADD COLUMN embedding_next vector({int(settings.embedding_dimensions)})")
        ids = [r[0] for r in conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()]
        for start in range(0, len(ids), args.batch):
            batch = conn.execute("SELECT id, embed_text FROM chunks WHERE id = ANY(%s)", (ids[start:start + args.batch],)).fetchall()
            vectors = embedder.embed([text for _, text in batch])
            with conn.cursor() as cur:
                cur.executemany("UPDATE chunks SET embedding_next=%s::vector WHERE id=%s", [(vector_literal(v), cid) for (cid, _), v in zip(batch, vectors)])
            print(f"  {min(start + args.batch, len(ids))}/{len(ids)}")
        with conn.transaction():
            missing = conn.execute("SELECT count(*) FROM chunks WHERE embedding_next IS NULL").fetchone()[0]
            if missing:
                raise SystemExit(f"{missing} chunks were added during reindexing; rerun with ingestion paused")
            conn.execute("ALTER TABLE chunks DROP COLUMN embedding")
            conn.execute("ALTER TABLE chunks RENAME COLUMN embedding_next TO embedding")
            conn.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL")
            conn.execute("UPDATE rag_config SET embedding_id=%s, dimensions=%s WHERE id=true", (settings.embedding_id, settings.embedding_dimensions))
        print("done; restart the API with the new embedding settings")


if __name__ == "__main__":
    main()
