import argparse
import json
import uuid
from pathlib import Path

from .config import settings
from .providers import make_providers
from .service import answer, ingest
from .store import connection, init_db


def main():
    parser = argparse.ArgumentParser(description="Ingest fixture files and evaluate support questions")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--account", default="evaluation")
    args = parser.parse_args()
    init_db(settings.database_url, settings.embedding_dimensions, f"{settings.embedding_provider}:{settings.embedding_model}")
    embedder, generator = make_providers(settings)
    with connection(settings.database_url) as conn:
        for path in sorted(args.documents.iterdir()):
            if path.is_file():
                print("ingested", path.name, ingest(conn, embedder, path.name, path.read_bytes(), args.account))
        conn.commit()
        questions = json.loads(args.questions.read_text())
        for case in questions:
            result = answer(conn, embedder, generator, case["question"], args.account, "evaluation", None, settings)
            conn.execute("INSERT INTO evaluations (question,expected,answer,abstained,citations) VALUES (%s,%s,%s,%s,%s::jsonb)", (case["question"], case.get("expected"), result["answer"], result["abstained"], json.dumps(result["references"])))
            print(json.dumps({"question": case["question"], "expected": case.get("expected"), **result}, indent=2))
        print("Review answers and citations against the expected labels; automated semantic accuracy is intentionally not inferred from string equality.")


if __name__ == "__main__":
    main()
