"""Delete chat sessions with no activity within SESSION_RETENTION_DAYS. Run it on a schedule."""

import argparse

from .config import settings
from .store import connection, purge_sessions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=settings.session_retention_days)
    args = parser.parse_args()
    if args.days < 1:
        raise SystemExit("--days must be at least 1")
    with connection(settings.database_url) as conn:
        print(f"deleted {purge_sessions(conn, args.days)} sessions older than {args.days} days")


if __name__ == "__main__":
    main()
