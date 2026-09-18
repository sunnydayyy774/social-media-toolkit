"""Create a working DuckDB copy for downstream analysis.

This script never modifies the raw database. By default it refuses to
overwrite an existing working database; pass --force to refresh the copy.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DB = PROJECT_ROOT / "data" / "douyin.duckdb"
WORKING_DIR = PROJECT_ROOT / "data" / "working"
WORKING_DB = WORKING_DIR / "douyin_work.duckdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/working/douyin_work.duckdb.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing working database with a fresh raw copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not RAW_DB.exists():
        raise FileNotFoundError(f"Raw database not found: {RAW_DB}")

    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    if WORKING_DB.exists() and not args.force:
        raw_size = RAW_DB.stat().st_size
        working_size = WORKING_DB.stat().st_size
        logging.info("Working database already exists: %s", WORKING_DB)
        logging.info("Raw size=%s bytes, working size=%s bytes", raw_size, working_size)
        logging.info("Nothing changed. Use --force to refresh the working copy.")
        return

    temp_db = WORKING_DB.with_suffix(".duckdb.tmp")
    if temp_db.exists():
        temp_db.unlink()

    logging.info("Copying raw database:")
    logging.info("  from: %s", RAW_DB)
    logging.info("  to:   %s", WORKING_DB)
    shutil.copy2(RAW_DB, temp_db)
    temp_db.replace(WORKING_DB)

    logging.info("Working database created.")
    logging.info("Size: %s bytes", WORKING_DB.stat().st_size)


if __name__ == "__main__":
    main()
