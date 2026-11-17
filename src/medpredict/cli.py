"""MedPredict CLI.

    python -m medpredict.cli pipeline          # full evaluation study → SQLite + reports/
    python -m medpredict.cli generate-sample   # data/sample/cohort_sample.csv
"""
from __future__ import annotations

import argparse

from medpredict.common.config import PROJECT_ROOT, get_settings
from medpredict.common.logging import get_logger

log = get_logger("medpredict.cli")


def generate_sample() -> None:
    from medpredict.data.cohort import generate_cohort

    df = generate_cohort(get_settings().dataset["synthetic"])
    out = PROJECT_ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    df.head(400).to_csv(out / "cohort_sample.csv", index=False)
    log.info("Wrote synthetic sample (%d rows)", 400)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="medpredict")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pipeline")
    sub.add_parser("generate-sample")
    a = ap.parse_args(argv)
    generate_sample()
    if a.cmd == "pipeline":
        from medpredict.pipeline import run_pipeline

        run_pipeline()
        log.info("Done → %s", get_settings().db_path)


if __name__ == "__main__":
    main()
