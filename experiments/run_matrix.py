"""Run and record the baseline/ablation matrix.

This runner intentionally records commands and results instead of hiding
choices in notebook state. Use ``--dry-run`` to inspect the planned study.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MATRIX = [
    {"name": "persistence", "kind": "baseline"},
    {"name": "gru", "world_model": "gru", "jepa_weight": 0.0},
    {"name": "full", "world_model": "gru"},
    {"name": "no_ema", "target_momentum": 0.0},
    {"name": "no_vicreg", "variance_weight": 0.0, "cov_weight": 0.0},
    {"name": "no_constraints", "enforce_monotonic": False},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/reports/ablation_matrix.json")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    records = []
    for spec in MATRIX:
        command = [sys.executable, "train.py"]
        record = {"spec": spec, "command": command, "status": "planned"}
        if args.epochs is not None:
            record["epochs"] = args.epochs
        print(spec["name"], "->", " ".join(command))
        if not args.dry_run:
            record["status"] = "planned_manual_config"
            record["note"] = "Run with a fixed Config override and attach evaluate() output; no row is fabricated."
        records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "runs": records}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
