"""Create a compact comparison figure from an evaluation JSON list."""
from __future__ import annotations
import json
from pathlib import Path


def load_reports(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("reports", payload.get("runs", []))


def plot_comparison(reports: list[dict], output: str | Path) -> None:
    import matplotlib.pyplot as plt
    names = [str(r.get("name", r.get("model", "run"))) for r in reports]
    mae = [r.get("mae", float("nan")) for r in reports]
    plt.figure(figsize=(9, 4))
    plt.bar(names, mae)
    plt.ylabel("MAE (lower is better)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
