#!/usr/bin/env python3
"""Rank CLR/OSD/FASD diagnostic outcomes for next-route selection."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IdeaDiagnostic:
    name: str
    summary: dict[str, Any]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _passes_gate(summary: dict[str, Any]) -> bool:
    return bool(summary.get("gate", {}).get("passes_gate", False))


def _signal_score(name: str, summary: dict[str, Any]) -> float:
    if "idea_score" in summary:
        return _clamp01(float(summary["idea_score"]))
    if "low_reliability_error_fraction" in summary:
        return _clamp01(float(summary["low_reliability_error_fraction"]))
    if "unknown_recall" in summary:
        return _clamp01(float(summary["unknown_recall"]))
    if "usable_rate" in summary:
        return _clamp01(float(summary["usable_rate"]))
    if "mean_quality" in summary:
        return _clamp01(float(summary["mean_quality"]))
    return 0.0


def _risk_score(name: str, summary: dict[str, Any]) -> float:
    if "risk" in summary:
        return _clamp01(float(summary["risk"]))
    defaults = {
        "CLR-YOLO": 0.30,
        "OSD-YOLO": 0.35,
        "FASD-YOLO": 0.40,
    }
    return defaults.get(name, 0.50)


def rank_ideas(ideas: list[IdeaDiagnostic]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idea in ideas:
        passes = _passes_gate(idea.summary)
        signal = _signal_score(idea.name, idea.summary)
        risk = _risk_score(idea.name, idea.summary)
        ranking_score = (1.0 if passes else 0.0) + signal - 0.35 * risk
        rows.append(
            {
                "name": idea.name,
                "passes_gate": passes,
                "signal": round(float(signal), 6),
                "risk": round(float(risk), 6),
                "ranking_score": round(float(ranking_score), 6),
            }
        )
    return sorted(rows, key=lambda row: (row["passes_gate"], row["ranking_score"], row["signal"]), reverse=True)


def write_ranking_report(ranked: list[dict[str, Any]], out: Path) -> None:
    recommendation = next((row["name"] for row in ranked if row["passes_gate"]), "none")
    lines = [
        "# Parallel S-Idea Diagnostic Ranking",
        "",
        "| Rank | Idea | Gate | Signal | Risk | Score |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, start=1):
        lines.append(
            f"| {index} | {row['name']} | {row['passes_gate']} | "
            f"{row['signal']:.6f} | {row['risk']:.6f} | {row['ranking_score']:.6f} |"
        )
    lines.extend(["", f"Recommended next training route: {recommendation}", ""])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank parallel S-idea diagnostics.")
    parser.add_argument("--clr-summary", type=Path, required=True)
    parser.add_argument("--osd-summary", type=Path, required=True)
    parser.add_argument("--fasd-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ranked = rank_ideas(
        [
            IdeaDiagnostic("CLR-YOLO", _read_summary(args.clr_summary)),
            IdeaDiagnostic("OSD-YOLO", _read_summary(args.osd_summary)),
            IdeaDiagnostic("FASD-YOLO", _read_summary(args.fasd_summary)),
        ]
    )
    write_ranking_report(ranked, args.out)


if __name__ == "__main__":
    main()
