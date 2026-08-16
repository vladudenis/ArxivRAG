"""
Generate Overleaf-ready LaTeX (booktabs) tables from a retrieval_eval.json.

Usage:
    python -m src.evaluation.make_tables \
        --input src/evaluation/output/retrieval_eval.json \
        --output src/evaluation/output/thesis_tables.tex

Emits Phase 1 (chunkers), Phase 2 (retrieval sweep), Phase 3 (embedding models),
and a significance table. Best value per column is bolded; significant p-values
(< 0.05) are bolded in the significance table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_LABELS = {
    "hit_at_k": "Hit@k",
    "paper_coverage": "Coverage@k",
    "precision_at_k": "P@k",
    "mrr": "MRR",
    "average_precision": "MAP",
}


def _f(x: float) -> str:
    return f"{x:.4f}"


def _bold_best(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, str]]:
    """Return per-row formatted metric strings with the column max bolded."""
    best = {k: max(r["metrics"][k] for r in rows) for k in keys}
    out = []
    for r in rows:
        cells = {}
        for k in keys:
            v = r["metrics"][k]
            cells[k] = f"\\textbf{{{_f(v)}}}" if abs(v - best[k]) < 1e-12 else _f(v)
        out.append(cells)
    return out


def _table(caption: str, label: str, header: list[str], body_rows: list[list[str]]) -> str:
    cols = "l" + "c" * (len(header) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in body_rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def phase1_table(d: dict[str, Any]) -> str:
    rows = d.get("phase1", [])
    if not rows:
        return ""
    keys = list(METRIC_LABELS.keys())
    fmt = _bold_best(rows, keys)
    body = []
    for r, cells in zip(rows, fmt):
        name = r["config"]["chunking_strategy"].replace("_", "\\_")
        body.append([f"\\texttt{{{name}}}"] + [cells[k] for k in keys])
    header = ["Chunking strategy"] + [METRIC_LABELS[k] for k in keys]
    return _table(
        "Phase 1: chunking-strategy comparison (dense, top-$k=8$, no rerank).",
        "tab:phase1", header, body,
    )


def phase2_table(d: dict[str, Any]) -> str:
    rows = d.get("phase2", [])
    if not rows:
        return ""
    keys = list(METRIC_LABELS.keys())
    fmt = _bold_best(rows, keys)
    body = []
    for r, cells in zip(rows, fmt):
        c = r["config"]
        body.append(
            [str(c["top_k"]), c["retrieval_type"], "yes" if c["re_ranking"] else "no"]
            + [cells[k] for k in keys]
        )
    header = ["top-$k$", "Retrieval", "Rerank"] + [METRIC_LABELS[k] for k in keys]
    return _table(
        "Phase 2: retrieval-configuration sweep on the winning chunker.",
        "tab:phase2", header, body,
    )


def phase3_table(d: dict[str, Any]) -> str:
    rows = d.get("phase3", [])
    if not rows:
        return ""
    keys = list(METRIC_LABELS.keys())
    fmt = _bold_best(rows, keys)
    body = []
    for r, cells in zip(rows, fmt):
        c = r["config"]
        model = str(c.get("embedding_model", "")).replace("_", "\\_")
        dim = str(c.get("embedding_dim", ""))
        body.append([f"\\texttt{{{model}}}", dim] + [cells[k] for k in keys])
    header = ["Embedding model", "dim"] + [METRIC_LABELS[k] for k in keys]
    cfg = rows[0]["config"]
    cap = (
        f"Phase 3: embedding-model comparison ({cfg['chunking_strategy'].replace('_', ' ').lower()}, "
        f"{cfg['retrieval_type']}, top-$k={cfg['top_k']}$, "
        f"{'rerank' if cfg['re_ranking'] else 'no rerank'}; fixed judge embedder)."
    )
    return _table(cap, "tab:phase3", header, body)


def _sig_label(t: dict[str, Any]) -> str:
    name = t["name"]
    metric = METRIC_LABELS.get(t["metric"], t["metric"])
    if "best_vs_second" in name:
        desc = "Ph2: best vs.\\ 2nd-best"
    elif "dense_vs_hybrid" in name:
        desc = "Ph2: dense vs.\\ hybrid ($k{=}10$)"
    elif "baseline_vs_" in name:
        model = str(t["b_config"].get("embedding_model", "")).split("/")[-1]
        desc = f"Ph3: gemma vs.\\ \\texttt{{{model}}}"
    else:
        desc = name.replace("_", "\\_")
    return f"{desc} ({metric})"


def significance_table(d: dict[str, Any]) -> str:
    tests = d.get("significance", [])
    if not tests:
        return ""
    body = []
    for t in tests:
        pb = t["paired_bootstrap"]
        w = t.get("wilcoxon")
        wp = _f(w["p_value"]) if w else "--"
        p = pb["p_value"]
        pstr = f"\\textbf{{{_f(p)}}}" if p < 0.05 else _f(p)
        ci = f"[{pb['lo']:+.4f}, {pb['hi']:+.4f}]"
        body.append([
            _sig_label(t),
            f"{pb['mean_diff']:+.4f}",
            ci,
            pstr,
            wp,
        ])
    header = ["Comparison (metric)", "$\\Delta$", "95\\% CI", "boot $p$", "Wilcoxon $p$"]
    return _table(
        "Paired significance tests (49 queries, seed 42). "
        "$\\Delta$ is the paired mean difference; bold $p<0.05$.",
        "tab:significance", header, body,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Emit LaTeX tables from retrieval_eval.json.")
    ap.add_argument("--input", type=Path, default=Path("src/evaluation/output/retrieval_eval.json"))
    ap.add_argument("--output", type=Path, default=Path("src/evaluation/output/thesis_tables.tex"))
    args = ap.parse_args()

    d = json.loads(args.input.read_text(encoding="utf-8"))
    parts = [
        "% Auto-generated from " + str(args.input).replace("\\", "/"),
        "% Metrics: Hit@k (Success@k; = practical 'Recall@k'), Coverage@k (paper-level,",
        "% capped at min(k,|gold|)/|gold|), P@k, MRR, MAP. Higher is better for all.",
        "",
        phase1_table(d),
        phase2_table(d),
        phase3_table(d),
        significance_table(d),
    ]
    text = "\n".join(p for p in parts if p is not None)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
