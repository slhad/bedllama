#!/usr/bin/env python3
"""
bedllama-benchmark — multi-dimensional latency + throughput benchmark via pi

Dimensions: models × prompt categories × thinking levels
Metrics:    elapsed_ms, word_count, char_count, tokens_est, words_per_sec, tokens_per_sec

Output:
    tmp/results.json   Machine-readable (meta + scenarios[])
    tmp/results.md     Human-readable (matrix + analysis)
    tmp/results.html   Interactive charts (Chart.js, self-contained)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Built-in prompt categories ─────────────────────────────────────────────────

PROMPT_CATEGORIES: dict[str, dict[str, str]] = {
    "short": {
        "prompt": "What is the capital of France? Reply with the city name only.",
        "description": "Single-word factual answer (~1 word)",
    },
    "average": {
        "prompt": (
            "Explain what a REST API is and give one real-world example of when "
            "to use it. Keep your answer to 2-3 sentences."
        ),
        "description": "Short explanation (~50-80 words)",
    },
    "long": {
        "prompt": (
            "Write a detailed technical explanation of how HTTPS works. "
            "Cover: TLS handshake, certificate validation, symmetric key exchange, "
            "and why each step matters for security. "
            "Structure your answer with clear sections and use 4-6 paragraphs."
        ),
        "description": "Multi-paragraph technical answer (~350-500 words)",
    },
}

CATEGORY_ABBREV = {"short": "S", "average": "A", "long": "L"}
THINKING_ABBREV = {"off": "off", "minimal": "min", "low": "low",
                   "medium": "med", "high": "hi", "xhigh": "xhi"}

# ── Default models ─────────────────────────────────────────────────────────────

DEFAULT_MODELS: list[dict] = [
    {"provider": "bedllama",       "model_id": "claude-haiku-4-5",                           "label": "Haiku 4.5 (bedllama)"},
    {"provider": "bedllama",       "model_id": "claude-sonnet-4-6",                          "label": "Sonnet 4.6 (bedllama)"},
    {"provider": "amazon-bedrock", "model_id": "eu.anthropic.claude-haiku-4-5-20251001-v1:0","label": "Haiku 4.5 (Bedrock direct)"},
    {"provider": "amazon-bedrock", "model_id": "eu.anthropic.claude-sonnet-4-6",             "label": "Sonnet 4.6 (Bedrock direct)"},
]

# ── ANSI ──────────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(
    r"\x1b(?:\[[0-9;?<>]*[a-zA-Z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[=>]|[PX^_][^\x1b]*\x1b\\|\\.?)"
)
def strip_ansi(t: str) -> str:
    return _ANSI_RE.sub("", t).replace("\r", "").replace("\x00", "")
def clean_response(raw: str) -> str:
    lines = [l.rstrip() for l in strip_ansi(raw).splitlines() if l.strip()]
    return "\n".join(lines) if lines else "(empty)"

# ── Error detection ────────────────────────────────────────────────────────────

_ERROR_RE = re.compile(
    r"(?i)no api key|validation error|authentication error|"
    r"connection refused|command not found|^Error\b|^error:"
)
def looks_like_error(r: str) -> bool:
    return bool(_ERROR_RE.search(r))

# ── Statistics ─────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data: return 0.0
    s, n = sorted(data), len(data)
    if n == 1: return s[0]
    idx = (p / 100) * (n - 1)
    lo = int(idx); hi = lo + 1
    return s[-1] if hi >= n else s[lo] + (idx - lo) * (s[hi] - s[lo])

def compute_stats(times: list[float], words: list[float] | None = None,
                  chars: list[float] | None = None) -> dict:
    if not times: return {}
    s: dict[str, Any] = {
        "count":     len(times),
        "min_ms":    round(min(times)),
        "max_ms":    round(max(times)),
        "mean_ms":   round(statistics.mean(times)),
        "median_ms": round(statistics.median(times)),
        "p95_ms":    round(percentile(times, 95)),
        "stdev_ms":  round(statistics.stdev(times)) if len(times) > 1 else 0,
    }
    if words:
        wps_list = [w * 1000 / t for w, t in zip(words, times) if t > 0]
        s["mean_words_per_sec"] = round(statistics.mean(wps_list), 1) if wps_list else 0
        s["mean_word_count"]    = round(statistics.mean(words))
    if chars:
        s["mean_char_count"]    = round(statistics.mean(chars))
        s["mean_tokens_est"]    = round(statistics.mean(chars) / 4)   # chars/4 heuristic
        toks_list = [c / 4 * 1000 / t for c, t in zip(chars, times) if t > 0]
        s["mean_tokens_per_sec"] = round(statistics.mean(toks_list), 1) if toks_list else 0
    return s

# ── Pi runner ─────────────────────────────────────────────────────────────────

def run_pi(provider: str, model_id: str, prompt: str, thinking: str,
           dry_run: bool = False) -> tuple[int, str]:
    cmd = ["pi", "--provider", provider, "--model", model_id,
           "--no-tools", "--no-session", "--thinking", thinking, "-p", prompt]
    if dry_run:
        print(f"        [dry] {' '.join(cmd[:9])} -p …")
        return 1000, "(dry-run)"
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed_ms = round((time.monotonic() - t0) * 1000)
        raw = r.stdout + (r.stderr or "")
    except FileNotFoundError:
        return -1, "ERROR: 'pi' command not found"
    except Exception as exc:
        return round((time.monotonic() - t0) * 1000), f"ERROR: {exc}"
    return elapsed_ms, clean_response(raw)

# ── JSON builder ──────────────────────────────────────────────────────────────

def build_json(scenarios: list[dict], meta: dict) -> dict:
    enriched = []
    for sc in scenarios:
        models_out = []
        for m in sc["models"]:
            valid_t = [r["elapsed_ms"] for r in m["runs"] if not r["error"] and r["elapsed_ms"] > 0]
            valid_w = [r["word_count"]  for r in m["runs"] if not r["error"] and r["elapsed_ms"] > 0]
            valid_c = [r["char_count"]  for r in m["runs"] if not r["error"] and r["elapsed_ms"] > 0]
            models_out.append({**m,
                "stats":       compute_stats(valid_t, valid_w, valid_c),
                "error_count": sum(1 for r in m["runs"] if r["error"])})
        enriched.append({**sc, "models": models_out})
    return {"meta": meta, "scenarios": enriched}

# ── Markdown builder ──────────────────────────────────────────────────────────

def _ms(v: Any) -> str: return str(v) if v is not None else "—"

def _mean(entry: dict | None) -> int | None:
    return entry.get("stats", {}).get("mean_ms") if entry else None

def _derive_pairs(models: list[dict]) -> list[tuple[str, str, str]]:
    """Auto-detect (bedllama_label, direct_label, base_name) pairs.

    Matches models by stripping the provider suffix in parentheses:
      'Haiku 4.5 (bedllama)'      -> base 'Haiku 4.5'
      'Haiku 4.5 (Bedrock direct)' -> base 'Haiku 4.5'
    Falls back to matching by model_id substring when labels don't share a base.
    """
    import re
    def base(label: str) -> str:
        return re.sub(r'\s*\([^)]+\)\s*$', '', label).strip()

    bedllama = {base(m['label']): m['label'] for m in models if m.get('provider') == 'bedllama'}
    direct   = {base(m['label']): m['label'] for m in models if m.get('provider') != 'bedllama'}
    pairs = []
    for b, bl_lbl in bedllama.items():
        if b in direct:
            pairs.append((bl_lbl, direct[b], b))
    return pairs

def build_markdown(data: dict) -> str:
    meta      = data["meta"]
    scenarios = data["scenarios"]
    models    = meta["models"]
    thinking_levels = meta["thinking_levels"]
    categories      = list(meta["prompt_categories"].keys())

    lookup: dict[tuple, dict] = {}
    for sc in scenarios:
        for m in sc["models"]:
            lookup[(sc["thinking"], sc["category"], m["label"])] = m

    L: list[str] = []

    # Header
    L += ["# bedllama Benchmark Results", "",
          f"**Date:** {meta['run_date']}  ",
          f"**Iterations/cell:** {meta['iterations']}  ",
          f"**Thinking levels:** {', '.join(f'`{t}`' for t in thinking_levels)}  ",
          f"**Categories:** {', '.join(f'`{c}`' for c in categories)}  ",
          f"**Total runs:** {meta['total_runs']}  ",
          f"**Wall time:** {meta['wall_time_s']//60}m{meta['wall_time_s']%60:02d}s  ",
          f"**pi:** `{meta.get('pi_version','unknown')}`  ", "",
          "## Prompts", ""]
    for cat, info in meta["prompt_categories"].items():
        L += [f"**`{cat}`** — *{info['description']}*", f"> {info['prompt']}", ""]

    # ── bedllama vs Bedrock-direct comparison (PRIMARY section) ──────────────
    route_pairs = _derive_pairs(models)
    col_keys  = list(itertools.product(categories, thinking_levels))
    col_heads = [f"{CATEGORY_ABBREV[c]}/{THINKING_ABBREV.get(t,t)}" for c,t in col_keys]
    if route_pairs:
        L += ["---", "", "## bedllama vs Bedrock-direct", "",
              "Δ ms = bedllama mean − direct mean · (+) bedllama slower · (−) bedllama faster", ""]
        for think in thinking_levels:
            L += [f"### Thinking = `{think}`", "",
                  "| Model pair | " + " | ".join(f"{CATEGORY_ABBREV[c]} Δ" for c in categories) + " |",
                  "|:-----------|" + "|".join("------:" for _ in categories) + "|"]
            for bl_lbl, bd_lbl, pair in route_pairs:
                cells = []
                for cat in categories:
                    ebl = lookup.get((think, cat, bl_lbl)); ebd = lookup.get((think, cat, bd_lbl))
                    mbl = _mean(ebl); mbd = _mean(ebd)
                    if mbl is None or mbd is None:
                        cells.append("—")
                    else:
                        d = mbl - mbd
                        pct = round(d / mbd * 100)
                        sign = "+" if d >= 0 else ""
                        cells.append(f"{sign}{d} ms ({sign}{pct}%)")
                L.append(f"| **{pair}** | " + " | ".join(cells) + " |")
            L.append("")

    # Full matrix — timing
    col_keys  = list(itertools.product(categories, thinking_levels))
    col_heads = [f"{CATEGORY_ABBREV[c]}/{THINKING_ABBREV.get(t,t)}" for c,t in col_keys]
    L += ["---", "", "## Full Matrix — mean response time (ms)", "",
          "> **S**=short · **A**=average · **L**=long  |  **off/min** = thinking level", ""]
    L.append("| Model | " + " | ".join(col_heads) + " |")
    L.append("|:------|" + "|".join("------:" for _ in col_keys) + "|")
    for m in models:
        lbl = m["label"]
        cells = []
        for cat, think in col_keys:
            e = lookup.get((think, cat, lbl))
            mean = _mean(e) if e else None
            cells.append("—" if mean is None else str(mean))
        L.append(f"| **{lbl}** | " + " | ".join(cells) + " |")
    L.append("")

    # Throughput matrix — words/sec
    L += ["## Full Matrix — throughput (words/sec, long prompt only)", "",
          "> Higher = faster token generation. Useful for comparing streaming efficiency.", ""]
    L.append("| Model | " + " | ".join(col_heads) + " |")
    L.append("|:------|" + "|".join("------:" for _ in col_keys) + "|")
    for m in models:
        lbl = m["label"]
        cells = []
        for cat, think in col_keys:
            e = lookup.get((think, cat, lbl))
            wps = e.get("stats", {}).get("mean_words_per_sec") if e else None
            cells.append("—" if wps is None else str(wps))
        L.append(f"| **{lbl}** | " + " | ".join(cells) + " |")
    L.append("")

    # Thinking overhead
    if len(thinking_levels) >= 2:
        base_t = thinking_levels[0]
        L += ["---", "", "## Thinking Overhead (off → next level)", "",
              "Δ = mean_ms(thinking) − mean_ms(off).  +positive = slower.", ""]
        for next_t in thinking_levels[1:]:
            abbrev = THINKING_ABBREV.get(next_t, next_t)
            L += [f"### `off` → `{next_t}`", "",
                  "| Model | " + " | ".join(f"{CATEGORY_ABBREV[c]} Δ" for c in categories) + " |",
                  "|:------|" + "|".join("------:" for _ in categories) + "|"]
            for m in models:
                lbl = m["label"]
                cells = []
                for cat in categories:
                    e0 = lookup.get((base_t, cat, lbl)); e1 = lookup.get((next_t, cat, lbl))
                    m0 = _mean(e0) if e0 else None; m1 = _mean(e1) if e1 else None
                    if m0 is None or m1 is None: cells.append("—")
                    else:
                        d = m1 - m0; pct = round(d / m0 * 100) if m0 else 0
                        cells.append(f"{'+' if d>=0 else ''}{d} ({'+' if d>=0 else ''}{pct}%)")
                L.append(f"| **{lbl}** | " + " | ".join(cells) + " |")
            L.append("")

    # Prompt-length scaling
    if len(categories) >= 2:
        base_c = categories[0]
        L += ["---", "", "## Prompt-Length Scaling (short baseline)", "",
              "Δ = mean_ms(category) − mean_ms(short).  Shows streaming/generation cost.", ""]
        for think in thinking_levels:
            abbrev = THINKING_ABBREV.get(think, think)
            L += [f"### Thinking = `{think}`", "",
                  "| Model | " + " | ".join(f"{CATEGORY_ABBREV[c]} Δ" for c in categories[1:]) + " | short base | short w/s |",
                  "|:------|" + "|".join("------:" for _ in categories[1:]) + "|------:|------:|"]
            for m in models:
                lbl = m["label"]
                e_base = lookup.get((think, base_c, lbl))
                m_base = _mean(e_base) if e_base else None
                wps_base = e_base.get("stats",{}).get("mean_words_per_sec") if e_base else None
                cells = []
                for cat in categories[1:]:
                    e = lookup.get((think, cat, lbl))
                    mv = _mean(e) if e else None
                    if m_base is None or mv is None: cells.append("—")
                    else:
                        d = mv - m_base; pct = round(d/m_base*100) if m_base else 0
                        cells.append(f"+{d} (+{pct}%)" if d >= 0 else f"{d} ({pct}%)")
                L.append(f"| **{lbl}** | " + " | ".join(cells) +
                         f" | {m_base or '—'} | {wps_base or '—'} |")
            L.append("")

    # Per-scenario detail tables
    L += ["---", "", "## Detailed Scenario Results", ""]
    for sc in scenarios:
        think, cat = sc["thinking"], sc["category"]
        at, ac = THINKING_ABBREV.get(think, think), CATEGORY_ABBREV[cat]
        L += [f"### {cat} / thinking={think}  (`{ac}/{at}`)", "",
              f"**Prompt:** `{sc['prompt'][:100]}{'…' if len(sc['prompt'])>100 else ''}`", "",
              "| # | Model | Min | Mean | P95 | σ | Words | Tok≈ | w/s | tok/s | Err |",
              "|--:|:------|----:|-----:|----:|--:|-----:|-----:|----:|------:|----:|"]
        ranked = sorted(sc["models"], key=lambda x: x.get("stats",{}).get("mean_ms", 999999))
        for i, m in enumerate(ranked, 1):
            s = m.get("stats", {}); e = m.get("error_count", 0)
            medal = " 🥇" if i == 1 and not e else ""
            L.append(f"| {i} | **{m['label']}**{medal}"
                     f" | {s.get('min_ms','—')} | {s.get('mean_ms','—')}"
                     f" | {s.get('p95_ms','—')} | {s.get('stdev_ms','—')}"
                     f" | {s.get('mean_word_count','—')} | {s.get('mean_tokens_est','—')}"
                     f" | {s.get('mean_words_per_sec','—')} | {s.get('mean_tokens_per_sec','—')}"
                     f" | {e} |")
        L.append("")
        for m in sc["models"]:
            L += [f"<details><summary>{m['label']} — individual runs</summary>", "",
                  "| run | ms | words | chars | w/s | response (truncated) |",
                  "|----:|---:|------:|------:|----:|:---------------------|"]
            for r in m["runs"]:
                status = "❌ " if r["error"] else ""
                first  = r["response"].splitlines()[0] if r["response"] else ""
                prev   = (status + first)[:85] + ("…" if len(first) > 85 else "")
                wps_r  = round(r["word_count"] * 1000 / r["elapsed_ms"], 1) if r["elapsed_ms"] > 0 and not r["error"] else "—"
                L.append(f"| {r['run']} | {r['elapsed_ms']} | {r['word_count']}"
                         f" | {r['char_count']} | {wps_r} | {prev} |")
            L += ["", "</details>", ""]

    # Notes
    L += ["---", "", "## Notes", "",
          "- Times are **end-to-end wall-clock**: pi startup (~1.4 s) + model latency + streaming.",
          "- `w/s` = words per second (mean across valid runs). Higher = faster streaming.",
          "- `bedllama` route: `client → front (:4000) → LiteLLM (:4001) → AWS Bedrock`.",
          "- `amazon-bedrock` route: `client → pi built-in provider → AWS Bedrock`.",
          "- Thinking `off` = standard inference. `minimal` = shortest reasoning budget.",
          "- For net model latency, subtract pi baseline (~1 400 ms).",
          "", f"*Generated by `scripts/benchmark.py` on {meta['run_date']}*"]

    return "\n".join(L) + "\n"

# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(data: dict) -> str:
    meta      = data["meta"]
    scenarios = data["scenarios"]
    models    = meta["models"]
    thinking_levels = meta["thinking_levels"]
    categories      = list(meta["prompt_categories"].keys())

    lookup: dict[tuple, dict] = {}
    for sc in scenarios:
        for m in sc["models"]:
            lookup[(sc["thinking"], sc["category"], m["label"])] = m

    col_keys  = list(itertools.product(categories, thinking_levels))
    col_heads = [f"{CATEGORY_ABBREV[c]}/{THINKING_ABBREV.get(t,t)}" for c,t in col_keys]

    model_labels  = [m["label"] for m in models]
    model_labels  = [m["label"] for m in models]

    # Color families: each pair gets a solid + light shade (bedllama=solid, direct=light)
    PAIR_COLORS = [
        ("rgba(59,130,246,0.85)",  "rgba(147,197,253,0.75)"),   # blue  (Haiku)
        ("rgba(16,185,129,0.85)",  "rgba(110,231,183,0.75)"),   # green (Sonnet)
        ("rgba(249,115,22,0.85)",  "rgba(253,186,116,0.75)"),   # orange
        ("rgba(168,85,247,0.85)",  "rgba(216,180,254,0.75)"),   # purple
    ]
    model_colors  = [
        "rgba(59,130,246,0.8)", "rgba(16,185,129,0.8)",
        "rgba(249,115,22,0.8)", "rgba(168,85,247,0.8)",
        "rgba(239,68,68,0.8)",  "rgba(234,179,8,0.8)",
    ]

    # ── chart dataset helpers ────────────────────────────────────────────────
    def matrix_row(lbl: str) -> list:
        return [lookup.get((t, c, lbl), {}).get("stats", {}).get("mean_ms") for c, t in col_keys]

    def wps_row(lbl: str) -> list:
        return [lookup.get((t, c, lbl), {}).get("stats", {}).get("mean_words_per_sec") for c, t in col_keys]

    route_pairs = _derive_pairs(models)

    # ── pair-grouped datasets (bedllama solid, direct light — same hue per pair) ──
    pair_chart_datasets: list[dict] = []
    for pi, (bl_lbl, bd_lbl, name) in enumerate(route_pairs):
        solid, light = PAIR_COLORS[pi % len(PAIR_COLORS)]
        pair_chart_datasets.append({
            "label": f"{name} (bedllama)",
            "data":  matrix_row(bl_lbl),
            "backgroundColor": solid,
            "borderColor":     solid.replace("0.85", "1"),
            "borderWidth": 1,
        })
        pair_chart_datasets.append({
            "label": f"{name} (direct)",
            "data":  matrix_row(bd_lbl),
            "backgroundColor": light,
            "borderColor":     light.replace("0.75", "1"),
            "borderWidth": 2,
            "borderDash": True,
        })

    # ── overhead (delta) datasets: Δ ms bedllama−direct, per-bar color ──────
    overhead_datasets: list[dict] = []
    for pi, (bl_lbl, bd_lbl, name) in enumerate(route_pairs):
        bl_row = matrix_row(bl_lbl)
        bd_row = matrix_row(bd_lbl)
        deltas = [bl - bd if bl is not None and bd is not None else None
                  for bl, bd in zip(bl_row, bd_row)]
        bar_colors = [
            "rgba(239,68,68,0.85)"  if (d is not None and d > 0) else
            "rgba(16,185,129,0.85)" if (d is not None and d < 0) else
            "rgba(100,116,139,0.5)"
            for d in deltas
        ]
        overhead_datasets.append({
            "label": name,
            "data":  deltas,
            "backgroundColor": bar_colors,
            "borderColor":     [c.replace("0.85","1").replace("0.5","1") for c in bar_colors],
            "borderWidth": 1,
        })

    # ── per-scenario charts: pair-grouped (bedllama/direct side-by-side) ────
    scenario_charts: list[dict] = []
    for sc in scenarios:
        think, cat = sc["thinking"], sc["category"]
        key = f"{cat}_{think}"
        sc_lookup = {m["label"]: m for m in sc["models"]}
        pair_labels, pair_means, pair_wps, pair_colors_sc = [], [], [], []
        for pi, (bl_lbl, bd_lbl, _name) in enumerate(route_pairs):
            solid, light = PAIR_COLORS[pi % len(PAIR_COLORS)]
            for lbl, color in [(bl_lbl, solid), (bd_lbl, light)]:
                entry = sc_lookup.get(lbl, {})
                pair_labels.append(lbl)
                pair_means.append(entry.get("stats", {}).get("mean_ms"))
                pair_wps.append(entry.get("stats", {}).get("mean_words_per_sec"))
                pair_colors_sc.append(color)
        ranked = sorted(sc["models"], key=lambda x: x.get("stats", {}).get("mean_ms", 999999))
        scenario_charts.append({
            "id":     f"sc_{key}",
            "title":  f"{cat.title()} / thinking={think}",
            "labels": pair_labels,
            "mean":   pair_means,
            "wps":    pair_wps,
            "colors": pair_colors_sc,
        })

    # ── full-matrix datasets (all models, for the detail table) ─────────────
    matrix_datasets = [
        {"label": m["label"],
         "data":  matrix_row(m["label"]),
         "backgroundColor": model_colors[i % len(model_colors)],
         "borderColor":     model_colors[i % len(model_colors)].replace("0.8", "1"),
         "borderWidth": 1}
        for i, m in enumerate(models)
    ]
    wps_datasets = [
        {"label": m["label"],
         "data":  wps_row(m["label"]),
         "backgroundColor": model_colors[i % len(model_colors)],
         "borderColor":     model_colors[i % len(model_colors)].replace("0.8", "1"),
         "borderWidth": 1}
        for i, m in enumerate(models)
    ]

    # thinking overhead per model
    thinking_delta_datasets: list[dict] = []
    if len(thinking_levels) >= 2:
        base_t = thinking_levels[0]
        for next_t in thinking_levels[1:]:
            for i, m in enumerate(models):
                lbl = m["label"]
                deltas = []
                for cat in categories:
                    e0 = lookup.get((base_t, cat, lbl)); e1 = lookup.get((next_t, cat, lbl))
                    m0 = e0.get("stats", {}).get("mean_ms") if e0 else None
                    m1 = e1.get("stats", {}).get("mean_ms") if e1 else None
                    deltas.append(m1 - m0 if (m0 and m1) else None)
                thinking_delta_datasets.append({
                    "label": f"{lbl} ({base_t}→{next_t})",
                    "data":  deltas,
                    "backgroundColor": model_colors[i % len(model_colors)],
                    "borderColor":     model_colors[i % len(model_colors)].replace("0.8","1"),
                    "borderWidth": 1,
                })

    route_pairs = _derive_pairs(models)

    embedded_data = {
        "meta":              {k: v for k, v in meta.items() if k not in ("models",)},
        "modelLabels":       model_labels,
        "colHeads":          col_heads,
        "matrixDatasets":    matrix_datasets,
        "pairChartDatasets": pair_chart_datasets,
        "overheadDatasets":  overhead_datasets,
        "wpsDatasets":       wps_datasets,
        "scenarioCharts":    scenario_charts,
        "thinkingDeltaDs":   thinking_delta_datasets,
        "categories":        categories,
        "scenarios":         scenarios,
        "pairs":             [{"name": p, "bedllama": bl, "direct": bd}
                              for bl, bd, p in route_pairs],
    }

    _HTML_DATA_PLACEHOLDER  = "__BENCHMARK_JSON__"
    _HTML_THABB_PLACEHOLDER = "__THABBREV_JSON__"

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>bedllama Benchmark — __RUN_DATE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.6}}
h1{{font-size:1.8rem;font-weight:700;color:#f8fafc}}
h2{{font-size:1.3rem;font-weight:600;color:#94a3b8;margin-top:2rem;padding-bottom:.4rem;border-bottom:1px solid #1e293b}}
h3{{font-size:1rem;font-weight:600;color:#64748b;margin:.8rem 0 .4rem}}
.page{{max-width:1400px;margin:0 auto;padding:1.5rem}}
.header{{background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid #1e293b;border-radius:.75rem;padding:1.5rem;margin-bottom:1.5rem}}
.header h1{{margin-bottom:.5rem}}
.meta-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:.5rem;margin-top:.75rem}}
.meta-item{{background:#1e293b;border-radius:.4rem;padding:.5rem .75rem;font-size:.8rem;color:#94a3b8}}
.meta-item strong{{color:#e2e8f0;display:block;font-size:.9rem}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:.75rem;padding:1.25rem;margin:1rem 0}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}
.chart-grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}}
@media(max-width:900px){{.chart-grid,.chart-grid-3{{grid-template-columns:1fr}}}}
.chart-wrap{{background:#0f172a;border-radius:.5rem;padding:1rem;min-height:280px;display:flex;align-items:center;justify-content:center}}
canvas{{max-width:100%}}
table{{width:100%;border-collapse:collapse;font-size:.82rem;margin:.5rem 0}}
th{{background:#0f172a;color:#94a3b8;text-align:left;padding:.5rem .6rem;font-weight:600;border-bottom:1px solid #334155}}
td{{padding:.45rem .6rem;border-bottom:1px solid #1e293b;vertical-align:middle}}
tr:hover td{{background:#1e293b}}
.num{{text-align:right;font-variant-numeric:tabular-nums;font-family:monospace}}
.badge{{display:inline-block;padding:.1rem .4rem;border-radius:.25rem;font-size:.72rem;font-weight:600;font-family:monospace}}
.fast{{background:#065f46;color:#6ee7b7}}.slow{{background:#7f1d1d;color:#fca5a5}}.mid{{background:#1e3a5f;color:#93c5fd}}
.medal{{font-size:1rem}}
details summary{{cursor:pointer;color:#60a5fa;font-size:.85rem;padding:.4rem 0;user-select:none}}
details summary:hover{{color:#93c5fd}}
details[open] summary{{color:#93c5fd;margin-bottom:.5rem}}
.tabs{{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0}}
.tab{{padding:.35rem .9rem;border-radius:.4rem;cursor:pointer;font-size:.82rem;background:#0f172a;color:#64748b;border:1px solid #334155;transition:all .15s}}
.tab.active,.tab:hover{{background:#1e40af;color:#e0e7ff;border-color:#3b82f6}}
.tab-content{{display:none}}.tab-content.active{{display:block}}
.prompt-box{{background:#0f172a;border-left:3px solid #3b82f6;padding:.6rem 1rem;border-radius:0 .4rem .4rem 0;font-size:.82rem;color:#94a3b8;margin:.3rem 0 .8rem}}
.legend-dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}}
.section-intro{{color:#64748b;font-size:.85rem;margin:.3rem 0 .8rem}}
.winner-row td{{background:#052e16!important;color:#6ee7b7}}
</style>
</head>
<body>
<div class="page">

<!-- Header -->
<div class="header">
  <h1>🦙 bedllama Benchmark Results</h1>
  <div class="meta-grid" id="metaGrid"></div>
</div>

<!-- Nav tabs -->
<div class="tabs">
  <div class="tab active" onclick="showTab('matrix',this)">📊 Matrix</div>
  <div class="tab" onclick="showTab('scenarios',this)">🎯 Scenarios</div>
  <div class="tab" onclick="showTab('throughput',this)">⚡ Throughput</div>
  <div class="tab" onclick="showTab('thinking',this)">🧠 Thinking</div>
  <div class="tab" onclick="showTab('runs',this)">📋 Raw Runs</div>
</div>

<!-- Matrix tab -->
<div id="tab-matrix" class="tab-content active">
  <div class="card">
    <h2>bedllama vs Bedrock-direct Overhead</h2>
    <p class="section-intro">Δ ms = bedllama mean − direct mean &nbsp;·&nbsp; <span class="badge fast">− faster</span> <span class="badge slow">+ slower</span></p>
    <table id="routingTable"></table>
  </div>
  <div class="card">
    <h2>Full Matrix — mean response time (ms)</h2>
    <p class="section-intro">Columns: S=short · A=average · L=long | off/min = thinking level</p>
    <table id="matrixTable"></table>
  </div>
  <div class="card">
    <h2>Pair Comparison — Response Time</h2>
    <p class="section-intro">Solid = bedllama &nbsp;·&nbsp; Light = Bedrock direct &nbsp;·&nbsp; Same hue = same model family</p>
    <div class="chart-wrap" style="min-height:360px">
      <canvas id="pairChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>bedllama Overhead per Scenario (Δ ms)</h2>
    <p class="section-intro">Green = bedllama faster · Red = bedllama slower · Zero line = parity</p>
    <div class="chart-wrap" style="min-height:280px">
      <canvas id="overheadChart"></canvas>
    </div>
  </div>
</div>

<!-- Scenarios tab -->
<div id="tab-scenarios" class="tab-content">
  <div class="card">
    <h2>Per-Scenario Ranking</h2>
    <div class="tabs" id="scenarioTabs"></div>
    <div id="scenarioContent"></div>
  </div>
</div>

<!-- Throughput tab -->
<div id="tab-throughput" class="tab-content">
  <div class="card">
    <h2>Words per Second (mean)</h2>
    <p class="section-intro">Higher = faster token streaming. Most meaningful for long responses.</p>
    <table id="wpsTable"></table>
  </div>
  <div class="card">
    <div class="chart-wrap" style="min-height:350px">
      <canvas id="wpsChart"></canvas>
    </div>
  </div>
</div>

<!-- Thinking tab -->
<div id="tab-thinking" class="tab-content">
  <div class="card">
    <h2>Thinking Overhead (off → minimal)</h2>
    <p class="section-intro">Δ ms added when switching from thinking=off to thinking=minimal. Negative = faster with thinking.</p>
    <table id="thinkingTable"></table>
    <div class="chart-wrap" style="min-height:320px;margin-top:1rem">
      <canvas id="thinkingChart"></canvas>
    </div>
  </div>
</div>

<!-- Raw runs tab -->
<div id="tab-runs" class="tab-content">
  <div class="card">
    <h2>Individual Runs</h2>
    <div id="rawRuns"></div>
  </div>
</div>

</div><!-- end .page -->

<script>
const D = __BENCHMARK_JSON__;

function showTab(name, el) {{
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(e => e.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
}}

const COLORS = [
  'rgba(59,130,246,0.85)','rgba(16,185,129,0.85)','rgba(249,115,22,0.85)',
  'rgba(168,85,247,0.85)','rgba(239,68,68,0.85)','rgba(234,179,8,0.85)',
];
const BORDERS = COLORS.map(c => c.replace('0.85','1'));

// ── Meta grid ──────────────────────────────────────────────────────────────
(function buildMeta() {{
  const m = D.meta;
  const items = [
    ['Date', m.run_date.replace('T',' ').replace('Z','')],
    ['Iterations/cell', m.iterations],
    ['Thinking levels', m.thinking_levels.join(', ')],
    ['Categories', m.categories.join(', ')],
    ['Total runs', m.total_runs],
    ['Wall time', Math.floor(m.wall_time_s/60)+'m'+String(m.wall_time_s%60).padStart(2,'0')+'s'],
    ['pi version', m.pi_version],
  ];
  document.getElementById('metaGrid').innerHTML = items.map(([k,v]) =>
    `<div class="meta-item"><strong>${{v}}</strong>${{k}}</div>`).join('');
}})();

// ── Helpers ────────────────────────────────────────────────────────────────
function colorCell(val, min, max) {{
  if (val == null) return '';
  const norm = max > min ? (val - min) / (max - min) : 0;
  if (norm < 0.33) return 'fast';
  if (norm > 0.66) return 'slow';
  return 'mid';
}}

// ── Matrix table ───────────────────────────────────────────────────────────
(function buildMatrix() {{
  const heads = D.colHeads;
  const all = [];
  D.matrixDatasets.forEach(ds => ds.data.forEach(v => v && all.push(v)));
  const mn = Math.min(...all), mx = Math.max(...all);

  let html = '<thead><tr><th>Model</th>' + heads.map(h=>`<th class="num">${{h}}</th>`).join('') + '</tr></thead><tbody>';
  // find winner per column
  const winners = heads.map((_,ci) => {{
    let best = Infinity, bestLbl = '';
    D.matrixDatasets.forEach(ds => {{ if (ds.data[ci] != null && ds.data[ci] < best) {{ best=ds.data[ci]; bestLbl=ds.label; }} }});
    return bestLbl;
  }});
  D.matrixDatasets.forEach(ds => {{
    html += '<tr>';
    html += `<td><span class="legend-dot" style="background:${{ds.borderColor}}"></span>${{ds.label}}</td>`;
    ds.data.forEach((v,ci) => {{
      const isWinner = ds.label === winners[ci];
      const cls = isWinner ? 'num fast' : 'num ' + colorCell(v,mn,mx);
      html += `<td class="${{cls}}">${{v != null ? v+'ms' : '—'}}${{isWinner?' 🥇':''}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody>';
  document.getElementById('matrixTable').innerHTML = html;
}})();

// ── Pair comparison chart ──────────────────────────────────────────────────
new Chart(document.getElementById('pairChart'), {{
  type: 'bar',
  data: {{ labels: D.colHeads, datasets: D.pairChartDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', boxWidth:12 }} }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.raw}}ms` }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#1e293b' }} }},
      y: {{ ticks: {{ color:'#64748b', callback: v => v+'ms' }}, grid: {{ color:'#1e293b' }},
           title: {{ display: true, text: 'Response time (ms)', color:'#64748b' }} }},
    }},
  }},
}});

// ── Overhead delta chart ───────────────────────────────────────────────────
new Chart(document.getElementById('overheadChart'), {{
  type: 'bar',
  data: {{ labels: D.colHeads, datasets: D.overheadDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color:'#94a3b8', boxWidth:12 }} }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.raw >= 0 ? '+' : ''}}${{ctx.raw}}ms` }} }},
    }},
    scales: {{
      x: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#1e293b' }} }},
      y: {{
        ticks: {{ color:'#64748b', callback: v => (v >= 0 ? '+' : '')+v+'ms' }},
        grid: {{ color:'#1e293b' }},
        title: {{ display: true, text: 'Δ ms (+ = bedllama slower)', color:'#64748b' }},
      }},
    }},
  }},
}});

// ── Routing table ──────────────────────────────────────────────────────────
(function buildRouting() {{
  const thinkLevels = D.meta.thinking_levels;
  const cats = D.meta.categories;
  const sc_index = {{}};
  D.scenarios.forEach(sc => {{
    sc.models.forEach(m => {{
      sc_index[`${{sc.thinking}}|${{sc.category}}|${{m.label}}`] = m;
    }});
  }});
  if (!D.pairs || D.pairs.length === 0) return;
  let html='<thead><tr><th>Model pair</th><th>Thinking</th>' +
    cats.map(c=>`<th class="num">${{c}} Δms</th><th class="num">${{c}} Δ%</th>`).join('') + '</tr></thead><tbody>';
  D.pairs.forEach(p => {{
    thinkLevels.forEach(think => {{
      html += '<tr>';
      html += `<td><strong>${{p.name}}</strong></td><td><code>${{think}}</code></td>`;
      cats.forEach(cat => {{
        const ebl = sc_index[`${{think}}|${{cat}}|${{p.bedllama}}`];
        const ebd = sc_index[`${{think}}|${{cat}}|${{p.direct}}`];
        const mbl = ebl?.stats?.mean_ms; const mbd = ebd?.stats?.mean_ms;
        if (!mbl || !mbd) {{ html += '<td class="num">—</td><td class="num">—</td>'; return; }}
        const d = mbl - mbd;
        const pct = Math.round(d / mbd * 100);
        const cls = d < 0 ? 'fast' : d > 1000 ? 'slow' : 'mid';
        const sign = d >= 0 ? '+' : '';
        html += `<td class="num"><span class="badge ${{cls}}">${{sign}}${{d}}ms</span></td>`;
        html += `<td class="num"><span class="badge ${{cls}}">${{sign}}${{pct}}%</span></td>`;
      }});
      html += '</tr>';
    }});
  }});
  html += '</tbody>';
  document.getElementById('routingTable').innerHTML = html;
}})();

// ── Scenarios tab ──────────────────────────────────────────────────────────
(function buildScenarios() {{
  const tabsEl = document.getElementById('scenarioTabs');
  const contentEl = document.getElementById('scenarioContent');
  D.scenarioCharts.forEach((sc, i) => {{
    const tab = document.createElement('div');
    tab.className = 'tab' + (i===0?' active':'');
    tab.textContent = sc.title;
    tab.onclick = () => {{
      tabsEl.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
      tab.classList.add('active');
      contentEl.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
      document.getElementById('sc-' + i).classList.add('active');
    }};
    tabsEl.appendChild(tab);

    const div = document.createElement('div');
    div.id = 'sc-' + i;
    div.className = 'tab-content' + (i===0?' active':'');

    // Find prompt for this scenario
    const matchSc = D.scenarios.find(s=>s.thinking===sc.title.split(' / thinking=')[1]&&s.category===sc.title.split('/')[0].toLowerCase().trim());
    const promptText = matchSc?.prompt || '';
    div.innerHTML = `<div class="prompt-box">${{promptText}}</div>
    <div class="chart-grid">
      <div>
        <h3>Response Time — bedllama vs direct</h3>
        <div class="chart-wrap"><canvas id="scChart_${{i}}"></canvas></div>
      </div>
      <div>
        <h3>Throughput (words/sec)</h3>
        <div class="chart-wrap"><canvas id="scWpsChart_${{i}}"></canvas></div>
      </div>
    </div>
    <table id="scTable_${{i}}" style="margin-top:.75rem"></table>`;
    contentEl.appendChild(div);

    setTimeout(() => {{
      // Pair-grouped horizontal bar: solid=bedllama, light=direct, same hue per family
      new Chart(document.getElementById('scChart_'+i), {{
        type:'bar',
        data:{{ labels: sc.labels, datasets:[{{ label:'mean ms', data:sc.mean,
          backgroundColor:sc.colors, borderColor:sc.colors.map(c=>c.replace('0.85','1').replace('0.75','1')), borderWidth:2 }}] }},
        options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{display:false}},
            tooltip:{{callbacks:{{label:ctx=>' '+ctx.raw+'ms'}}}} }},
          scales:{{ x:{{ticks:{{color:'#64748b',callback:v=>v+'ms'}},grid:{{color:'#1e293b'}}}},
                    y:{{ticks:{{color:'#94a3b8',font:{{size:11}}}},grid:{{color:'#1e293b'}}}} }} }},
      }});
      new Chart(document.getElementById('scWpsChart_'+i), {{
        type:'bar',
        data:{{ labels: sc.labels, datasets:[{{ label:'words/sec', data:sc.wps,
          backgroundColor:sc.colors, borderColor:sc.colors.map(c=>c.replace('0.85','1').replace('0.75','1')), borderWidth:2 }}] }},
        options:{{ indexAxis:'y', responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{display:false}},
            tooltip:{{callbacks:{{label:ctx=>' '+ctx.raw+' w/s'}}}} }},
          scales:{{ x:{{ticks:{{color:'#64748b',callback:v=>v+' w/s'}},grid:{{color:'#1e293b'}}}},
                    y:{{ticks:{{color:'#94a3b8',font:{{size:11}}}},grid:{{color:'#1e293b'}}}} }} }},
      }});
      // table
      let thtml='<thead><tr><th>#</th><th>Model</th><th class="num">Min</th><th class="num">Mean</th><th class="num">P95</th><th class="num">σ</th><th class="num">Words</th><th class="num">Tok≈</th><th class="num">w/s</th><th class="num">tok/s</th><th class="num">Err</th></tr></thead><tbody>';
      const ranked_models = [...(matchSc?.models||[])].sort((a,b)=>(a.stats?.mean_ms||999999)-(b.stats?.mean_ms||999999));
      ranked_models.forEach((m,ri) => {{
        const s = m.stats||{{}};
        thtml += `<tr class="${{ri===0&&!m.error_count?'winner-row':''}}"><td>${{ri+1}}${{ri===0&&!m.error_count?' 🥇':''}}</td><td>${{m.label}}</td><td class="num">${{s.min_ms||'—'}}ms</td><td class="num"><strong>${{s.mean_ms||'—'}}ms</strong></td><td class="num">${{s.p95_ms||'—'}}ms</td><td class="num">${{s.stdev_ms||'—'}}ms</td><td class="num">${{s.mean_word_count||'—'}}</td><td class="num">${{s.mean_tokens_est||'—'}}</td><td class="num">${{s.mean_words_per_sec||'—'}}</td><td class="num">${{s.mean_tokens_per_sec||'—'}}</td><td class="num">${{m.error_count||0}}</td></tr>`;
      }});
      thtml += '</tbody>';
      document.getElementById('scTable_'+i).innerHTML = thtml;
    }}, 50);
  }});
}})();
const THABBREV = __THABBREV_JSON__;

// ── Throughput tab ─────────────────────────────────────────────────────────
(function buildWps() {{
  const heads = D.colHeads;
  let html='<thead><tr><th>Model</th>'+heads.map(h=>`<th class="num">${{h}}</th>`).join('')+'</tr></thead><tbody>';
  const all=[]; D.wpsDatasets.forEach(ds=>ds.data.forEach(v=>v&&all.push(v)));
  const mn=Math.min(...all), mx=Math.max(...all);
  const winnersPer = heads.map((_,ci)=>{{
    let best=-Infinity,bestLbl='';
    D.wpsDatasets.forEach(ds=>{{ if(ds.data[ci]!=null&&ds.data[ci]>best){{best=ds.data[ci];bestLbl=ds.label;}} }});
    return bestLbl;
  }});
  D.wpsDatasets.forEach(ds=>{{
    html+='<tr><td><span class="legend-dot" style="background:'+ds.borderColor+'"></span>'+ds.label+'</td>';
    ds.data.forEach((v,ci)=>{{
      const isW=ds.label===winnersPer[ci];
      const norm=mx>mn?(v-mn)/(mx-mn):0;
      const cls=isW?'fast':norm>0.66?'fast':norm<0.33?'slow':'mid';
      html+=`<td class="num"><span class="badge ${{cls}}">${{v!=null?v+' w/s':'—'}}</span>${{isW?' 🥇':''}}</td>`;
    }});
    html+='</tr>';
  }});
  html+='</tbody>';
  document.getElementById('wpsTable').innerHTML=html;
}})();
new Chart(document.getElementById('wpsChart'), {{
  type:'bar',
  data:{{ labels: D.colHeads, datasets: D.wpsDatasets }},
  options:{{
    responsive:true, maintainAspectRatio:false,
    plugins:{{ legend:{{ labels:{{ color:'#94a3b8', boxWidth:12 }} }},
               tooltip:{{ callbacks:{{ label:ctx=>` ${{ctx.dataset.label}}: ${{ctx.raw}} w/s` }} }} }},
    scales:{{
      x:{{ ticks:{{ color:'#64748b' }}, grid:{{ color:'#1e293b' }} }},
      y:{{ ticks:{{ color:'#64748b', callback:v=>v+' w/s' }}, grid:{{ color:'#1e293b' }} }},
    }},
  }},
}});

// ── Thinking tab ───────────────────────────────────────────────────────────
(function buildThinking() {{
  const cats = D.meta.categories;
  let html='<thead><tr><th>Model</th><th>Δ Level</th>'+cats.map(c=>`<th class="num">${{c}}</th>`).join('')+'</tr></thead><tbody>';
  D.thinkingDeltaDs.forEach(ds=>{{
    html+='<tr><td>'+ds.label.split(' (')[0]+'</td><td><code>'+ds.label.split(' (')[1].replace(')','')+'</code></td>';
    ds.data.forEach(v=>{{
      if(v==null){{html+='<td class="num">—</td>';return;}}
      const cls=v<0?'fast':v>1000?'slow':'mid';
      html+=`<td class="num"><span class="badge ${{cls}}">${{v>=0?'+':''}}${{v}}ms</span></td>`;
    }});
    html+='</tr>';
  }});
  html+='</tbody>';
  document.getElementById('thinkingTable').innerHTML=html;
}})();
if (D.thinkingDeltaDs.length > 0) {{
  new Chart(document.getElementById('thinkingChart'), {{
    type:'bar',
    data:{{ labels: D.meta.categories, datasets: D.thinkingDeltaDs }},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{ labels:{{ color:'#94a3b8', boxWidth:12 }} }},
                 tooltip:{{ callbacks:{{ label:ctx=>` ${{ctx.dataset.label}}: ${{ctx.raw>=0?'+':''}}${{ctx.raw}}ms` }} }} }},
      scales:{{
        x:{{ ticks:{{ color:'#64748b' }}, grid:{{ color:'#1e293b' }} }},
        y:{{ ticks:{{ color:'#64748b', callback:v=>(v>=0?'+':'')+v+'ms' }}, grid:{{ color:'#1e293b' }},
             title:{{ display:true, text:'Δ ms (positive = thinking is slower)', color:'#64748b' }} }},
      }},
    }},
  }});
}}

// ── Raw runs tab ───────────────────────────────────────────────────────────
(function buildRuns() {{
  const el = document.getElementById('rawRuns');
  D.scenarios.forEach(sc => {{
    const section = document.createElement('div');
    section.className = 'card';
    section.style.margin = '.5rem 0';
    const prompt_short = sc.prompt.length > 80 ? sc.prompt.slice(0,80)+'…' : sc.prompt;
    let html = `<h3>${{sc.category}} / thinking=${{sc.thinking}}</h3>
      <div class="prompt-box">${{prompt_short}}</div>`;
    sc.models.forEach(m => {{
      html += `<details><summary>${{m.label}} — ${{m.runs.length}} runs · mean ${{m.stats?.mean_ms||'?'}}ms · ${{m.stats?.mean_words_per_sec||'?'}} w/s</summary>
        <table><thead><tr><th>#</th><th class="num">ms</th><th class="num">words</th><th class="num">chars</th><th class="num">w/s</th><th>response</th></tr></thead><tbody>`;
      m.runs.forEach(r => {{
        const wps = r.elapsed_ms>0&&!r.error ? (r.word_count*1000/r.elapsed_ms).toFixed(1) : '—';
        const first = (r.response||'').split('\\n')[0].slice(0,100);
        const statusCls = r.error ? 'slow' : 'fast';
        html += `<tr><td>${{r.run}}</td><td class="num"><span class="badge ${{statusCls}}">${{r.elapsed_ms}}ms</span></td>
          <td class="num">${{r.word_count}}</td><td class="num">${{r.char_count}}</td>
          <td class="num">${{wps}}</td><td style="font-size:.78rem;color:#94a3b8">${{first}}</td></tr>`;
      }});
      html += '</tbody></table></details>';
    }});
    section.innerHTML = html;
    el.appendChild(section);
  }});
}})();
</script>
</body>
</html>
"""
    return (html
        .replace("{{", "{")
        .replace("}}", "}")
        .replace(_HTML_DATA_PLACEHOLDER, json.dumps(embedded_data, ensure_ascii=False))
        .replace(_HTML_THABB_PLACEHOLDER, json.dumps(THINKING_ABBREV, ensure_ascii=False))
        .replace("__RUN_DATE__", meta["run_date"][:10])
    )

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_model_spec(spec: str) -> dict:
    path, _, label = spec.rpartition(":")
    if not path: path, label = spec, spec
    provider, _, model_id = path.partition("/")
    if not model_id: raise ValueError(f"Expected 'provider/model-id[:Label]', got: {spec!r}")
    return {"provider": provider, "model_id": model_id, "label": label or model_id}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bedllama multi-dimensional benchmark")
    p.add_argument("--iterations",    type=int, default=int(os.environ.get("BENCHMARK_ITERATIONS","10")), metavar="N")
    p.add_argument("--thinking",      default=os.environ.get("BENCHMARK_THINKING_LEVELS", os.environ.get("BENCHMARK_THINKING","off")), metavar="LEVELS")
    p.add_argument("--categories",    default=os.environ.get("BENCHMARK_CATEGORIES","short,average,long"), metavar="CATS")
    p.add_argument("--prompt-short",  default=os.environ.get("BENCHMARK_PROMPT_SHORT"))
    p.add_argument("--prompt-average",default=os.environ.get("BENCHMARK_PROMPT_AVERAGE"))
    p.add_argument("--prompt-long",   default=os.environ.get("BENCHMARK_PROMPT_LONG"))
    p.add_argument("--output-dir",    default=os.environ.get("BENCHMARK_OUTPUT_DIR", str(Path(__file__).parent.parent/"tmp")), metavar="DIR")
    p.add_argument("--models-file",   metavar="FILE")
    p.add_argument("--add-model",     action="append", dest="add_models", default=[], metavar="SPEC")
    p.add_argument("--warmup",        action="store_true", default=os.environ.get("BENCHMARK_WARMUP","").lower() in ("1","true","yes"))
    p.add_argument("--dry-run",       action="store_true", default=os.environ.get("BENCHMARK_DRY_RUN","").lower() in ("1","true","yes"))
    return p.parse_args()

# ── Progress ──────────────────────────────────────────────────────────────────

def eta_str(elapsed_s: float, done: int, total: int) -> str:
    if done == 0: return "…"
    rem = elapsed_s / done * (total - done)
    return f"{int(rem)//60}m{int(rem)%60:02d}s"

def get_pi_version() -> str:
    try:
        r = subprocess.run(["pi","--version"], capture_output=True, text=True, timeout=5)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception: return "unknown"

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    thinking_levels = [t.strip() for t in args.thinking.split(",") if t.strip()]
    categories      = [c.strip() for c in args.categories.split(",") if c.strip()]

    unknown = [c for c in categories if c not in PROMPT_CATEGORIES]
    if unknown:
        print(f"Error: unknown categories {unknown}. Valid: {list(PROMPT_CATEGORIES)}", file=sys.stderr)
        return 1

    prompt_cats = {k: dict(v) for k, v in PROMPT_CATEGORIES.items()}
    if args.prompt_short:   prompt_cats["short"]["prompt"]   = args.prompt_short
    if args.prompt_average: prompt_cats["average"]["prompt"] = args.prompt_average
    if args.prompt_long:    prompt_cats["long"]["prompt"]    = args.prompt_long

    if args.models_file:
        with open(args.models_file) as f: models = json.load(f)
    else:
        models = list(DEFAULT_MODELS)
    for spec in args.add_models:
        try: models.append(parse_model_spec(spec))
        except ValueError as e: print(f"Error: {e}", file=sys.stderr); return 1

    if not args.dry_run:
        try: subprocess.run(["pi","--version"], capture_output=True, check=False)
        except FileNotFoundError: print("Error: 'pi' not found in PATH", file=sys.stderr); return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pi_version = get_pi_version()
    run_date   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_runs = len(models) * len(thinking_levels) * len(categories) * args.iterations

    W = 66
    print(); print("═"*W); print("  bedllama multi-dimensional benchmark"); print("═"*W)
    print(f"  Thinking levels  : {', '.join(thinking_levels)}")
    print(f"  Prompt categories: {', '.join(categories)}")
    print(f"  Iterations/cell  : {args.iterations}")
    print(f"  Models           : {len(models)}")
    print(f"  Total runs       : {total_runs}")
    if args.warmup:  print("  Warmup           : enabled")
    if args.dry_run: print("  Mode             : DRY RUN")
    print(f"  Output           : {output_dir}")
    for m in models: print(f"    • {m['label']}  ({m['provider']}/{m['model_id']})")
    print("═"*W); print()

    scenarios:  list[dict] = []
    run_start   = time.monotonic()
    runs_done   = 0
    scenario_list = list(itertools.product(thinking_levels, categories))

    for sc_idx, (think, cat) in enumerate(scenario_list, 1):
        prompt   = prompt_cats[cat]["prompt"]
        abbrev_t = THINKING_ABBREV.get(think, think)
        abbrev_c = CATEGORY_ABBREV[cat]
        elapsed  = time.monotonic() - run_start
        print(f"{'─'*W}")
        print(f"  Scenario {sc_idx}/{len(scenario_list)}: {cat} / thinking={think}  [{abbrev_c}/{abbrev_t}]"
              f"  (ETA: {eta_str(elapsed, runs_done, total_runs)})")
        print(f"  Prompt: {prompt[:80]}{'…' if len(prompt)>80 else ''}")
        print(f"{'─'*W}")

        sc_models: list[dict] = []

        for m in models:
            provider, model_id, label = m["provider"], m["model_id"], m["label"]
            print(f"\n  ▶  {label}")
            print(f"     ┌{'─'*54}")

            if args.warmup:
                print(f"     │ warmup  …", end="", flush=True)
                run_pi(provider, model_id, prompt, think, args.dry_run)
                print(" done")

            runs: list[dict] = []
            for i in range(1, args.iterations + 1):
                print(f"     │ run {i:2d}/{args.iterations}  ", end="", flush=True)
                elapsed_ms, response = run_pi(provider, model_id, prompt, think, args.dry_run)
                error      = looks_like_error(response) or elapsed_ms < 0
                word_count = len(response.split())
                char_count = len(response)
                first_line = response.splitlines()[0] if response else ""
                preview    = first_line[:54] + ("…" if len(first_line) > 54 else "")
                wps        = round(word_count * 1000 / elapsed_ms, 1) if elapsed_ms > 0 and not error else 0
                if error: print(f"{'ERROR':>8}   ↳ {preview}")
                else:     print(f"{elapsed_ms:>8}ms  ↳ {preview}  [{word_count}w {wps}w/s]")
                runs.append({"run": i, "elapsed_ms": elapsed_ms, "response": response,
                             "word_count": word_count, "char_count": char_count, "error": error})
                runs_done += 1

            valid = [r["elapsed_ms"] for r in runs if not r["error"] and r["elapsed_ms"]>0]
            valid_w = [r["word_count"] for r in runs if not r["error"] and r["elapsed_ms"]>0]
            stats = compute_stats(valid, valid_w)
            errs  = sum(1 for r in runs if r["error"])
            print(f"     └{'─'*54}")
            if stats:
                print(f"     ↳ mean={stats['mean_ms']}ms · min={stats['min_ms']}ms · "
                      f"max={stats['max_ms']}ms · p95={stats['p95_ms']}ms · "
                      f"σ={stats['stdev_ms']}ms · {stats.get('mean_words_per_sec','?')}w/s")
            if errs: print(f"     ⚠  {errs}/{args.iterations} errors")

            sc_models.append({"provider": provider, "model_id": model_id,
                               "label": label, "runs": runs})

        print()
        scenarios.append({"thinking": think, "category": cat, "prompt": prompt,
                          "models": sc_models})

    # Write output
    print("Writing results…", end="", flush=True)
    meta = {
        "run_date": run_date, "iterations": args.iterations,
        "thinking_levels": thinking_levels,
        "prompt_categories": {k: v for k,v in prompt_cats.items() if k in categories},
        "categories": categories, "models": models, "warmup": args.warmup,
        "pi_version": pi_version, "total_runs": total_runs,
        "wall_time_s": round(time.monotonic() - run_start),
    }
    data = build_json(scenarios, meta)
    json_path = output_dir / "results.json"
    md_path   = output_dir / "results.md"
    html_path = output_dir / "results.html"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    md_path.write_text(build_markdown(data))
    html_path.write_text(build_html(data))
    wall = meta["wall_time_s"]
    print(f" done.  ({wall//60}m{wall%60:02d}s)")
    print(f"\n  📄  {json_path}\n  📄  {md_path}\n  🌐  {html_path}\n")

    # Terminal summary
    all_col_keys = list(itertools.product(categories, thinking_levels))
    lookup2: dict[tuple, dict] = {}
    for sc in data["scenarios"]:
        for m2 in sc["models"]:
            lookup2[(sc["thinking"], sc["category"], m2["label"])] = m2

    hdr_parts = [f"{CATEGORY_ABBREV[c]}/{THINKING_ABBREV.get(t,t)}" for c,t in all_col_keys]
    print(f"  {'Model':<36}" + "".join(f"  {h:>7}" for h in hdr_parts))
    print("  " + "─"*(36 + 9*len(all_col_keys)))
    for m in models:
        lbl = m["label"]
        cells = []
        for cat, think in all_col_keys:
            entry = lookup2.get((think, cat, lbl))
            mean  = entry.get("stats", {}).get("mean_ms") if entry else None
            cells.append(f"{mean:>7}" if mean else "      —")
        print(f"  {lbl:<36}" + "".join(f"  {c}" for c in cells))
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
