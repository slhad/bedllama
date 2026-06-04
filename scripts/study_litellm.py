#!/usr/bin/env python3
"""
study_litellm.py — Sequential LiteLLM config variants benchmark

Tests 4 LiteLLM proxy configurations in sequence, isolating each with a full
LiteLLM restart between runs.  Only tests bedllama-routed models (not Bedrock
direct) since those are the ones that go through LiteLLM.

Usage:
    python3 scripts/study_litellm.py [--iterations N] [--output-dir DIR]

Output per variant:
    tmp/study/variant_<name>.json   Raw benchmark data
    tmp/study/study_report.json     Comparison across all variants
    tmp/study/study_report.md       Human-readable summary
    tmp/study/study_report.html     Interactive comparison

Each variant is tested by a dedicated child agent that:
  1. Writes the LiteLLM config
  2. Restarts LiteLLM
  3. Runs a focused benchmark (long + average prompts, bedllama models only)
  4. Saves results
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── LiteLLM config variants ────────────────────────────────────────────────────

# Base config template (model list injected from the running LiteLLM config)
# Each variant adds or changes the general_settings / litellm_settings sections.

VARIANTS: list[dict] = [
    {
        "name":       "baseline",
        "label":      "Baseline (current config)",
        "description": "Unmodified bedllama config with adminUi=true: store_prompts_in_spend_logs=true, no litellm perf flags.",
        "general_settings_extra": {},
        "litellm_settings_extra": {},
    },
    {
        "name":       "no_stream_log",
        "label":      "disable_streaming_logging",
        "description": "Adds litellm_settings.disable_streaming_logging: true — skips per-chunk thread-pool dispatches (future.result() per token).",
        "general_settings_extra": {},
        "litellm_settings_extra": {"disable_streaming_logging": True},
    },
    {
        "name":       "no_spend",
        "label":      "disable_spend_logs",
        "description": "Adds general_settings.disable_spend_logs: true — stops the DB spend-log queue monitor and disables batch DB writes.",
        "general_settings_extra": {"disable_spend_logs": True, "store_prompts_in_spend_logs": False},
        "litellm_settings_extra": {},
    },
    {
        "name":       "lean",
        "label":      "Full lean config",
        "description": "Combines disable_streaming_logging + disable_spend_logs + proxy_batch_write_at:60 + background_health_checks:false.",
        "general_settings_extra": {
            "disable_spend_logs":        True,
            "store_prompts_in_spend_logs": False,
            "background_health_checks":  False,
            "proxy_batch_write_at":      60,
        },
        "litellm_settings_extra": {"disable_streaming_logging": True},
    },
]

# ── Paths ──────────────────────────────────────────────────────────────────────

LITELLM_CONFIG_PATH = Path.home() / ".cache" / "bedllama" / "litellm.config.yaml"
LITELLM_PID_PATH    = Path.home() / ".cache" / "bedllama" / "litellm.pid"
LITELLM_LOG_PATH    = Path.home() / ".cache" / "bedllama" / "litellm.log"
LITELLM_PORT        = int(os.environ.get("BEDLLAMA_LITELLM_PORT", "4001"))
LITELLM_API_KEY     = os.environ.get("BEDLLAMA_API_KEY", "sk-local")

# ── Benchmark models (bedllama only — the ones that go through LiteLLM) ────────

STUDY_MODELS = [
    {"provider": "bedllama", "model_id": "claude-haiku-4-5",  "label": "Haiku 4.5 (bedllama)"},
    {"provider": "bedllama", "model_id": "claude-sonnet-4-6", "label": "Sonnet 4.6 (bedllama)"},
]

# Benchmark only the prompts that show the most variance:
# - long (700+ words) — where streaming overhead dominates
# - average (~70 words) — mid-range reference
STUDY_CATEGORIES = {
    "average": "Explain what a REST API is and give one real-world example of when to use it. Keep your answer to 2-3 sentences.",
    "long":    "Write a detailed technical explanation of how HTTPS works. Cover: TLS handshake, certificate validation, symmetric key exchange, and why each step matters for security. Structure your answer with clear sections and use 4-6 paragraphs.",
}

# ── ANSI / error ──────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?<>]*[a-zA-Z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[=>]|\\.?)")
def strip_ansi(t: str) -> str:
    return _ANSI_RE.sub("", t).replace("\r","").replace("\x00","")
def clean_response(raw: str) -> str:
    lines = [l.rstrip() for l in strip_ansi(raw).splitlines() if l.strip()]
    return "\n".join(lines) if lines else "(empty)"
_ERR_RE = re.compile(r"(?i)no api key|validation error|authentication error|connection refused|command not found|^Error\b|^error:")
def looks_like_error(r: str) -> bool: return bool(_ERR_RE.search(r))

# ── Statistics ─────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    if not data: return 0.0
    s, n = sorted(data), len(data)
    if n == 1: return s[0]
    idx = (p/100)*(n-1); lo=int(idx); hi=lo+1
    return s[-1] if hi>=n else s[lo]+(idx-lo)*(s[hi]-s[lo])

def stats(times: list[float], words: list[float] | None = None) -> dict:
    if not times: return {}
    s: dict = {
        "count":     len(times),
        "min_ms":    round(min(times)),
        "max_ms":    round(max(times)),
        "mean_ms":   round(statistics.mean(times)),
        "median_ms": round(statistics.median(times)),
        "p95_ms":    round(percentile(times, 95)),
        "stdev_ms":  round(statistics.stdev(times)) if len(times)>1 else 0,
    }
    if words:
        wps = [w*1000/t for w,t in zip(words,times) if t>0]
        s["mean_words_per_sec"] = round(statistics.mean(wps),1) if wps else 0
    return s

# ── LiteLLM config manipulation ────────────────────────────────────────────────

def read_current_config() -> str:
    return LITELLM_CONFIG_PATH.read_text()

def extract_model_list(config_text: str) -> str:
    """Extract everything before 'general_settings:' as the model_list block."""
    idx = config_text.find("\ngeneral_settings:")
    return config_text[:idx].rstrip() if idx != -1 else config_text

def write_variant_config(variant: dict, base_config_text: str) -> None:
    """Write a modified LiteLLM config for this variant."""
    model_list = extract_model_list(base_config_text)

    # Parse existing general_settings from base
    lines = [model_list, ""]
    lines.append("general_settings:")
    # Always keep master_key and DB settings
    for line in base_config_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("master_key:") or \
           stripped.startswith("database_url:") or \
           stripped.startswith("store_model_in_db:"):
            lines.append(f"  {stripped}")
    # Add variant overrides
    for key, val in variant["general_settings_extra"].items():
        if isinstance(val, bool):
            lines.append(f"  {key}: {'true' if val else 'false'}")
        else:
            lines.append(f"  {key}: {val}")
    lines.append("")

    # litellm_settings from base (preserve ui_access_mode etc.)
    lines.append("litellm_settings:")
    in_litellm_settings = False
    for line in base_config_text.splitlines():
        if line.strip() == "litellm_settings:":
            in_litellm_settings = True; continue
        if in_litellm_settings:
            if line and not line[0].isspace(): break
            stripped = line.strip()
            if stripped and not any(k in stripped for k in variant["litellm_settings_extra"]):
                lines.append(f"  {stripped}")
    # Add variant overrides
    for key, val in variant["litellm_settings_extra"].items():
        if isinstance(val, bool):
            lines.append(f"  {key}: {'true' if val else 'false'}")
        else:
            lines.append(f"  {key}: {val}")
    lines.append("")

    # Preserve environment_variables block
    in_env = False
    env_lines = []
    for line in base_config_text.splitlines():
        if line.strip() == "environment_variables:":
            in_env = True; env_lines.append(line); continue
        if in_env:
            if line and not line[0].isspace(): break
            env_lines.append(line)
    if env_lines:
        lines.extend(env_lines)
        lines.append("")

    LITELLM_CONFIG_PATH.write_text("\n".join(lines))
    LITELLM_CONFIG_PATH.chmod(0o600)

def get_litellm_pid() -> int | None:
    try:
        pid = int(LITELLM_PID_PATH.read_text().strip())
        return pid if pid > 0 else None
    except Exception: return None

def is_running(pid: int | None) -> bool:
    if not pid: return False
    try: os.kill(pid, 0); return True
    except: return False

def stop_litellm() -> None:
    pid = get_litellm_pid()
    if is_running(pid):
        print(f"  ↳ stopping LiteLLM pid={pid}…", end="", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):
                time.sleep(0.5)
                if not is_running(pid): break
            if is_running(pid):
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
        except Exception: pass
        print(" stopped")
    else:
        print("  ↳ LiteLLM not running")

def start_litellm() -> int | None:
    print(f"  ↳ starting LiteLLM…", end="", flush=True)
    log_fd = LITELLM_LOG_PATH.open("a")
    proc = subprocess.Popen(
        ["litellm", "--config", str(LITELLM_CONFIG_PATH), "--port", str(LITELLM_PORT)],
        stdout=log_fd, stderr=log_fd,
        start_new_session=True,
    )
    LITELLM_PID_PATH.write_text(f"{proc.pid}\n")
    print(f" pid={proc.pid}")
    return proc.pid

def wait_for_litellm(timeout_s: int = 60) -> bool:
    url = f"http://127.0.0.1:{LITELLM_PORT}/v1/models"
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    print(f"  ↳ waiting for LiteLLM ready…", end="", flush=True)
    import urllib.request, urllib.error
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    print(f" ready ({round(deadline - time.monotonic() - timeout_s + (timeout_s - (deadline - time.monotonic())), 1)}s wait)")
                    return True
        except Exception: pass
        time.sleep(0.5)
    print(" TIMEOUT")
    return False

def restart_litellm_with_variant(variant: dict, base_config: str) -> bool:
    print(f"  Applying variant: {variant['label']}")
    write_variant_config(variant, base_config)
    stop_litellm()
    time.sleep(2)  # brief pause for port release
    start_litellm()
    return wait_for_litellm()

# ── Pi runner ─────────────────────────────────────────────────────────────────

def run_pi(provider: str, model_id: str, prompt: str) -> tuple[int, str]:
    cmd = ["pi","--provider",provider,"--model",model_id,"--no-tools","--no-session","--thinking","off","-p",prompt]
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed_ms = round((time.monotonic()-t0)*1000)
        raw = r.stdout + (r.stderr or "")
    except FileNotFoundError: return -1, "ERROR: pi not found"
    except Exception as e: return round((time.monotonic()-t0)*1000), f"ERROR: {e}"
    return elapsed_ms, clean_response(raw)

# ── Run one variant benchmark ──────────────────────────────────────────────────

def run_variant_benchmark(variant: dict, iterations: int) -> dict:
    result: dict = {
        "variant": variant["name"],
        "label":   variant["label"],
        "description": variant["description"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells": {},
    }

    for cat, prompt in STUDY_CATEGORIES.items():
        for m in STUDY_MODELS:
            key = f"{cat}|{m['label']}"
            print(f"\n  ▶  {m['label']}  /  {cat}")
            print(f"     ┌{'─'*50}")
            runs = []
            for i in range(1, iterations+1):
                print(f"     │ run {i:2d}/{iterations}  ", end="", flush=True)
                elapsed_ms, response = run_pi(m["provider"], m["model_id"], prompt)
                error = looks_like_error(response) or elapsed_ms < 0
                wc = len(response.split()); cc = len(response)
                wps = round(wc*1000/elapsed_ms, 1) if elapsed_ms > 0 and not error else 0
                first = response.splitlines()[0][:56] if response else ""
                if error: print(f"{'ERROR':>8}   ↳ {first}")
                else:     print(f"{elapsed_ms:>8}ms  ↳ {first}  [{wc}w {wps}w/s]")
                runs.append({"run":i,"elapsed_ms":elapsed_ms,"word_count":wc,"char_count":cc,"error":error})
            valid_t = [r["elapsed_ms"] for r in runs if not r["error"] and r["elapsed_ms"]>0]
            valid_w = [r["word_count"]  for r in runs if not r["error"] and r["elapsed_ms"]>0]
            st = stats(valid_t, valid_w)
            print(f"     └{'─'*50}")
            if st:
                print(f"     ↳ mean={st['mean_ms']}ms · p95={st['p95_ms']}ms · {st.get('mean_words_per_sec','?')}w/s")
            result["cells"][key] = {"model": m["label"], "category": cat,
                                     "prompt": prompt, "stats": st, "runs": runs}
    return result

# ── Comparison report ─────────────────────────────────────────────────────────

def build_comparison(all_results: list[dict]) -> dict:
    """Build a cross-variant comparison keyed by (model, category)."""
    keys = set()
    for r in all_results:
        keys.update(r["cells"].keys())

    comparison: dict[str, dict] = {}
    for key in sorted(keys):
        comparison[key] = {}
        for r in all_results:
            cell = r["cells"].get(key)
            if cell:
                comparison[key][r["variant"]] = cell["stats"]

    return comparison

def build_comparison_md(all_results: list[dict], comparison: dict) -> str:
    L: list[str] = [
        "# LiteLLM Config Study Results", "",
        "Sequential test of 4 LiteLLM proxy configurations. Same prompt, same models, "
        "full LiteLLM restart between variants.",
        "",
        "## Variants Tested", "",
    ]
    for r in all_results:
        L += [f"### `{r['variant']}` — {r['label']}", f"> {r['description']}", ""]

    L += ["---", "", "## Comparison Table (mean ms + words/sec)", "",
          "> 🥇 = fastest mean for that (model × category) cell", ""]

    # One table per category
    for cat in ["average", "long"]:
        L += [f"### Category: `{cat}`", ""]
        variant_names = [r["variant"] for r in all_results]
        L.append("| Model | " + " | ".join(f"{n} ms | {n} w/s" for n in variant_names) + " |")
        L.append("|:------|" + "|".join("---:|---:" for _ in variant_names) + "|")
        for m in STUDY_MODELS:
            key = f"{cat}|{m['label']}"
            comp = comparison.get(key, {})
            best_ms = min((comp[v].get("mean_ms",999999) for v in variant_names if v in comp), default=None)
            row_cells = []
            for v in variant_names:
                s = comp.get(v, {})
                ms_val = s.get("mean_ms")
                wps_val = s.get("mean_words_per_sec")
                ms_str  = f"**{ms_val}** 🥇" if ms_val == best_ms else str(ms_val or "—")
                row_cells.append(f"{ms_str} | {wps_val or '—'}")
            L.append(f"| {m['label']} | " + " | ".join(row_cells) + " |")
        L.append("")

    # Delta vs baseline
    L += ["---", "", "## Delta vs Baseline (ms, negative = faster)", ""]
    for cat in ["average", "long"]:
        L += [f"### `{cat}`", ""]
        variant_names = [r["variant"] for r in all_results if r["variant"] != "baseline"]
        L.append("| Model | " + " | ".join(f"Δ {n}" for n in variant_names) + " |")
        L.append("|:------|" + "|".join("------:" for _ in variant_names) + "|")
        for m in STUDY_MODELS:
            key = f"{cat}|{m['label']}"
            comp = comparison.get(key, {})
            base_ms = comp.get("baseline", {}).get("mean_ms")
            cells = []
            for v in variant_names:
                s = comp.get(v, {}); ms = s.get("mean_ms")
                if base_ms and ms:
                    d = ms - base_ms; pct = round(d/base_ms*100)
                    cells.append(f"{'+' if d>=0 else ''}{d}ms ({'+' if d>=0 else ''}{pct}%)")
                else: cells.append("—")
            L.append(f"| {m['label']} | " + " | ".join(cells) + " |")
        L.append("")

    # Winner
    best_variant_votes: dict[str, int] = {}
    for key, comp_data in comparison.items():
        best_ms = min((v.get("mean_ms",999999) for v in comp_data.values()), default=None)
        for vname, s in comp_data.items():
            if s.get("mean_ms") == best_ms:
                best_variant_votes[vname] = best_variant_votes.get(vname, 0) + 1
    winner = max(best_variant_votes, key=best_variant_votes.get) if best_variant_votes else "—"
    L += ["---", "", f"## 🏆 Recommendation", "",
          f"**Best overall:** `{winner}` (fastest in {best_variant_votes.get(winner,0)}/{len(comparison)} cells)", ""]

    # Applied config snippet
    winner_v = next((v for v in VARIANTS if v["name"] == winner), None)
    if winner_v:
        L += ["```yaml", "# Recommended litellm.config.yaml additions:", ""]
        if winner_v["general_settings_extra"]:
            L.append("general_settings:")
            for k, val in winner_v["general_settings_extra"].items():
                L.append(f"  {k}: {'true' if val is True else 'false' if val is False else val}")
        if winner_v["litellm_settings_extra"]:
            L.append("litellm_settings:")
            for k, val in winner_v["litellm_settings_extra"].items():
                L.append(f"  {k}: {'true' if val is True else 'false' if val is False else val}")
        L.append("```"); L.append("")

    L += ["---", "", "## Notes", "",
          "- All variants tested with `thinking=off`, 3 iterations per cell.",
          "- Full LiteLLM restart between variants (config hot-swap + SIGTERM + re-launch).",
          "- bedllama front server was NOT restarted between variants.",
          "- Times include pi startup (~1.4s) + bedllama front + LiteLLM + Bedrock.",
          "", f"*Generated by `scripts/study_litellm.py`*"]
    return "\n".join(L)+"\n"

# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Sequential LiteLLM config study")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output-dir", default=str(Path(__file__).parent.parent/"tmp"/"study"))
    p.add_argument("--skip-restart", action="store_true",
                   help="Skip LiteLLM restart (useful if you pre-applied the config)")
    p.add_argument("--variant", help="Only run one named variant (e.g. lean)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not LITELLM_CONFIG_PATH.exists():
        print(f"Error: LiteLLM config not found at {LITELLM_CONFIG_PATH}", file=sys.stderr)
        print("Run `bedllama start` first.", file=sys.stderr)
        return 1

    base_config = read_current_config()
    variants_to_run = [v for v in VARIANTS if args.variant is None or v["name"] == args.variant]
    total_variants  = len(variants_to_run)
    total_cells     = total_variants * len(STUDY_MODELS) * len(STUDY_CATEGORIES) * args.iterations

    print(); print("═"*60)
    print("  LiteLLM config study")
    print("═"*60)
    print(f"  Variants       : {total_variants}")
    print(f"  Models         : {len(STUDY_MODELS)}")
    print(f"  Categories     : {', '.join(STUDY_CATEGORIES.keys())}")
    print(f"  Iterations/cell: {args.iterations}")
    print(f"  Total runs     : {total_cells}")
    print(f"  Output         : {output_dir}")
    if args.skip_restart: print("  LiteLLM restart: SKIPPED")
    print("═"*60); print()

    all_results: list[dict] = []
    study_start = time.monotonic()

    for vi, variant in enumerate(variants_to_run, 1):
        W = 60
        print(f"\n{'═'*W}")
        print(f"  Variant {vi}/{total_variants}: {variant['label']}")
        print(f"  {variant['description']}")
        print(f"{'═'*W}")

        if not args.skip_restart:
            ok = restart_litellm_with_variant(variant, base_config)
            if not ok:
                print(f"  ⚠  LiteLLM failed to start for variant {variant['name']} — skipping")
                continue
            # Brief warmup after restart
            print("  ↳ brief warmup…", end="", flush=True)
            time.sleep(3)
            print(" done")
        else:
            print("  ↳ skip restart (using current LiteLLM config)")

        result = run_variant_benchmark(variant, args.iterations)
        all_results.append(result)

        # Save per-variant result
        variant_path = output_dir / f"variant_{variant['name']}.json"
        variant_path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
        print(f"\n  Saved: {variant_path}")

    # Restore original config (best-effort)
    if not args.skip_restart:
        print("\n  Restoring original LiteLLM config…", end="", flush=True)
        LITELLM_CONFIG_PATH.write_text(base_config)
        LITELLM_CONFIG_PATH.chmod(0o600)
        stop_litellm()
        time.sleep(1)
        start_litellm()
        wait_for_litellm()
        print("  Original config restored.")

    # Build comparison report
    print("\nBuilding comparison report…", end="", flush=True)
    comparison = build_comparison(all_results)
    wall_s = round(time.monotonic() - study_start)

    report = {
        "meta": {
            "run_date": datetime.now(timezone.utc).isoformat(),
            "iterations": args.iterations,
            "wall_time_s": wall_s,
        },
        "variants": [{"name": r["variant"], "label": r["label"],
                       "description": r["description"]} for r in all_results],
        "comparison": comparison,
        "raw": all_results,
    }

    report_json = output_dir / "study_report.json"
    report_md   = output_dir / "study_report.md"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    report_md.write_text(build_comparison_md(all_results, comparison))
    print(" done.")
    print(f"\n  📄  {report_json}")
    print(f"  📄  {report_md}")
    print(f"  ⏱   Total wall time: {wall_s//60}m{wall_s%60:02d}s\n")

    # Terminal summary
    print(f"\n  {'Model':<32}  {'Category':<9}  " +
          "  ".join(f"{r['variant'][:12]:>12}" for r in all_results))
    print("  " + "─"*(32+9+4+13*len(all_results)))
    for m in STUDY_MODELS:
        for cat in STUDY_CATEGORIES:
            key = f"{cat}|{m['label']}"
            best_ms = min((all_results[i]["cells"].get(key,{}).get("stats",{}).get("mean_ms",999999)
                          for i in range(len(all_results))), default=None)
            cells = []
            for r in all_results:
                ms = r["cells"].get(key,{}).get("stats",{}).get("mean_ms")
                s  = "🥇 " if ms == best_ms else "   "
                cells.append(f"{s}{ms or '—':>9}")
            print(f"  {m['label']:<32}  {cat:<9}  " + "  ".join(cells))
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
