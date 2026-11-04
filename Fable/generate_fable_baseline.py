"""
Generate a baseline PDF-to-text RAG pipeline with Claude Fable 5.1 via the Anthropic API.

Each run creates its own folder under runs/ containing:
  - prompt.md            the exact prompt that was sent
  - response.md          Fable's full raw text response
  - metadata.json        model, effort, timestamp, token usage, stop reason, prompt hash
  - code/                the extracted project files, ready to run

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."        # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sk-ant-..."

Usage:
    python generate_fable_baseline.py                       # one run, effort "high"
    python generate_fable_baseline.py --effort xhigh        # more thorough (and pricier)
    python generate_fable_baseline.py --runs 3              # several independent runs for robustness
    python generate_fable_baseline.py --prompt my_prompt.md
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

MODEL = "claude-fable-5-1"
MAX_TOKENS = 128_000  # Fable 5.1 max output; covers thinking + response text
EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"]

# Pricing per million tokens (USD) for the cost estimate. Check current pricing before relying on it.
PRICE_INPUT_PER_MTOK = 10.0
PRICE_OUTPUT_PER_MTOK = 50.0

FILE_BLOCK_RE = re.compile(
    # The body may not contain another "<<<FILE" start line, so a block that was cut off
    # (no END marker) is skipped instead of swallowing the next file.
    r"^<<<FILE\s+(?P<path>[^\n>]+?)\s*>>>\s*\n(?P<body>(?:(?!^<<<FILE\s).)*?)\n?<<<END FILE>>>",
    re.MULTILINE | re.DOTALL,
)


def extract_files(response_text: str, code_dir: Path) -> list[Path]:
    """Write each <<<FILE ...>>> block to code_dir, refusing paths that escape it."""
    written = []
    code_root = code_dir.resolve()
    for match in FILE_BLOCK_RE.finditer(response_text):
        rel_path = match.group("path").strip().strip("`'\"")
        target = (code_dir / rel_path).resolve()
        if code_root not in target.parents:
            print(f"  ! Skipping unsafe path: {rel_path}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(match.group("body") + "\n", encoding="utf-8")
        written.append(target)
    return written


def run_once(client: anthropic.Anthropic, prompt: str, effort: str, out_root: Path, run_idx: int) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{timestamp}_effort-{effort}_run{run_idx}"
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")

    print(f"\n=== Run {run_idx}: {MODEL}, effort={effort} -> {run_dir}")
    print("Streaming response (this can take several minutes)...")

    text_parts: list[str] = []
    chars = 0
    # Streaming is required for long generations; a non-streaming request
    # with a large max_tokens can time out.
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            text_parts.append(text)
            chars += len(text)
            if chars // 2000 != (chars - len(text)) // 2000:
                print(f"  ...{chars:,} characters received", end="\r", flush=True)
        final = stream.get_final_message()

    response_text = "".join(text_parts)
    (run_dir / "response.md").write_text(response_text, encoding="utf-8")
    print(f"\n  Response saved ({len(response_text):,} characters).")

    usage = final.usage
    est_cost = (
        usage.input_tokens * PRICE_INPUT_PER_MTOK + usage.output_tokens * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000

    metadata = {
        "model": final.model,
        "requested_model": MODEL,
        "effort": effort,
        "max_tokens": MAX_TOKENS,
        "timestamp_utc": timestamp,
        "message_id": final.id,
        "stop_reason": final.stop_reason,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
        "estimated_cost_usd": round(est_cost, 4),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "anthropic_sdk_version": anthropic.__version__,
    }

    if final.stop_reason == "max_tokens":
        print("  ! Hit max_tokens: output is truncated. Try a lower effort or ask for fewer/shorter files.")
    elif final.stop_reason == "refusal":
        print("  ! The model declined the request. See response.md for details.")

    files = extract_files(response_text, code_dir)
    metadata["extracted_files"] = [str(p.relative_to(code_dir.resolve())) for p in files]
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"  Extracted {len(files)} file(s) into {code_dir}")
    if not files:
        print("  ! No <<<FILE>>> blocks found. Check response.md; the model may not have followed the format.")
    print(
        f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out "
        f"(~${est_cost:.2f}), stop_reason={final.stop_reason}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Fable 5.1 baseline pipeline.")
    parser.add_argument("--prompt", default="prompt.md", help="Path to the prompt file.")
    parser.add_argument("--effort", default="high", choices=EFFORT_LEVELS)
    parser.add_argument("--runs", type=int, default=1, help="Number of independent generations.")
    parser.add_argument("--out", default="runs", help="Output directory.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable first.")

    prompt_path = Path(args.prompt)
    if not prompt_path.exists():
        sys.exit(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")

    if re.search(r"\[(describe|function signature|signature|output schema|version|paste)", prompt):
        print("Warning: the prompt still contains [placeholder] text. Fill it in for a fair comparison.")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted.")

    client = anthropic.Anthropic(timeout=60 * 60)  # long generations need a generous timeout
    out_root = Path(args.out)

    for i in range(1, args.runs + 1):
        try:
            run_once(client, prompt, args.effort, out_root, i)
        except anthropic.APIStatusError as e:
            print(f"  ! API error on run {i}: {e.status_code} {e.message}")
        except anthropic.APIConnectionError as e:
            print(f"  ! Connection error on run {i}: {e}")


if __name__ == "__main__":
    main()
