"""
Continue a Fable run that was cut off by the 128K output limit (stop_reason = max_tokens).

It resends the original prompt plus everything Fable has written so far, and asks it to
rewrite the cut-off file in full and then write the remaining files. This repeats until
Fable finishes (stop_reason = end_turn) or --max-continuations is reached.

Usage:
    python continue_run.py runs/<run_folder>
    python continue_run.py runs/<run_folder> --effort medium --max-continuations 2

Writes into the run folder:
    response_continuation_<n>.md     raw text of each continuation
    response_full.md                 original + all continuations, concatenated
    metadata_continuation_<n>.json   tokens, cost, stop reason for each continuation
and re-extracts all files into code/ (a file rewritten in a later turn replaces the earlier version).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from generate_fable_baseline import (
    EFFORT_LEVELS,
    MAX_TOKENS,
    MODEL,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    extract_files,
)

FILE_START_RE = re.compile(r"^<<<FILE\s+(?P<path>[^\n>]+?)\s*>>>", re.MULTILINE)
FILE_END = "<<<END FILE>>>"


def completed_and_open_files(text: str) -> tuple[list[str], str | None]:
    """Return (completed file paths, path of a file that was started but not finished)."""
    completed, open_file = [], None
    for m in FILE_START_RE.finditer(text):
        path = m.group("path").strip().strip("`'\"")
        rest = text[m.end():]
        next_start = FILE_START_RE.search(rest)
        segment = rest[: next_start.start()] if next_start else rest
        if FILE_END in segment:
            if path not in completed:
                completed.append(path)
            if open_file == path:  # a cut-off file that was later rewritten in full
                open_file = None
        else:
            open_file = path
    return completed, open_file


def build_continue_message(text_so_far: str) -> str:
    completed, open_file = completed_and_open_files(text_so_far)
    lines = [
        "Your previous response was cut off because it reached the maximum output length.",
        "Continue the same project from where you stopped, using exactly the same output format.",
        "",
        "Files already completed (do NOT repeat them): " + (", ".join(completed) if completed else "none"),
    ]
    if open_file:
        lines.append(f"File that was cut off mid-way: {open_file}. Rewrite this file in full, from its first line.")
    lines += [
        "Then write all remaining files of the project, each in full.",
        "Keep everything consistent with the interfaces and data structures already defined in the completed files.",
        "Be economical with planning: the design is already settled, so focus your output on writing the files.",
        "End with the short plain-text summary of the project structure.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue a truncated Fable run.")
    parser.add_argument("run_dir", help="Run folder created by generate_fable_baseline.py")
    parser.add_argument("--effort", choices=EFFORT_LEVELS, default=None,
                        help="Defaults to the effort recorded in the run's metadata.json")
    parser.add_argument("--max-continuations", type=int, default=3)
    parser.add_argument("--reextract", action="store_true",
                        help="Only re-extract code/ from the saved responses; no API call, no cost.")
    args = parser.parse_args()

    if args.reextract:
        run_dir = Path(args.run_dir)
        texts = [(run_dir / "response.md").read_text(encoding="utf-8")]
        k = 1
        while (run_dir / f"response_continuation_{k}.md").exists():
            texts.append("\n" + (run_dir / f"response_continuation_{k}.md").read_text(encoding="utf-8"))
            k += 1
        full_text = "".join(texts)
        (run_dir / "response_full.md").write_text(full_text, encoding="utf-8")
        files = extract_files(full_text, run_dir / "code")
        _, still_open = completed_and_open_files(full_text)
        print(f"Re-extracted {len(set(files))} file(s) into {run_dir / 'code'}")
        print(f"! Last file still incomplete: {still_open}" if still_open else "All files complete.")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable first.")

    run_dir = Path(args.run_dir)
    prompt = (run_dir / "prompt.md").read_text(encoding="utf-8")
    original = (run_dir / "response.md").read_text(encoding="utf-8")
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    effort = args.effort or meta.get("effort", "high")

    parts = [original]
    n = 1
    while (run_dir / f"response_continuation_{n}.md").exists():  # resume earlier continuations
        parts.append((run_dir / f"response_continuation_{n}.md").read_text(encoding="utf-8"))
        n += 1

    client = anthropic.Anthropic(timeout=60 * 60)
    total_cost = 0.0

    for _ in range(args.max_continuations):
        text_so_far = "".join(parts)
        continue_msg = build_continue_message(text_so_far)
        print(f"\n=== Continuation {n} (effort={effort})")
        print(continue_msg.splitlines()[3][:200])

        chunks: list[str] = []
        chars = 0
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": effort},
            messages=[
                {"role": "user", "content": prompt},
                # Assistant turns may not end in whitespace.
                {"role": "assistant", "content": text_so_far.rstrip()},
                {"role": "user", "content": continue_msg},
            ],
        ) as stream:
            for t in stream.text_stream:
                chunks.append(t)
                chars += len(t)
                if chars // 2000 != (chars - len(t)) // 2000:
                    print(f"  ...{chars:,} characters received", end="\r", flush=True)
            final = stream.get_final_message()

        cont_text = "".join(chunks)
        (run_dir / f"response_continuation_{n}.md").write_text(cont_text, encoding="utf-8")
        parts.append("\n" + cont_text)

        u = final.usage
        cost = (u.input_tokens * PRICE_INPUT_PER_MTOK + u.output_tokens * PRICE_OUTPUT_PER_MTOK) / 1_000_000
        total_cost += cost
        (run_dir / f"metadata_continuation_{n}.json").write_text(json.dumps({
            "model": final.model,
            "effort": effort,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "message_id": final.id,
            "stop_reason": final.stop_reason,
            "usage": {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens},
            "estimated_cost_usd": round(cost, 4),
            "continue_message": continue_msg,
        }, indent=2), encoding="utf-8")

        print(f"\n  {len(cont_text):,} characters, {u.input_tokens:,} in / {u.output_tokens:,} out "
              f"(~${cost:.2f}), stop_reason={final.stop_reason}")
        n += 1

        if final.stop_reason != "max_tokens":
            break
    else:
        print("\n! Still truncated after the maximum number of continuations. "
              "Run again to continue further, or re-run the prompt with --effort medium.")

    full_text = "".join(parts)
    (run_dir / "response_full.md").write_text(full_text, encoding="utf-8")
    files = extract_files(full_text, run_dir / "code")  # later versions of a file overwrite earlier ones
    _, still_open = completed_and_open_files(full_text)

    print(f"\nExtracted {len(set(files))} file(s) into {run_dir / 'code'}")
    if still_open:
        print(f"! Last file still incomplete: {still_open}")
    print(f"Cost of this continuation session: ~${total_cost:.2f}")


if __name__ == "__main__":
    main()
