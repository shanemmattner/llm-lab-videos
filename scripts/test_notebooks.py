# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "nbclient>=0.10",
#     "jupyter-client>=8.0",
#     "ipykernel>=6.0",
#     "openai>=1.0",
#     "psutil",
#     "markdown",
#     "mlx",
# ]
# ///

"""
Test harness that executes Jupyter notebooks cell-by-cell with validation.

Usage:
    uv run scripts/test_notebooks.py                          # Test all notebooks
    uv run scripts/test_notebooks.py path/to/notebook.ipynb  # Test one notebook
    uv run scripts/test_notebooks.py --dry-run               # Validate config only
    uv run scripts/test_notebooks.py --strict                # Warn on unconfigured cells
    uv run scripts/test_notebooks.py --validate-ids          # Check config IDs exist in notebook
"""

import argparse
import html.parser
import re
import socket
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# ---------------------------------------------------------------------------
# Notebook configuration
# ---------------------------------------------------------------------------

NOTEBOOK_CONFIGS: dict[str, dict[str, dict]] = {
    "01-mlx-inference.ipynb": {
        "cell-cover": {"type": "static"},
        "cell-setup-check": {"type": "static"},
        "cell-warmup": {
            "type": "live",
            "needs_ports": "any",
            "expect": {
                "output_contains": ["models ready"],
                "html_contains": ["Model", "Port"],
            },
        },
        "cell-helpers": {
            "type": "static",
            "expect": {
                "output_contains": ["Helpers loaded"],
            },
        },
        "cell-first-inference": {"type": "live", "needs_ports": "any"},
        "cell-performance": {"type": "live", "needs_ports": "any"},
        "cell-memory": {"type": "static"},
        "cell-temperature": {"type": "live", "needs_ports": "any"},
        "cell-pirate-arena": {"type": "live", "needs_ports": "any"},
        "cell-personas-122b": {"type": "live", "needs_ports": [8800]},
        "cell-quantization": {"type": "static"},
        "cell-stress-test": {"type": "live", "needs_ports": "any", "timeout": 300},
        "cell-kv-cache": {"type": "live", "needs_ports": "any", "timeout": 180},
    },
    "01b-model-datasheet.ipynb": {
        "cell-cover": {"type": "static"},
        "cell-setup": {"type": "live", "needs_ports": "any"},
        "cell-helpers": {"type": "static"},
        "cell-fetch-configs": {"type": "static"},  # HuggingFace fetch, no MLX needed
        "cell-hybrid-diagram": {"type": "static"},
        "cell-kv-cache-math": {"type": "static"},
        "cell-memory-budget": {"type": "static"},
        "cell-moe-comparison": {"type": "live", "needs_ports": "any"},
        "cell-moe-ask-models": {"type": "live", "needs_ports": "any"},
    },
    "01c-inference-optimization.ipynb": {
        "cell-cover": {"type": "static"},
        "cell-setup": {"type": "live", "needs_ports": "any"},
        "cell-helpers": {"type": "static"},
        "cell-bandwidth": {"type": "live", "needs_ports": "any"},
        "cell-speculative": {"type": "static"},
        "cell-prefix-cache": {"type": "live", "needs_ports": "any"},
        "cell-quant-formats": {"type": "static"},
        "cell-quant-quality": {"type": "live", "needs_ports": "any"},
        "cell-batching": {"type": "live", "needs_ports": "any"},
    },
}

MLX_PORTS = [8800, 8801, 8802]

# ---------------------------------------------------------------------------
# Pre-flight: port scanning
# ---------------------------------------------------------------------------

def check_port(port: int, timeout: float = 1.0) -> bool:
    """Return True if something is listening on the given localhost port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def scan_mlx_servers(ports: list[int] = MLX_PORTS) -> dict[int, bool]:
    """Scan ports and return {port: is_up} mapping."""
    results: dict[int, bool] = {}
    for port in ports:
        results[port] = check_port(port)
    return results


def print_port_status(port_status: dict[int, bool]) -> None:
    print("Scanning MLX servers...")
    labels = {8800: "8800", 8801: "8801", 8802: "8802"}
    for port, up in port_status.items():
        label = labels.get(port, "?")
        status = f"UP ({label})" if up else "DOWN"
        print(f"  Port {port}: {status}")
    print()


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def should_skip(config: dict, available_ports: set[int]) -> bool:
    """Return True if a cell should be skipped given available ports."""
    needs_ports = config.get("needs_ports")
    if needs_ports is None:
        return False
    if needs_ports == "any":
        return len(available_ports) == 0
    # needs_ports is a list of specific ports
    required = set(needs_ports)
    return not required.issubset(available_ports)


# ---------------------------------------------------------------------------
# Output validation helpers
# ---------------------------------------------------------------------------

# VOID_ELEMENTS: self-closing HTML tags that do not need a closing tag
_VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}

# Common paired HTML elements we track for balance
_PAIRED_ELEMENTS = {
    "div", "span", "p", "table", "thead", "tbody", "tr", "th", "td",
    "ul", "ol", "li", "a", "b", "strong", "em", "i", "h1", "h2", "h3",
    "h4", "h5", "h6", "section", "article", "header", "footer", "main",
    "nav", "aside", "figure", "figcaption", "blockquote", "pre", "code",
    "form", "label", "button", "select", "option", "textarea", "script",
    "style", "html", "head", "body",
}


class _HTMLTagBalanceParser(html.parser.HTMLParser):
    """Tracks open/close counts for common paired HTML elements."""

    def __init__(self) -> None:
        super().__init__()
        self.open_counts: dict[str, int] = {}
        self.close_counts: dict[str, int] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in _PAIRED_ELEMENTS:
            self.open_counts[tag] = self.open_counts.get(tag, 0) + 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _PAIRED_ELEMENTS:
            self.close_counts[tag] = self.close_counts.get(tag, 0) + 1

    def unbalanced_tags(self) -> list[str]:
        """Return list of tag names that have mismatched open/close counts."""
        all_tags = set(self.open_counts) | set(self.close_counts)
        unbalanced = []
        for tag in sorted(all_tags):
            opens = self.open_counts.get(tag, 0)
            closes = self.close_counts.get(tag, 0)
            if opens != closes:
                unbalanced.append(f"<{tag}> opens={opens} closes={closes}")
        return unbalanced


def _validate_html_output(html_text: str) -> list[str]:
    """
    Parse HTML text and return a list of warning strings.
    Returns empty list if HTML appears well-formed.
    """
    warnings = []
    if not html_text.strip():
        warnings.append("HTML output is empty")
        return warnings
    parser = _HTMLTagBalanceParser()
    try:
        parser.feed(html_text)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"HTML parse error: {exc}")
        return warnings
    unbalanced = parser.unbalanced_tags()
    if unbalanced:
        for ub in unbalanced:
            warnings.append(f"Possibly malformed HTML — unbalanced tag: {ub}")
    return warnings


def _validate_cell_outputs(cell, config: dict, cell_id: str) -> list[str]:
    """
    Inspect cell.outputs and return a list of warning strings.
    Checks:
      1. HTML display_data: non-empty and tag-balanced
      2. Stream outputs: no traceback / Error: lines
      3. Per-cell 'expect' assertions (output_contains, html_contains)
    """
    warnings: list[str] = []
    outputs = getattr(cell, "outputs", []) or []

    # Collect stream text and HTML text for expect checks
    all_stream_text = ""
    all_html_text = ""

    for output in outputs:
        otype = output.get("output_type", "")

        # --- 1. HTML validation ---
        if otype == "display_data":
            data = output.get("data", {})
            html_text = data.get("text/html", "")
            if html_text:
                all_html_text += html_text
                html_warnings = _validate_html_output(html_text)
                for w in html_warnings:
                    warnings.append(f"[HTML] {w}")

        # --- 2. Stream output checks ---
        if otype == "stream":
            text = output.get("text", "")
            all_stream_text += text
            if "Traceback (most recent call last)" in text:
                warnings.append("[stream] Traceback detected in stream output")
            for line in text.splitlines():
                if re.match(r"^Error:", line):
                    warnings.append(f"[stream] Error line detected: {line.strip()[:200]}")

    # --- 3. Per-cell expect assertions ---
    expect = config.get("expect", {})

    output_contains = expect.get("output_contains", [])
    for needle in output_contains:
        if needle.lower() not in all_stream_text.lower():
            warnings.append(
                f"[expect] output_contains: {needle!r} not found in stream output"
            )

    html_contains = expect.get("html_contains", [])
    for needle in html_contains:
        if needle.lower() not in all_html_text.lower():
            warnings.append(
                f"[expect] html_contains: {needle!r} not found in HTML output"
            )

    return warnings


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------

def run_notebook(
    notebook_path: Path,
    port_status: dict[int, bool],
    dry_run: bool = False,
    kernel_name: str = "python3",
    strict: bool = False,
) -> dict:
    """
    Execute a single notebook cell-by-cell.

    Returns a dict with keys:
        notebook: str — notebook filename
        results: list of {cell_id, status, duration, error, warnings}
        counts: {pass, fail, skip, warn}
        unconfigured_cells: list of cell IDs not in NOTEBOOK_CONFIGS
    """
    available_ports = {port for port, up in port_status.items() if up}
    nb_name = notebook_path.name
    notebook_config = NOTEBOOK_CONFIGS.get(nb_name, {})

    print(f"=== {nb_name} ===", flush=True)

    nb = nbformat.read(str(notebook_path), as_version=4)

    results = []
    counts = {"pass": 0, "fail": 0, "skip": 0, "warn": 0}
    unconfigured_cells: list[str] = []

    if dry_run:
        # Just validate config without executing
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            cell_id = cell.get("id", f"cell-{i}")
            config = notebook_config.get(cell_id, {"type": "static"})
            skip = should_skip(config, available_ports)
            status = "SKIP" if skip else "DRY-RUN"
            print(f"  {cell_id:<30} {status}")
            if skip:
                counts["skip"] += 1
            else:
                counts["pass"] += 1
            results.append({"cell_id": cell_id, "status": status, "duration": 0.0, "error": None, "warnings": []})

            if strict and cell_id not in notebook_config:
                unconfigured_cells.append(cell_id)

        _print_counts(counts, nb_name)
        return {"notebook": nb_name, "results": results, "counts": counts, "unconfigured_cells": unconfigured_cells}

    # Live execution
    kernel_cwd = str(notebook_path.parent.resolve())
    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name=kernel_name,
        resources={"metadata": {"path": kernel_cwd}},
    )

    km = KernelManager(kernel_name=kernel_name)
    km.start_kernel(cwd=kernel_cwd)
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=30)
    client.km = km
    client.kc = kc

    try:

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            cell_id = cell.get("id", f"cell-{i}")
            config = notebook_config.get(cell_id, {"type": "static"})

            # --strict: track unconfigured cells
            if strict and cell_id not in notebook_config:
                unconfigured_cells.append(cell_id)

            # Skip?
            if should_skip(config, available_ports):
                _print_cell_result(cell_id, "SKIP", 0.0)
                counts["skip"] += 1
                results.append({"cell_id": cell_id, "status": "SKIP", "duration": 0.0, "error": None, "warnings": []})
                continue

            # Execute
            cell_timeout = config.get("timeout")
            if cell_timeout:
                old_timeout = client.timeout
                client.timeout = cell_timeout
            start = time.monotonic()
            try:
                client.execute_cell(cell, i)
                duration = time.monotonic() - start

                # --- Output validation (after success, before appending) ---
                cell_warnings = _validate_cell_outputs(cell, config, cell_id)
                for w in cell_warnings:
                    print(f"    WARN [{cell_id}]: {w}", flush=True)
                counts["warn"] += len(cell_warnings)

                _print_cell_result(cell_id, "PASS", duration)
                counts["pass"] += 1
                results.append({
                    "cell_id": cell_id,
                    "status": "PASS",
                    "duration": duration,
                    "error": None,
                    "warnings": cell_warnings,
                })
            except CellExecutionError as exc:
                duration = time.monotonic() - start
                # CellExecutionError has .traceback with the kernel-side error
                error_msg = getattr(exc, "traceback", str(exc))[-1000:]
                _print_cell_result(cell_id, "FAIL", duration, error=error_msg)
                counts["fail"] += 1
                results.append({"cell_id": cell_id, "status": "FAIL", "duration": duration, "error": error_msg, "warnings": []})
            except Exception as exc:  # noqa: BLE001
                duration = time.monotonic() - start
                error_msg = f"{type(exc).__name__}: {exc}"[:1000]
                _print_cell_result(cell_id, "FAIL", duration, error=error_msg)
                counts["fail"] += 1
                results.append({"cell_id": cell_id, "status": "FAIL", "duration": duration, "error": error_msg, "warnings": []})
            finally:
                if cell_timeout:
                    client.timeout = old_timeout

    finally:
        try:
            kc.stop_channels()
            km.shutdown_kernel()
        except Exception:  # noqa: BLE001
            pass

    _print_counts(counts, nb_name)
    return {"notebook": nb_name, "results": results, "counts": counts, "unconfigured_cells": unconfigured_cells}


def _print_cell_result(cell_id: str, status: str, duration: float, error: str | None = None) -> None:
    line = f"  {cell_id:<30} {status:<6}  ({duration:.1f}s)"
    print(line, flush=True)
    if error:
        # Indent error lines
        for err_line in error.splitlines()[-15:]:
            print(f"    ERROR: {err_line}", flush=True)


def _print_counts(counts: dict, nb_name: str) -> None:
    total = counts["pass"] + counts["fail"] + counts["skip"]
    passed = counts["pass"]
    failed = counts["fail"]
    skipped = counts["skip"]
    warned = counts.get("warn", 0)
    print()
    print(f"  {passed}/{total} passed, {failed} failed, {skipped} skipped, {warned} warnings")
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    print("=== Summary ===")
    for r in all_results:
        nb = r["notebook"]
        c = r["counts"]
        warned = c.get("warn", 0)
        print(
            f"  {nb:<45} {c['pass']} pass, {c['fail']} fail, {c['skip']} skip, {warned} warn"
        )


# ---------------------------------------------------------------------------
# Notebook discovery
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_notebooks() -> list[Path]:
    """Find all .ipynb files under sections/, excluding checkpoints."""
    sections_dir = REPO_ROOT / "sections"
    notebooks = sorted(
        p
        for p in sections_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )
    return notebooks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Jupyter notebooks cell-by-cell with validation."
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        metavar="NOTEBOOK",
        help="Notebook path(s) to test. If omitted, all notebooks are tested.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without executing cells.",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: python3). Use 'homebrew-py3' on Mac Studio.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Warn on code cells not listed in NOTEBOOK_CONFIGS. Exit 1 if any found.",
    )
    parser.add_argument(
        "--validate-ids",
        action="store_true",
        help="Check that every cell ID in NOTEBOOK_CONFIGS exists in the notebook. Exit 1 if stale IDs found.",
    )
    return parser.parse_args()


def validate_ids_for_notebook(notebook_path: Path) -> list[str]:
    """
    For each cell ID in NOTEBOOK_CONFIGS for this notebook, verify it exists
    in the notebook's actual code cells. Returns list of stale/missing IDs.
    """
    nb_name = notebook_path.name
    notebook_config = NOTEBOOK_CONFIGS.get(nb_name, {})
    if not notebook_config:
        return []

    nb = nbformat.read(str(notebook_path), as_version=4)
    actual_ids = {
        cell.get("id", f"cell-{i}")
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code"
    }

    stale = []
    for config_id in notebook_config:
        if config_id not in actual_ids:
            stale.append(config_id)
    return stale


def main() -> None:
    args = parse_args()

    # Resolve notebook paths
    if args.notebooks:
        notebook_paths = [Path(p).resolve() for p in args.notebooks]
        for p in notebook_paths:
            if not p.exists():
                print(f"ERROR: Notebook not found: {p}", file=sys.stderr)
                sys.exit(1)
    else:
        notebook_paths = find_notebooks()
        if not notebook_paths:
            print("No notebooks found under sections/.", file=sys.stderr)
            sys.exit(1)

    # --validate-ids: check config IDs against actual notebook cells
    if args.validate_ids:
        any_stale = False
        for nb_path in notebook_paths:
            stale = validate_ids_for_notebook(nb_path)
            if stale:
                any_stale = True
                for cell_id in stale:
                    print(
                        f"ERROR: [{nb_path.name}] Config cell ID {cell_id!r} not found in notebook cells.",
                        file=sys.stderr,
                    )
        if any_stale:
            sys.exit(1)

    # Pre-flight
    port_status = scan_mlx_servers()
    print_port_status(port_status)

    # Execute notebooks
    all_results = []
    for nb_path in notebook_paths:
        result = run_notebook(
            nb_path,
            port_status,
            dry_run=args.dry_run,
            kernel_name=args.kernel,
            strict=args.strict,
        )
        all_results.append(result)

    # --strict: report unconfigured cells across all notebooks
    if args.strict:
        any_unconfigured = False
        for result in all_results:
            nb_name = result["notebook"]
            for cell_id in result.get("unconfigured_cells", []):
                any_unconfigured = True
                print(
                    f"WARN [--strict]: [{nb_name}] Cell {cell_id!r} is not in NOTEBOOK_CONFIGS.",
                    flush=True,
                )
        if any_unconfigured:
            print()

    # Summary
    print_summary(all_results)

    # Exit code
    any_failed = any(r["counts"]["fail"] > 0 for r in all_results)
    strict_failed = args.strict and any(
        len(r.get("unconfigured_cells", [])) > 0 for r in all_results
    )
    sys.exit(1 if (any_failed or strict_failed) else 0)


if __name__ == "__main__":
    main()
