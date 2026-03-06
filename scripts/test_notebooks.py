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
"""

import argparse
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
        "cell-warmup": {"type": "live", "needs_ports": "any"},
        "cell-helpers": {"type": "static"},
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
# Notebook execution
# ---------------------------------------------------------------------------

def run_notebook(
    notebook_path: Path,
    port_status: dict[int, bool],
    dry_run: bool = False,
    kernel_name: str = "python3",
) -> dict:
    """
    Execute a single notebook cell-by-cell.

    Returns a dict with keys:
        notebook: str — notebook filename
        results: list of {cell_id, status, duration, error}
        counts: {pass, fail, skip}
    """
    available_ports = {port for port, up in port_status.items() if up}
    nb_name = notebook_path.name
    notebook_config = NOTEBOOK_CONFIGS.get(nb_name, {})

    print(f"=== {nb_name} ===", flush=True)

    nb = nbformat.read(str(notebook_path), as_version=4)

    results = []
    counts = {"pass": 0, "fail": 0, "skip": 0}

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
            results.append({"cell_id": cell_id, "status": status, "duration": 0.0, "error": None})
        _print_counts(counts, nb_name)
        return {"notebook": nb_name, "results": results, "counts": counts}

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

            # Skip?
            if should_skip(config, available_ports):
                _print_cell_result(cell_id, "SKIP", 0.0)
                counts["skip"] += 1
                results.append({"cell_id": cell_id, "status": "SKIP", "duration": 0.0, "error": None})
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
                _print_cell_result(cell_id, "PASS", duration)
                counts["pass"] += 1
                results.append({"cell_id": cell_id, "status": "PASS", "duration": duration, "error": None})
            except CellExecutionError as exc:
                duration = time.monotonic() - start
                # CellExecutionError has .traceback with the kernel-side error
                error_msg = getattr(exc, "traceback", str(exc))[-1000:]
                _print_cell_result(cell_id, "FAIL", duration, error=error_msg)
                counts["fail"] += 1
                results.append({"cell_id": cell_id, "status": "FAIL", "duration": duration, "error": error_msg})
            except Exception as exc:  # noqa: BLE001
                duration = time.monotonic() - start
                error_msg = f"{type(exc).__name__}: {exc}"[:1000]
                _print_cell_result(cell_id, "FAIL", duration, error=error_msg)
                counts["fail"] += 1
                results.append({"cell_id": cell_id, "status": "FAIL", "duration": duration, "error": error_msg})
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
    return {"notebook": nb_name, "results": results, "counts": counts}


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
    print()
    print(f"  {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    print("=== Summary ===")
    for r in all_results:
        nb = r["notebook"]
        c = r["counts"]
        print(
            f"  {nb:<45} {c['pass']} pass, {c['fail']} fail, {c['skip']} skip"
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
    return parser.parse_args()


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

    # Pre-flight
    port_status = scan_mlx_servers()
    print_port_status(port_status)

    # Execute notebooks
    all_results = []
    for nb_path in notebook_paths:
        result = run_notebook(nb_path, port_status, dry_run=args.dry_run, kernel_name=args.kernel)
        all_results.append(result)

    # Summary
    print_summary(all_results)

    # Exit code
    any_failed = any(r["counts"]["fail"] > 0 for r in all_results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
