import re
import time
import html
import markdown
import threading
import socket
from IPython.display import HTML, display

# Color palette for N models (cycles if more than 8)
COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]

# Module-level state (set by init())
MODELS = []
clients = {}


def init(models, openai_clients):
    """Called by notebooks after port scanning to inject runtime state."""
    global MODELS, clients
    MODELS = models
    clients = openai_clients


def _port_open(port):
    """Check if a port is open on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


PORTS = [8800, 8801, 8802]

# Known sizes for sorting (largest first) — maps label prefix to sort weight
_SIZE_ORDER = {
    "122B": 0, "35B": 1, "27B": 2, "9B": 3, "8B": 4,
    "4B": 5, "2B": 6, "1.7B": 7, "0.8B": 8, "0.6B": 9, "0.5B": 10,
}

# Approximate memory footprints by label
MODEL_FOOTPRINTS = {
    "122B": "~65 GB", "35B": "~20 GB", "27B": "~15 GB", "9B": "~5 GB",
    "8B": "~4.5 GB", "4B": "~2.5 GB", "2B": "~1.2 GB", "1.7B": "~1 GB",
    "0.8B": "~0.5 GB", "0.6B": "~0.4 GB", "0.5B": "~0.3 GB",
}


def _label_from_model_id(model_id):
    """Derive a short label (e.g. '9B-A3B', '122B-A10B') from a model ID string."""
    name = model_id.split("/")[-1].lower()
    # Extract size — check longer patterns first to avoid "0.8b" matching "8b"
    import re as _re2
    size = None
    size_match = _re2.search(r'(?<![.\d])(122|35|27|9|8|4|2|1\.7|0\.8|0\.6|0\.5)b', name)
    if size_match:
        size = size_match.group(0).upper()
    if not size:
        return name[:12]
    # Extract MoE active param tag (e.g. A3B, A10B)
    import re as _re
    moe = _re.search(r'a(\d+b)', name)
    if moe:
        return f"{size}-{moe.group(0).upper()}"
    return size


def _get_model_id_from_process(port):
    """Get the --model argument from the mlx_lm process listening on a port."""
    import subprocess, os
    env = os.environ.copy()
    env["PATH"] = env.get("PATH", "") + ":/usr/sbin:/sbin"
    try:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-i", f":{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5, env=env,
        )
        if result.returncode != 0:
            return None
        pid = result.stdout.strip().split("\n")[0]
        result = subprocess.run(
            ["ps", "-p", pid, "-o", "args="],
            capture_output=True, text=True, timeout=5, env=env,
        )
        args = result.stdout.strip()
        # Parse --model from: python -m mlx_lm server --model <id> --port ...
        parts = args.split()
        for i, part in enumerate(parts):
            if part == "--model" and i + 1 < len(parts):
                return parts[i + 1]
    except Exception:
        pass
    return None


def _sort_key(label):
    """Sort key for model labels — largest first, unknown at end."""
    # Strip MoE suffix for lookup (e.g. "35B-A3B" → "35B")
    base = label.split("-")[0] if "-" in label else label
    return _SIZE_ORDER.get(base, 99)


def list_available_models(port=8800):
    """Query an MLX server's /v1/models endpoint to list all cached models.

    Returns list of dicts sorted by size (smallest first):
        [{"model": "full/model-id", "label": "35B-A3B"}, ...]
    """
    import urllib.request, json
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(f"Cannot reach MLX server on port {port}: {e}") from e
    models = []
    for m in data.get("data", []):
        model_id = m["id"]
        label = _label_from_model_id(model_id)
        models.append({"model": model_id, "label": label})
    models.sort(key=lambda m: _sort_key(m["label"]), reverse=True)
    return models


def discover_models(port=8800, model_ids=None):
    """Set up N models on a single MLX server port.

    If model_ids is None, discovers all available models via /v1/models.
    Returns (models_list, clients_dict) ready to pass to init().
    """
    from openai import OpenAI

    if model_ids is None:
        available = list_available_models(port)
        model_ids = [m["model"] for m in available]

    models = []
    _clients = {}
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
    for i, model_id in enumerate(model_ids):
        label = _label_from_model_id(model_id)
        color = COLORS[i % len(COLORS)]
        m = {"label": label, "model": model_id, "port": port, "color": color}
        models.append(m)
        _clients[label] = client  # All share same client (same port)
    return models, _clients


def discover_servers(ports=None):
    """Discover running MLX servers by checking each port.

    Gets the actual model ID from the process command line (not /v1/models,
    which lists all cached models and causes model-swap errors).

    Returns (models_list, clients_dict) ready to pass to init().
    """
    from openai import OpenAI

    ports = ports or PORTS
    models = []
    _clients = {}
    for port in ports:
        if not _port_open(port):
            continue
        model_id = _get_model_id_from_process(port)
        if not model_id:
            continue
        label = _label_from_model_id(model_id)
        m = {"label": label, "model": model_id, "port": port}
        models.append(m)
        _clients[label] = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
    # Sort by size (smallest first, left to right)
    models.sort(key=lambda m: _sort_key(m["label"]), reverse=True)
    for i, m in enumerate(models):
        m["color"] = COLORS[i % len(COLORS)]
    return models, _clients


def strip_think(text):
    """Remove <think>...</think> blocks from Qwen3.5 reasoning output."""
    if not text:
        return ""
    # Strip complete <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # Handle unclosed <think> tag — strip the tag and keep the content after it
    if '<think>' in cleaned:
        # Try removing just the tag, keep content
        no_tag = cleaned.replace('<think>', '').replace('</think>', '')
        if no_tag.strip():
            cleaned = no_tag
        else:
            # Even after removing tags, nothing left — return raw text minus tags
            cleaned = text.replace('<think>', '').replace('</think>', '')
    return cleaned.strip()


def _md(text):
    """Convert markdown text to HTML. Falls back to escaped text on error."""
    try:
        return markdown.markdown(text, extensions=["fenced_code", "tables"])
    except Exception:
        return html.escape(text)


def _render_cards(state, models_order):
    """Render N-column HTML cards from current state."""
    min_w = "200px" if len(models_order) > 4 else "250px"
    cards = ""
    for m in models_order:
        s = state[m["label"]]
        text = strip_think(s["text"])
        if not text and not s["done"]:
            rendered = "<em>waiting...</em>"
        elif not text and s["done"]:
            rendered = "<em>(empty response)</em>"
        else:
            rendered = _md(text)
        if s["done"]:
            ttft_ms = int(s["ttft"] * 1000) if s.get("ttft") is not None else None
            ttft_str = f"TTFT {ttft_ms}ms" if ttft_ms is not None else ""
            tps_str = f"{s['tps']:.1f} tok/s"
            # Show warning if strip_think produced empty from non-empty raw
            if not text and s["text"]:
                tps_str += " ⚠️ think-stripped"
            parts = [p for p in [ttft_str, tps_str] if p]
            status = " · ".join(parts)
        else:
            ttft = s.get("ttft")
            elapsed_str = f"{s['elapsed']:.1f}s" if s["elapsed"] > 0 else ""
            if ttft is not None:
                ttft_ms = int(ttft * 1000)
                status = f"streaming... TTFT: {ttft_ms}ms · {s['tokens']} tok {elapsed_str}"
            else:
                status = f"streaming... {s['tokens']} tok {elapsed_str}"
        cards += f"""
        <div style="flex:1; min-width:{min_w}; background:#f9fafb; border:1px solid #d1d5db;
                    border-left:4px solid {s['color']}; border-radius:0 8px 8px 0; padding:12px 18px;">
          <div style="color:{s['color']}; font-weight:bold; font-size:0.85em; margin-bottom:6px;">
            {m['label']} · {status}
          </div>
          <div style="color:#1f2937; line-height:1.6; word-wrap:break-word; font-size:0.9em;">
            {rendered}
          </div>
          <div style="color:#9ca3af; font-size:0.75em; margin-top:8px;">
            {s['tokens']} tokens in {s['elapsed']:.1f}s
          </div>
        </div>"""
    return f'<div style="display:flex; gap:16px; flex-wrap:wrap; margin:10px 0;">{cards}</div>'


def compare_models(prompt, system_prompt=None, **kwargs):
    """Fire the same prompt at all N models in parallel with streaming, display side-by-side cards."""
    kwargs.setdefault("max_tokens", 1024)
    kwargs.setdefault("extra_body", {"chat_template_kwargs": {"enable_thinking": False}})

    # Use MODELS order as-is (already sorted by discover_models or user selection)
    models_order = list(MODELS)

    # Shared state for each model
    state = {}
    for m in MODELS:
        state[m["label"]] = {
            "text": "", "tokens": 0, "elapsed": 0, "tps": 0,
            "ttft": None, "done": False, "color": m["color"],
            "token_times": [],  # List of elapsed_seconds per token for tok/s plotting
            "error": None,      # Capture error message if any
        }

    handle = display(HTML(_render_cards(state, models_order)), display_id=True)

    def stream_model(m):
        label = m["label"]
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        try:
            response = clients[label].chat.completions.create(
                model=m["model"], messages=messages, stream=True, **kwargs
            )
            token_count = 0
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if state[label]["ttft"] is None:
                        state[label]["ttft"] = time.time() - t0
                    state[label]["text"] += content
                    token_count += 1
                    elapsed = time.time() - t0
                    state[label]["tokens"] = token_count
                    state[label]["elapsed"] = elapsed
                    state[label]["tps"] = token_count / elapsed if elapsed > 0 else 0
                    state[label]["token_times"].append(elapsed)
        except Exception as e:
            state[label]["error"] = str(e)
            if not state[label]["text"]:
                state[label]["text"] = f"Error: {e}"
        finally:
            elapsed = time.time() - t0
            state[label]["elapsed"] = elapsed
            if state[label]["tokens"] > 0:
                state[label]["tps"] = state[label]["tokens"] / elapsed
            state[label]["done"] = True

    # Launch all streaming threads
    threads = []
    for m in MODELS:
        t = threading.Thread(target=stream_model, args=(m,))
        t.start()
        threads.append(t)

    # Refresh display every 300ms until all done
    while not all(state[m["label"]]["done"] for m in MODELS):
        time.sleep(0.3)
        handle.update(HTML(_render_cards(state, models_order)))

    # Final render
    handle.update(HTML(_render_cards(state, models_order)))

    # Return results
    results = []
    for m in models_order:
        s = state[m["label"]]
        results.append({
            "label": m["label"], "text": strip_think(s["text"]),
            "raw_text": s["text"],
            "tokens": s["tokens"], "elapsed": s["elapsed"],
            "tps": s["tps"], "ttft": s["ttft"], "color": s["color"],
            "token_times": s.get("token_times", []),
            "error": s.get("error"),
        })
    return results


def show_metrics_table(results):
    """Render a comparison metrics table from compare_models results."""
    rows = ""
    for r in results:
        ttft_ms = int(r["ttft"] * 1000) if r.get("ttft") is not None else None
        ttft_str = f"{ttft_ms} ms" if ttft_ms is not None else "—"
        rows += (
            f"<tr>"
            f"<td style='padding:6px 12px; color:{r['color']}; font-weight:bold; border-bottom:1px solid #e5e7eb;'>{r['label']}</td>"
            f"<td style='padding:6px 12px; color:#111827; border-bottom:1px solid #e5e7eb;'>{ttft_str}</td>"
            f"<td style='padding:6px 12px; color:#111827; border-bottom:1px solid #e5e7eb;'>{r['tokens']}</td>"
            f"<td style='padding:6px 12px; color:#111827; border-bottom:1px solid #e5e7eb;'>{r['elapsed']:.1f}s</td>"
            f"<td style='padding:6px 12px; color:#16a34a; font-weight:bold; border-bottom:1px solid #e5e7eb;'>{r['tps']:.1f}</td>"
            f"</tr>"
        )
    display(HTML(f"""
    <table style="background:#f9fafb; border:1px solid #d1d5db; border-collapse:collapse; border-radius:8px; overflow:hidden; margin:10px 0; font-family:monospace; min-width:400px;">
      <thead><tr style="background:#1e3a5f;">
        <th style="padding:8px 12px; color:white; text-align:left;">Model</th>
        <th style="padding:8px 12px; color:white; text-align:left;">TTFT</th>
        <th style="padding:8px 12px; color:white; text-align:left;">Tokens</th>
        <th style="padding:8px 12px; color:white; text-align:left;">Time</th>
        <th style="padding:8px 12px; color:white; text-align:left;">tok/s</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """))
    # Show errors/warnings as footnotes
    footnotes = []
    for r in results:
        if r.get("error"):
            footnotes.append(f"<div style='color:#dc2626; font-size:0.8em;'>⚠️ {r['label']}: {r['error']}</div>")
        if r.get("raw_text") and not r.get("text"):
            footnotes.append(f"<div style='color:#f59e0b; font-size:0.8em;'>⚠️ {r['label']}: Response stripped by think-tag filter (raw: {len(r['raw_text'])} chars)</div>")
    if footnotes:
        display(HTML("".join(footnotes)))


def show_tps_chart(results):
    """Render an inline SVG chart of tok/s over time from compare_models results."""
    width, height = 600, 150
    pad_left, pad_right, pad_top, pad_bottom = 45, 15, 25, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Compute rolling tok/s for each model
    series = []
    max_time = 0
    max_tps = 0
    window = 10  # rolling window size

    for r in results:
        times = r.get("token_times", [])
        if len(times) < 2:
            series.append({"label": r["label"], "color": r["color"], "points": []})
            continue
        points = []
        for i in range(window, len(times)):
            dt = times[i] - times[i - window]
            if dt > 0:
                tps = window / dt
                points.append((times[i], tps))
                max_tps = max(max_tps, tps)
        max_time = max(max_time, times[-1]) if times else max_time
        series.append({"label": r["label"], "color": r["color"], "points": points})

    if max_time == 0 or max_tps == 0:
        return  # Nothing to plot

    # Round up max_tps for nice axis
    max_tps = max_tps * 1.1

    def scale_x(t):
        return pad_left + (t / max_time) * plot_w

    def scale_y(v):
        return pad_top + plot_h - (v / max_tps) * plot_h

    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#fafafa; border:1px solid #e5e7eb; border-radius:6px; font-family:monospace;">'
    ]

    # Title
    svg_parts.append(
        f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="11" fill="#374151" font-weight="bold">tok/s over time</text>'
    )

    # Y-axis labels
    for frac in [0, 0.5, 1.0]:
        val = max_tps * frac
        y = scale_y(val)
        svg_parts.append(f'<text x="{pad_left - 5}" y="{y + 4}" text-anchor="end" font-size="9" fill="#9ca3af">{val:.0f}</text>')
        svg_parts.append(f'<line x1="{pad_left}" y1="{y}" x2="{width - pad_right}" y2="{y}" stroke="#e5e7eb" stroke-width="0.5"/>')

    # X-axis labels
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        val = max_time * frac
        x = scale_x(val)
        svg_parts.append(f'<text x="{x}" y="{height - 8}" text-anchor="middle" font-size="9" fill="#9ca3af">{val:.0f}s</text>')

    # Plot lines
    for s in series:
        if len(s["points"]) < 2:
            continue
        path_d = "M " + " L ".join(f"{scale_x(t):.1f},{scale_y(v):.1f}" for t, v in s["points"])
        svg_parts.append(f'<path d="{path_d}" fill="none" stroke="{s["color"]}" stroke-width="1.5" opacity="0.8"/>')

    # Legend (bottom-left, inline)
    legend_x = pad_left + 5
    for i, s in enumerate(series):
        lx = legend_x + i * 80
        ly = height - 6
        svg_parts.append(f'<line x1="{lx}" y1="{ly - 3}" x2="{lx + 15}" y2="{ly - 3}" stroke="{s["color"]}" stroke-width="2"/>')
        svg_parts.append(f'<text x="{lx + 18}" y="{ly}" font-size="9" fill="#6b7280">{s["label"]}</text>')

    svg_parts.append('</svg>')
    display(HTML('\n'.join(svg_parts)))
