# Notebook Conventions

Quick reference for authors of notebooks in `sections/`.

---

## 1. Cell ID Naming

Every code cell **must** have an ID. Format: `cell-{descriptive-slug}`

Examples: `cell-cover`, `cell-setup`, `cell-helpers`, `cell-stress-test`

**Set in JupyterLab:** Property Inspector (right sidebar) → Advanced Tools → Cell ID

IDs are how the test harness identifies cells in `NOTEBOOK_CONFIGS`. A cell without a configured ID defaults to `{"type": "static"}`.

---

## 2. Standard Cell Order

| Position | Cell ID | Purpose |
|---|---|---|
| 1 | `cell-cover` | Hero card HTML — title, subtitle, gradient background |
| 2 | `cell-setup-check` | Port scan + server discovery via `discover_servers()` |
| 3 | `cell-setup` | Import libs, call `notebook_helpers.init(models, clients)` |
| 4 | `cell-helpers` | Define any notebook-local helper functions |
| 5+ | `cell-{topic}` | Content cells — demos, experiments, visualisations |

`cell-cover` and `cell-helpers` are almost always `"type": "static"` — they produce HTML output but don't need a live server.

---

## 3. Creating a New Notebook

```bash
python3 scripts/create_notebook.py NN "Title" "Subtitle"
```

This scaffolds `sections/NN-slug/NN-slug.ipynb` with the standard cell structure.

After scaffolding:
1. Add your content cells with meaningful IDs.
2. Register the notebook in `NOTEBOOK_CONFIGS` (see §4).

---

## 4. Test Configuration

Add an entry to `NOTEBOOK_CONFIGS` in `scripts/test_notebooks.py`:

```python
NOTEBOOK_CONFIGS: dict[str, dict[str, dict]] = {
    "02-my-topic.ipynb": {
        "cell-cover":        {"type": "static"},
        "cell-setup-check":  {"type": "live", "needs_ports": "any"},
        "cell-setup":        {"type": "live", "needs_ports": "any"},
        "cell-helpers":      {"type": "static"},
        "cell-my-demo":      {"type": "live", "needs_ports": "any"},
        "cell-122b-only":    {"type": "live", "needs_ports": [8800]},
        "cell-slow-bench":   {"type": "live", "needs_ports": "any", "timeout": 300},
    },
}
```

### `"type"`
- `"static"` — cell runs without any MLX server (HTML rendering, pure Python, HuggingFace fetch). Always executed.
- `"live"` — cell requires at least one server. Skipped when `needs_ports` conditions are not met.

### `"needs_ports"`
- Omitted — cell always runs regardless of server state (use with `"type": "static"`).
- `"any"` — skip if **no** MLX ports (8800, 8801, 8802) are up.
- `[8800]` — skip unless **all listed ports** are up. Use when a cell targets a specific model size.

### `"timeout"`
Default execution timeout is 120 s. Override per-cell for slow operations:
```python
"cell-stress-test": {"type": "live", "needs_ports": "any", "timeout": 300},
```

### `"expect"` *(planned)*
Validate cell output after execution:
```python
"cell-my-demo": {
    "type": "live",
    "needs_ports": "any",
    "expect": {
        "output_contains": "tok/s",   # plain text in stdout
        "html_contains":   "<table",  # string in HTML output
    },
},
```

---

## 5. How Server Discovery Works

`discover_servers()` in `notebook_helpers.py` does **not** call `/v1/models`.

**Why:** `/v1/models` lists every model in the HuggingFace cache — not just the one currently loaded. Passing the wrong model ID to the OpenAI client causes the server to silently swap models mid-session.

**What it does instead:**
1. Checks each port (8800–8802) with a TCP connect.
2. For live ports, runs `lsof` to get the PID, then `ps` to read the process command line.
3. Parses the `--model <id>` argument directly from the command line.

This guarantees the model ID matches exactly what the server loaded.

```python
models, clients = discover_servers()   # returns only what's actually running
notebook_helpers.init(models, clients)
```

---

## 6. Visual Patterns

### Hero / cover card
Gradient background (`#1e3a5f → #2d6a9f`), white text, subtitle in lighter weight. Used in `cell-cover`.

### Section dividers
```html
<!-- Gradient rule -->
<hr style="border:none; height:2px;
     background:linear-gradient(90deg,#1e3a5f,#2d6a9f,transparent); margin:32px 0;">

<!-- Coloured left-border callout -->
<div style="border-left:4px solid #2563eb; padding:8px 16px; background:#eff6ff;">
  Section header text
</div>
```

### HTML data tables
```python
# Header row background
"background:#1e3a5f"   # dark navy
# Cell text color
"color:#111827"
# Table font
"font-family:monospace"
# Row separator
"border-bottom:1px solid #e5e7eb"
```

See `notebook_helpers.show_metrics_table()` for a reusable implementation.

---

## Running Tests

```bash
# All notebooks
uv run scripts/test_notebooks.py

# Single notebook
uv run scripts/test_notebooks.py sections/01-mlx-inference/01-mlx-inference.ipynb

# Validate config only (no kernel)
uv run scripts/test_notebooks.py --dry-run
```
