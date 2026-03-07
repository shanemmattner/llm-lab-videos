# Local LLM Lab — Apple Silicon

Hands-on notebooks for running LLMs locally on Apple Silicon. Written from an embedded systems and electronics perspective — unified memory as SoC architecture, quantization as ADC resolution, prefill/decode as flash reads vs bit-banging a peripheral.

Designed to be used with [Claude Code](https://claude.ai/claude-code) if you need to modify or extend anything.

## Requirements

- Apple Silicon Mac (MLX only runs on Apple Silicon)
- Python 3.12+ (Homebrew Python recommended)
- RAM depends on which models you run — 8 GB is enough for small models, 128 GB for the full 122B

## Quick Start

```bash
git clone https://github.com/shanemmattner/llm-lab-videos.git
cd llm-lab-videos

# Install the Jupyter kernel (Homebrew Python with required packages)
pip3 install openai psutil markdown mlx-lm ipykernel
python3 -m ipykernel install --user --name homebrew-py3 --display-name "Python 3 (Homebrew)"

# Run the hardware check
python3 scripts/setup_check.py

# Start MLX servers (edit start_servers.sh to choose models/ports)
bash scripts/start_servers.sh

# Launch Jupyter and select "Python 3 (Homebrew)" kernel
jupyter notebook sections/01-mlx-inference/01-mlx-inference.ipynb

# When done:
bash scripts/stop_servers.sh
```

Notebooks auto-detect all running MLX servers on ports 8800-8809. Run as many or as few models as your RAM allows — no configuration changes needed.

## Sections

| # | Topic | Notebook | Description |
|---|-------|----------|-------------|
| 01 | [MLX Inference](sections/01-mlx-inference/) | `01-mlx-inference.ipynb` | Run and compare models side-by-side — streaming, personas, tok/s charts |
| 01b | [Model Datasheet](sections/01-mlx-inference/01b-model-datasheet.ipynb) | `01b-model-datasheet.ipynb` | Architecture deep dive — DeltaNet/GQA hybrid, MoE routing, KV cache math |
| 01c | [Inference Optimization](sections/01-mlx-inference/01c-inference-optimization.ipynb) | `01c-inference-optimization.ipynb` | Memory bandwidth analysis, quantization formats, batch efficiency |
| 02 | [Tokenization & Embeddings](sections/02-tokenization-and-embeddings/) | `02-tokenization-and-embeddings.ipynb` | BPE tokenization, token-level analysis, embedding spaces |
| 03 | [Prompting Techniques](sections/03-prompting-techniques/) | `03-prompting-techniques.ipynb` | System prompts, few-shot, chain-of-thought, structured output |

## Architecture

```
llm-lab-videos/
├── sections/
│   ├── 01-mlx-inference/
│   │   ├── 01-mlx-inference.ipynb
│   │   ├── 01b-model-datasheet.ipynb
│   │   ├── 01c-inference-optimization.ipynb
│   │   └── SCRIPT-01.md
│   ├── 02-tokenization-and-embeddings/
│   │   └── 02-tokenization-and-embeddings.ipynb
│   └── 03-prompting-techniques/
│       └── 03-prompting-techniques.ipynb
├── scripts/
│   ├── notebook_helpers.py      # Shared module: discover_servers(), warmup_models(), compare_models()
│   ├── setup_check.py           # Hardware/MLX detection (imports from notebook_helpers)
│   ├── test_notebooks.py        # Test harness: cell-by-cell execution + validation
│   ├── create_notebook.py       # Scaffold new notebooks from template
│   ├── notebook_template.ipynb  # Template with standard cells
│   ├── install_kernel.sh        # Install homebrew-py3 kernel spec
│   ├── kernel-spec/             # Canonical kernel.json (committed)
│   ├── start_servers.sh         # Launch MLX servers
│   ├── stop_servers.sh          # Stop MLX servers (scans 8800-8809)
│   └── NOTEBOOK_CONVENTIONS.md  # Authoring guide for contributors
└── requirements.txt
```

### Server Discovery

`notebook_helpers.discover_servers()` is the single source of truth for finding running MLX servers. It scans ports 8800-8809, uses `lsof` + `ps` to read the model ID from the process command line (not `/v1/models`, which lists cached models). All notebooks call this — no hardcoded port lists or model names anywhere.

### Notebook Conventions

Every notebook follows a standard cell structure:

| Cell ID | Purpose |
|---------|---------|
| `cell-cover` | Hero card with title and gradient background |
| `cell-setup-check` | Hardware scan via `setup_check.py` |
| `cell-setup` | Import guard + `discover_servers()` + `warmup_models()` + `init()` |
| `cell-helpers` | Notebook-local helper functions |
| `cell-*` | Content cells |

The setup cell includes an import guard that catches wrong-kernel errors immediately. All notebooks use the `homebrew-py3` kernel.

See [scripts/NOTEBOOK_CONVENTIONS.md](scripts/NOTEBOOK_CONVENTIONS.md) for the full authoring guide.

## Testing

```bash
# Smoke test — import/kernel validation only (~15s, no servers needed)
uv run scripts/test_notebooks.py --smoke

# Config validation (no kernel needed)
uv run scripts/test_notebooks.py --dry-run --strict --validate-ids

# Full test (requires MLX servers running)
uv run scripts/test_notebooks.py

# Single notebook
uv run scripts/test_notebooks.py sections/01-mlx-inference/01-mlx-inference.ipynb
```

## License

MIT
