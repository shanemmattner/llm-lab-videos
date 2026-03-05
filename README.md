# Local LLM Lab — Apple Silicon

Hands-on notebooks for running LLMs locally on Apple Silicon. Written from an embedded systems and electronics perspective — unified memory as SoC architecture, quantization as ADC resolution, prefill/decode as flash reads vs bit-banging a peripheral.

Designed to be used with [Claude Code](https://claude.ai/claude-code) if you need to modify or extend anything.

## Requirements

- Apple Silicon Mac (MLX only runs on Apple Silicon)
- 128 GB RAM for Section 01 (runs 3 models simultaneously)
- Python 3.12+

## Quick Start

```bash
git clone https://github.com/shanemmattner/llm-lab-videos.git
cd llm-lab-videos
pip install -r requirements.txt
python scripts/setup_check.py
bash scripts/start_servers.sh
jupyter notebook sections/01-model-arena/01-mlx-inference.ipynb
# when done:
bash scripts/stop_servers.sh
```

## Sections

| # | Topic | Notebook |
|---|-------|----------|
| 01 | [3-Model Arena Comparison](sections/01-model-arena/) | Run 122B, 35B, and 2B side-by-side |

## Structure

```
sections/01-model-arena/   Notebook + presenter notes
scripts/                   Server management, hardware check
requirements.txt           Python dependencies
```

## License

MIT
