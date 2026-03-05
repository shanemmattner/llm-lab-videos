# LLM Lab Videos — Local AI on Apple Silicon

> YouTube companion notebooks for hands-on LLM engineering

A notebook series accompanying the [LLM Lab](https://github.com/shanemmattner/llm-lab) project. Each episode maps to a YouTube video and provides runnable code for local AI development on Apple Silicon.

---

## Quick Start

```bash
git clone https://github.com/shanemmattner/llm-lab-videos.git
cd llm-lab-videos
pip install -r requirements.txt
python setup_check.py
jupyter notebook
```

---

## Episodes

| Episode | Title | Notebook |
|---------|-------|----------|
| 01 | MLX Inference | [01-mlx-inference.ipynb](01-mlx-inference.ipynb) |
| 02 | Coming soon | — |
| 03 | Coming soon | — |
| 04 | Coming soon | — |

---

## Hardware Requirements

| RAM | Status |
|-----|--------|
| 16 GB | Minimum — small models only |
| 64 GB+ | Recommended — comfortable for most episodes |
| 128 GB | Ideal — full-scale model experimentation |

All notebooks target Apple Silicon (M-series). `mlx-lm` is Apple Silicon only and will be skipped on Linux/Windows.

---

## Related

- **LLM Lab** (infrastructure & tooling): [github.com/shanemmattner/llm-lab](https://github.com/shanemmattner/llm-lab)

---

## License

[MIT](LICENSE)
