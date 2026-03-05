# LLM Lab Videos — Local AI on Apple Silicon

> YouTube companion notebooks for hands-on LLM engineering

A notebook series accompanying the [LLM Lab](https://github.com/shanemmattner/llm-lab) project. Each episode maps to a YouTube video and provides runnable code for local AI development on Apple Silicon.

---

## About This Series

This stuff is dense — there's a lot to learn, and making the series is how I'm learning it. I retain things best by teaching, so building this out is as much for me as it is for anyone watching. I just picked up a Mac Studio M4 Max and wanted to actually push it rather than just browse the web on a very expensive machine. My background is embedded systems and firmware, so the analogies I reach for tend to come from that world — unified memory maps more naturally to me as an SoC architecture than a discrete GPU setup, quantization makes sense framed against ADC resolution, and the prefill/decode split feels a lot like flash reads versus bit-banging a peripheral. If you come from a hardware or firmware background, those comparisons will probably land; if not, feel free to skip past them.

Who is this for? Anyone who wants to learn local LLM stuff — and realistically a Mac is required, not just recommended. The MLX stack only runs on Apple Silicon. You can follow along on Linux/Windows with Ollama but the scripts won't be identical. 32GB+ RAM recommended to run models at a useful size.

Episode 01 runs three models simultaneously in a side-by-side arena comparison — a 122B, a 35B, and a 0.8B — served concurrently via MLX so you can watch them respond to the same prompt at the same time.

---

## Quick Start

```bash
git clone https://github.com/shanemmattner/llm-lab-videos.git
cd llm-lab-videos
pip install -r requirements.txt
python setup_check.py
# Launch 3 MLX servers (requires 128GB RAM)
bash start_servers.sh
jupyter notebook
# When done:
bash stop_servers.sh
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

| RAM | Notes |
|-----|-------|
| 16 GB | Works — limited to small/quantized models |
| 32 GB | Usable for most episodes |
| 64 GB+ | Comfortable — covers the majority of what this series runs |
| 128 GB | Headroom for full-scale model experimentation — Required for Episode 01's 3-model arena (122B + 35B + 0.8B simultaneously) |

**Apple Silicon Mac is required.** The MLX stack does not run on Linux or Windows. Linux/Windows users can substitute Ollama for the inference pieces, but the notebooks are written for MLX and won't run as-is.

---

## Related

- **LLM Lab** (infrastructure & tooling): [github.com/shanemmattner/llm-lab](https://github.com/shanemmattner/llm-lab)

---

## License

[MIT](LICENSE)
