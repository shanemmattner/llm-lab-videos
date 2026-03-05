# SCRIPT-01.md — MLX Inference (Video 01)
## Talking Points for Presenter | Target: 15–20 min

> **Format key:**
> - **SAY** — things to say out loud (talk naturally, not verbatim)
> - **SHOW** — what to have visible / click through on screen
> - **KEY NUMBERS** — anchor facts worth hitting exactly

---

## [0:00–1:00] Welcome & Series Intro

**SAY**
- Welcome — this is a hands-on engineering series, not a prompt-engineering course. This is for people who want to know what's *inside* the box.
- Who it's for: engineers comfortable with Python, virtual environments, and floating-point formats. You don't need HuggingFace or Apple ML experience — that's what we're building here.
- By the end of this episode you'll have a large language model running locally on your Mac, and you'll understand *why* it works at the hardware level — not just that it works.
- Quick teaser of the series arc: inference → tokenization → fine-tuning → agents. Episode 1 is the foundation everything else builds on.

**SHOW**
- Notebook open to the top markdown cell — series title, the quote *"The best model is the one you actually control."*
- Quickly scroll to show the notebook structure (sections visible) so viewers know what's coming.

**KEY NUMBERS**
- None needed here — save the numbers for later.

---

## [1:00–2:00] Hardware Check (run setup_check.py)

**SAY**
- Before anything else, let's make sure the environment is actually ready. Nothing worse than 20 minutes of theory and then a broken install.
- Point out the three things `setup_check.py` verifies: Apple Silicon chip detected, Python environment correct, MLX installed and importable.
- If anything's red, check the repo README — it covers the common install gotchas.
- Also note: the MLX server needs to be running in a terminal *before* executing the inference cells. Show the one-liner command.

**SHOW**
- Run `!python setup_check.py` in the notebook — let output appear live.
- Briefly show the terminal with `mlx_lm.server --model mlx-community/Qwen3-32B-4bit --port 8800` running (or ready to run).

**KEY NUMBERS**
- Server port: **8800**
- Install footprint: just three packages — `openai psutil mlx`

---

## [2:00–5:00] The Big Picture (unified memory, MLX, why local)

**SAY**
- Set the scene: imagine you've spent years on firmware, you care about every byte, cycle counts matter to you. Someone hands you a model that generates language and says "just call the API." That's not enough. We're going to open the box.
- The good news: a transformer is not magic. It's matrix multiplies and additions, repeated dozens of times. "122 billion parameters" means 122 billion numbers sitting in weight matrices. Inference reads those weights and multiplies them with your input. That's it.
- The reason you can do this locally comes down to one architectural decision Apple made: **unified memory**. On a discrete GPU setup, there's CPU RAM on one side, GPU VRAM on the other, connected by a slow PCIe bus. Models have to fit in VRAM — typically 24GB on a high-end consumer card. On Apple Silicon, the CPU, GPU, and Neural Engine all share the same physical memory pool. Your "VRAM" *is* your system RAM.
- The tradeoff is memory bandwidth. The M4 Max is ~1/6th the bandwidth of a server H100. But you're running privately, locally, at zero cost per token. That's the deal.
- MLX is Apple's framework built *specifically* for this topology. It's not PyTorch ported to Metal — it was designed from scratch with unified memory and lazy evaluation in mind. Brief explanation of lazy evaluation: you describe the computation, MLX builds the graph, nothing actually runs until you need the output. Like a DMA scatter-gather list — you describe the transfers, then kick them all off at once.

**SHOW**
- ASCII diagram from the notebook: Traditional GPU Setup vs Apple Silicon side-by-side. Pause here — this is the key visual.
- Bandwidth comparison table: M4 Max 546 GB/s vs H100 SXM5 3.35 TB/s.
- The lazy evaluation code cell — run it live: `x * 2`, `y + 1`, then `print(z)` triggers the compute.

**KEY NUMBERS**
- M4 Max memory bandwidth: **546 GB/s**
- H100 SXM5: **3.35 TB/s** (~6× faster)
- Discrete GPU VRAM ceiling: typically **24 GB**
- M4 Max unified memory (high config): **128 GB**

---

## [5:00–10:00] Core Concepts

### Quantization

**SAY**
- Every model weight is a number. Full precision (fp32) = 4 bytes each. At 4-bit quantization = 0.5 bytes. That's an 8× shrink in model size. For a 122B parameter model: fp32 would be 488 GB — impossible locally. 4-bit brings it to ~61 GB — fits with room to spare.
- The analogy: think of it like ADC resolution. 16-bit audio vs 8-bit vs 4-bit — you lose dynamic range and introduce quantization noise, but for most content 4-bit is surprisingly good. Neural network weights are robust to precision reduction because the model learned an approximation, and that approximation survives reduced precision.
- MLX uses **group quantization** — instead of one scale factor for the entire weight matrix, it quantizes in groups of ~64 weights independently. Like calibrating an ADC per channel instead of sharing one calibration across all channels. Dramatically reduces systematic error.
- The format we'll encounter most: **NVFP4** — a floating-point 4-bit format (E2M1 weights, E4M3 scales, group size 16). It places more representable values near zero, where neural network weights cluster. Better quality than integer INT4 at the same bit width.

**SHOW**
- Quantization table from the notebook: precision / bytes-per-param / 122B model size / quality column. Read across a couple rows.

**KEY NUMBERS**
- fp32: **4 bytes/param**, 122B model = **488 GB**
- 4-bit: **0.5 bytes/param**, 122B model = **61 GB**
- Quality at 4-bit: **~95–97%** of fp32
- Quality cliff below **3-bit**: noticeable degradation

---

### Transformers: Layers, Heads, Forward Pass

**SAY**
- A transformer is a stack of blocks called layers. Data flows in at the bottom, through each layer in sequence, output at the top. Each layer has two parts: an attention mechanism and a feed-forward network (FFN).
- Heads are inside the attention mechanism — instead of one big attention computation, the model splits into parallel "heads," each learning to focus on different patterns. One head might track subject-verb agreement, another tracks positional proximity, another tracks sentiment. Outputs get concatenated back together.
- The FFN is just two matrix multiplies with a nonlinearity — but modern models use **SwiGLU**, a gated variant that needs three weight matrices per layer instead of two. Empirically better quality at the same parameter count.
- Walk through the forward pass diagram briefly: token IDs → embedding lookup → for each layer: layer norm, Q/K/V projections, attention scores, FFN, residual → final projection → softmax → sample next token. Every weight matrix in every layer gets read from RAM for every single token generated.

**SHOW**
- Forward pass ASCII diagram from the notebook. Trace through it step by step.

**KEY NUMBERS**
- Attention heads in a large model: e.g., **64 query heads**, often **8 KV heads** (GQA)
- Three FFN weight matrices per layer with SwiGLU: **W_gate, W_up, W_down**

---

### KV Cache

**SAY**
- Every token the model generates, it needs to "remember" all previous tokens. It does this with the KV cache — storing the Key and Value tensors for every layer, every step. Without it, generating token N would require recomputing attention over all N-1 previous tokens from scratch.
- The formula: `2 × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_element`. The dangerous variable is **seq_len** — it grows with every token. Model weights are static, but the KV cache keeps growing through a conversation.
- Always check `num_key_value_heads` in the model's `config.json` — not `num_attention_heads`. Modern models use GQA (Grouped Query Attention) where multiple query heads share a single KV head pair. Fewer KV heads = much smaller cache.
- Key insight for mental model: same weights + same input = identical KV cache state, byte for byte, every time. The model is a pure mathematical function. You can write deterministic regression tests against it. At temperature=0, output is fully reproducible.

**SHOW**
- KV cache formula from the notebook. Work through the Qwen3.5-35B worked example numbers.

**KEY NUMBERS**
- "2" in the formula = Keys AND Values
- KV cache at 32K context (simplified 40-layer example): **~2.68 GB**
- Budget rule: `model_memory + kv_cache_memory < total_RAM`
- Safe cap flag: `--max-kv-size 4096`

---

### Prefill vs Decode

**SAY**
- Inference has two distinct phases that behave completely differently. **Prefill**: process the entire input prompt in parallel — GPU-bound, fast. **Decode**: generate one token at a time — sequential, memory-bandwidth-bound, slower.
- The analogy: prefill is like flashing firmware — you write a whole block at once in burst mode. Decode is like bit-banging a GPIO — one bit at a time, waiting on every cycle.
- This is why a model can feel "fast to start" — prefill finishes quickly — but then tokens stream at a steady, slower cadence. The pause-then-stream pattern you see in every chat UI is prefill vs decode in action.
- TTFT (Time To First Token) measures prefill. TPS (Tokens Per Second) measures decode. Don't conflate them — they respond to different optimizations.

**SHOW**
- Prefill/Decode side-by-side table from the notebook.

**KEY NUMBERS**
- Practical decode rate on M4 Max 128GB with Qwen3-32B-4bit: **~40–50 tok/s**
- Theoretical bandwidth ceiling for 122B MoE at NVFP4: **~110 tok/s**

---

### Mixture of Experts

**SAY**
- MoE models have a clever trick: instead of one big FFN per layer, they have N small "expert" FFNs. A router (a small learned linear layer) picks the top-K experts for each token, computes only those, and combines their outputs. All other experts sit idle for that token.
- The result: a 122B parameter model where only ~10B parameters are *active* per token. All 122B must fit in RAM — but inference speed is determined by the active 10B. You get big-model quality at smaller-model inference speed.
- The notation: "Qwen3.5-122B-A10B" — 122B total, 10B **A**ctive.
- Known failure mode worth mentioning: MoE can underperform dense models on deep multi-step reasoning (complex math proofs, long chains). The routing can fragment reasoning across expert boundaries in ways a dense model's continuous computation path doesn't.

**SHOW**
- MoE router diagram from the notebook — dense FFN vs experts with router side by side.

**KEY NUMBERS**
- Qwen3.5-122B-A10B: **122B total**, **10B active** per token
- Size at NVFP4: **~65 GB** — fits in 128 GB with headroom
- Bandwidth math: 546 GB/s ÷ ~5 GB/token active weights ≈ **~110 tok/s ceiling**

---

### Model Evaluation: Choosing What to Download

**SAY**
- There are thousands of models on HuggingFace. Here's the practical workflow: (1) Find candidates in the `mlx-community` org — they maintain pre-quantized MLX versions. (2) RAM fit check — apply the 60% rule: keep weights under ~77 GB on a 128 GB machine, leave headroom for KV cache. (3) Check Open LLM Leaderboard for standardized benchmarks — look across the mix (ARC, MMLU, GSM8K, IFEval), not just the headline number. Some models are benchmark-trained to look good without generalizing. (4) Check r/LocalLLaMA for real user reports — catches things benchmarks miss. (5) LMArena ELO — blind human preference comparisons. The gap between benchmark rank and arena ELO is a signal in itself.
- Don't download a model just because it's new. Have a reason.

**SHOW**
- The worked comparison table: Qwen3.5-122B-A10B vs DenseX-70B across RAM, MMLU, IFEval, community feedback, arena ELO.

**KEY NUMBERS**
- 60% rule: keep weights under **~77 GB** on 128 GB machine
- `mlx-community` org on HuggingFace = go-to source for MLX-native quantized models

---

## [10:00–15:00] Hands-On

### First Inference

**SAY**
- Enough theory — let's run it. Two lines of setup: confirm MLX sees the GPU, then make the first API call. The server is already running, so this is just an HTTP call to localhost.
- `mlx_lm.server` makes your local model a drop-in replacement for the OpenAI API. Any tool that speaks the OpenAI protocol — LangChain, LlamaIndex, Cursor — can point to `localhost:8800` with zero code changes. That's intentional and powerful.
- Watch the tok/s number in the output — that's your baseline. We'll improve on it in later modules.

**SHOW**
- Run the MLX version check cell — `mx.default_device()` output confirms GPU.
- Run the first inference cell live — let the response stream or print. Point out the elapsed time and tok/s calculation in the output.

**KEY NUMBERS**
- Baseline decode rate to call out: whatever shows up — typical is **35–55 tok/s** for 32B 4-bit
- `base_url="http://localhost:8800/v1"` — the drop-in API endpoint

---

### Benchmark

**SAY**
- MLX uses lazy evaluation and JIT-style compilation. The very first request includes connection setup and graph compilation overhead — that will make your timing look worse than reality. Always do a warmup pass before benchmarking. This is the most common mistake.
- Run a short warmup (5 tokens, say "hi"), then time the real request. The tok/s number you get from the warmup run is garbage — ignore it.
- Watch for the numbers: on a 128 GB M4 Max with Qwen3-32B-4bit, expect the post-warmup rate to be noticeably faster than the cold first call.

**SHOW**
- Run the benchmark cell — warmup fires first, then the timed run. Point at the printed `X.X tok/s` in output.

**KEY NUMBERS**
- Common pitfall to name explicitly: **"always warm up before timing"**
- Expected range post-warmup: **35–60 tok/s** depending on system load

---

### Sampling Parameters

**SAY**
- Temperature is the most important parameter you'll tune. Temperature=0 is greedy decoding — always picks the highest-probability token, fully deterministic, good for code and structured output. Temperature=0.7 is the typical chat default — balanced creativity. Temperature=1.2+ gets chaotic — can produce surprising outputs, but above ~1.3 on large MoE models tends to degrade toward gibberish.
- Top-p (nucleus sampling): only sample from the smallest set of tokens whose cumulative probability reaches p. At top_p=0.9, you throw away the long tail of unlikely tokens. Less nonsense without hurting creativity much.
- Run the same prompt at three temperatures — watch how different the outputs are. This is the clearest way to build intuition for what temperature actually does.

**SHOW**
- Run the sampling params cell live — three configs (temp 0, 0.7, 1.2) on the robot story prompt. Read one or two of the outputs aloud and contrast them.

**KEY NUMBERS**
- temp=**0.0**: deterministic / greedy
- temp=**0.7**: standard chat default
- temp above **~1.3**: quality degrades on large MoE models

---

### Memory Monitoring

**SAY**
- Running out of memory on Apple Silicon is sneaky. macOS will start swapping to NVMe before it actually crashes — you'll see tok/s drop to 1–3 tok/s as the system starts paging. That's your warning sign.
- The threshold: when active Metal memory exceeds ~90% of physical RAM (on 128 GB, that's ~115 GB), you are in swap territory. Check this periodically during long sessions.
- Four remediation options in order of least disruptive: (1) cap the KV cache with `--max-kv-size 4096`, (2) shorten your prompt or summarize history, (3) drop to a lower quantization, (4) load a smaller model.
- Run the memory monitoring cell — this is something worth bookmarking, you'll use it in every episode.

**SHOW**
- Run the memory monitoring cell — show `mx.metal.get_active_memory()`, peak memory, and `psutil` RAM stats all printing together.

**KEY NUMBERS**
- Swap warning threshold: **~90% of physical RAM** (e.g., ~115 GB on 128 GB machine)
- KV cache cap flag: `--max-kv-size 4096`
- Swap symptom: tok/s drops to **1–3 tok/s**

---

## [15:00–17:00] Takeaways & What's Next

**SAY**
- Eight things to walk away with — hit the highlights, not all eight verbatim:
  1. Unified memory is the reason this works at all. No VRAM bottleneck. CPU, GPU, and Neural Engine share one address space.
  2. Inference is memory-bandwidth-bound, not compute-bound. Throughput ceiling = memory bandwidth ÷ active weights per token. Quantization directly improves throughput by shrinking model size.
  3. Transformer = layers of attention + FFN stacked deep. Every weight matrix in every layer gets read from RAM for every token. That's why bandwidth is the constraint.
  4. KV cache memory grows with context length. Model weights are static — budget separately for weights and KV cache. Use `--max-kv-size` to cap the latter.
  5. Prefill and decode are fundamentally different operations. TTFT and TPS measure different things. Don't conflate them.
  6. `mlx_lm.server` makes your local model OpenAI-compatible. Zero code changes to plug into any existing tooling.
  7. The model is a pure function — same input, same output, same KV cache state every time. Debuggable and testable like any other system.
- What's next: Module 1B is the Qwen3.5 model datasheet — the specific architecture including the hybrid DeltaNet/GQA layers and the corrected KV cache formula for hybrid models. Think of it as the datasheet review before writing firmware against a chip. Module 1C covers inference optimization — speculative decoding, prompt caching, NVFP4/MXFP4 deep dive. Module 2 is tokenization and embeddings — how the model actually sees text.
- Subscribe if you want the notification when the next one drops. All code is in the repo.

**SHOW**
- Notebook takeaways cell — scroll through the 8 bullet points while talking.
- Briefly show the "What's Next" cell — module roadmap.

**KEY NUMBERS**
- The one-liner to commit to memory: `mlx_lm.server --model mlx-community/Qwen3-32B-4bit --port 8800`
- The bandwidth formula to repeat: **546 GB/s ÷ active_weights_per_token = throughput ceiling**

---

*End of talking points. Total estimated runtime: 16–18 min at a comfortable pace with live cell execution.*
