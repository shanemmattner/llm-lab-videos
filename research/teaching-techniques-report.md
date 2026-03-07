# LLM Concepts: Teaching Techniques & Metaphors from the Web

Research compiled 2026-03-07. Covers how different authors teach core LLM concepts using analogies, metaphors, and novel pedagogical approaches.

---

## 1. Tokenization / BPE (Byte Pair Encoding)

### Step-by-step merge walkthrough (Langformers Blog)
The most effective teaching approach found: walk through BPE training on a tiny corpus of just 4 words (`eat`, `eats`, `eating`, `eaten`). Each iteration shows:
1. Split words into individual characters
2. Count all adjacent symbol pairs and their frequencies
3. Merge the most frequent pair
4. Update the vocabulary and merge list
5. Repeat until vocab size reached

The power of this approach is showing that tokenization is **iterative compression** — the algorithm discovers that `e-a-t` always appears together, so it merges them into a single token `eat`, then builds `eats`, `eating`, `eaten` on top. The merge list IS the tokenizer — it's the lookup table used at inference time.

**Key insight taught**: Unknown words aren't catastrophic. If the tokenizer encounters `eated` (not in training), it applies merge rules as far as they go (`eate` + `d`) and falls back to character-level splitting. This graceful degradation is rarely explained elsewhere.

> Source: https://blog.langformers.com/bpe-tokenizer-explained/

### Vocabulary size as a design tradeoff
The Langformers post frames vocabulary size as a practical engineering decision: GPT-2 has ~50k tokens, LLaMA 4 has 202,048, DeepSeek has 129,280. Larger vocab = fewer tokens per sentence (faster inference) but larger embedding table (more memory). This tradeoff framing resonates with hardware engineers.

> Source: https://blog.langformers.com/bpe-tokenizer-explained/

---

## 2. Embeddings

### Points in space (Saad Ahmed, Medium)
Embeddings explained as points in a high-dimensional space where **similar words cluster together**. Cat and dog are nearby; horse and building are far apart. Each dimension represents some latent feature (the author illustrates with a table showing made-up dimensions like "is_animal", "is_living", etc., even though real dimensions aren't interpretable).

The key teaching move: show that the raw embedding captures word meaning but **not context**. The word "only" means different things in "Only Saad wants coffee" vs "Saad only wants coffee." That's why attention is needed — to create **contextualized** embeddings.

> Source: https://medium.com/@saad.ahmed1926q/kv-cache-explained-intuitively-2b425a36dfc7

---

## 3. Attention Mechanism

### People at a party (Saad Ahmed, Medium)
The most creative attention analogy found. Imagine tokens as people at a gathering:

- **Query** = "Who am I looking for?" (what this token wants to know)
- **Key** = "Here's my name tag" (what this token advertises about itself)
- **Value** = "Here's the actual information I carry"

The attention score between two tokens is how strongly one person's query matches another's key. If token "Saad" has high attention to token "coffee," it means the model considers their relationship important for prediction.

**Causal masking** taught as: "You can only look at people who arrived before you at the party. No peeking at future arrivals."

> Source: https://medium.com/@saad.ahmed1926q/kv-cache-explained-intuitively-2b425a36dfc7

---

## 4. KV Cache

### The redundant recomputation problem (Saad Ahmed, Medium)
Taught by first showing the problem WITHOUT caching: at each generation step, the model recomputes Q, K, V for ALL previous tokens, even though their K and V vectors never change (only the new token's Q matters). The author walks through generating "The cat sat on the mat" token by token, showing how computation grows quadratically.

**The fix**: Cache all previous K and V vectors. At each new step, only compute Q, K, V for the new token, then concatenate its K and V with the cache. The attention computation still touches all positions, but the expensive projection (embedding × weight matrix) only runs once per token.

**Visual approach**: Tables showing exactly which computations happen at each step, with cached values highlighted. No abstract math — just "here's what gets computed, here's what gets reused."

> Source: https://medium.com/@saad.ahmed1926q/kv-cache-explained-intuitively-2b425a36dfc7

---

## 5. Chain-of-Thought (CoT) Prompting

### "Showing your working in an exam" (Andrea, Medium)
CoT framed as the model "showing its work" — not just giving the answer but the logical steps to get there. This analogy appears across multiple sources and seems to be the dominant teaching metaphor.

> Source: https://medium.com/@andreaerisme/a-deep-dive-into-llm-prompting-techniques-744d93decebb

### Three flavors of CoT (Codecademy)
Clean taxonomy that makes CoT digestible:

| Type | How it works | When to use |
|------|-------------|-------------|
| **Zero-shot CoT** | Just add "Let's think step by step" | Quick, no examples needed |
| **Few-shot CoT** | Provide 2-3 solved examples with reasoning | More reliable, but manual effort |
| **Auto-CoT** | Cluster similar problems, generate examples automatically via zero-shot, then use as few-shot | Scales to many problem types |

The Auto-CoT explanation is particularly well done: cluster questions by similarity (using sentence transformers), pick one representative from each cluster, generate its reasoning chain via zero-shot, then bundle those as few-shot examples for new questions.

> Source: https://www.codecademy.com/article/chain-of-thought-cot-prompting

### CoT as emergent ability (Andrea, Medium)
Important nuance: CoT doesn't help small models. It's an "emergent ability of model scale" — only large models benefit. On the GSM8K math dataset, CoT **doubled** performance for large models but did nothing for small ones. This frames CoT not as a universal technique but as one that unlocks latent capability in sufficiently large models.

> Source: https://medium.com/@andreaerisme/a-deep-dive-into-llm-prompting-techniques-744d93decebb

---

## 6. Self-Consistency (CoT-SC)

### Panel of expert consultants (DEV Community)
"Imagine hiring a panel of expert consultants. Each one analyzes the problem independently and presents their conclusion. You then go with the answer that most experts agree on."

### 10 explorers in a maze (DEV Community)
"Send 10 explorers into a maze, each taking a different path. If 7 of them find the same exit, you can be pretty confident that's the real way out."

Both metaphors emphasize: **diverse reasoning paths, same conclusion = high confidence**. The mechanism is majority voting, not averaging.

### "Asking your kakis (friends)" (Andrea, Medium)
Singaporean cultural framing: instead of asking one friend, ask several. Each gives their own reasoning. If most agree, you trust the answer. Reinforces that the technique is about **sampling multiple independent reasoning chains** and taking the consensus.

**Key properties taught**:
- It's a self-ensemble (no external models needed)
- Works with any CoT prompt (zero-shot or few-shot)
- The diverse paths are generated by sampling from the model's decoder (temperature > 0)
- Correct reasoning paths tend to converge; incorrect paths scatter

> Sources:
> - https://dev.to/oliver_bennet_e005ad93e7/chain-of-thought-cot-prompting-the-definitive-guide-2025-3m5o
> - https://medium.com/@andreaerisme/a-deep-dive-into-llm-prompting-techniques-744d93decebb

---

## 7. Temperature

### Softmax scaling factor (Kelsey Wang, Medium)
The clearest mathematical explanation found. Temperature divides the logits before softmax:

```
softmax(logits / T)
```

- **T < 1**: Amplifies differences between logits → distribution becomes peaky → model picks the highest-probability token almost every time → deterministic, repetitive
- **T = 1**: Default behavior, unmodified distribution
- **T > 1**: Compresses differences → distribution flattens → model considers lower-probability tokens → more creative, more random

**The "dial" metaphor**: Temperature as a dial between "explorative" (high T) and "conservative" (low T). Not the most original metaphor, but the mathematical grounding (showing actual softmax calculations with different T values) makes it stick.

**Practical guidance taught**: T=0.2 for factual Q&A, T=0.7-0.9 for creative writing, T=1.2+ for brainstorming. This concrete guidance is more useful than abstract explanations.

> Source: https://medium.com/@kelseywang19/understanding-llm-temperature-a-deep-dive-into-how-it-shapes-ai-output-3d9ca8f3a3a

---

## 8. Speculative Decoding

### Branch prediction in CPUs (Google Research — original authors)
The original speculative decoding paper's authors explicitly draw the analogy to **speculative execution** in CPU pipelines:

- A small "draft" model generates several candidate tokens quickly (like a branch predictor guessing which branch to take)
- The large "target" model verifies them all in parallel (like the CPU pipeline executing speculatively)
- If the draft was right, you keep the work and saved time
- If the draft was wrong, you discard and revert (like a pipeline flush)

**Key insight**: The bottleneck for LLM inference is **memory bandwidth, not compute**. Modern hardware can do hundreds of FLOPS per byte read. So the draft model's speculative tokens can be verified "for free" — the large model was going to read its weights anyway, and it has spare compute cycles to verify multiple tokens at once.

**Two observations that motivate the technique**:
1. Some tokens are easy (copying "7" after "square root of 7 is"), some are hard (computing "2.646"). Small models handle easy tokens fine.
2. Hardware has spare compute during inference (memory-bound, not compute-bound). Verifying multiple tokens costs almost nothing extra.

**Speculative sampling** (the stochastic generalization): Unlike deterministic speculative execution, LLMs output probability distributions. The acceptance criterion is probabilistic — if the draft model's distribution matches the target's, accept with high probability. This guarantees **identical output distribution** to running the target model alone.

> Source: https://research.google/blog/looking-back-at-speculative-decoding/

---

## 9. Mixture of Experts (MoE)

### Hospital with specialized departments (Gregory Zem, Medium)
"Imagine a hospital where every patient doesn't see every doctor. Instead, a triage nurse (the router/gating network) directs each patient to the right specialist. The hospital has all the departments (total parameters), but each patient only visits 2-3 (active parameters)."

This maps cleanly:
- **Experts** = specialized departments (cardiology, neurology, etc.)
- **Router/Gate** = triage nurse deciding which experts handle each token
- **Sparse activation** = each patient only sees relevant specialists
- **Total vs active parameters** = hospital has 100 doctors but each patient sees 3

**Why MoE matters**: A 1.8T parameter model (like DeepSeek-V3) only activates ~37B per token. You get the knowledge of a massive model at the inference cost of a much smaller one.

**Top-k routing explained**: The router produces a probability distribution over all experts. Only the top-k (usually 2) experts are activated. Their outputs are weighted by the router's probability scores and combined.

> Source: https://medium.com/@gregory.zem/understanding-mixture-of-experts-moe-in-large-language-models-b0a1bab9e40d

---

## 10. Tree of Thoughts (ToT)

### Maze exploration with backtracking (Andrea, Medium)
ToT framed as exploring a maze where you can try multiple paths at each junction, evaluate which ones look promising, and backtrack from dead ends. Unlike CoT (single linear chain) or CoT-SC (multiple independent chains), ToT allows the model to **evaluate intermediate states** and prune bad branches early.

The key difference from CoT-SC: self-consistency runs multiple chains to completion and votes on final answers. ToT evaluates at each step and can abandon unpromising paths mid-way. This is more like tree search (BFS/DFS) than parallel independent chains.

> Source: https://medium.com/@andreaerisme/a-deep-dive-into-llm-prompting-techniques-744d93decebb

---

## 11. ReAct (Reasoning + Acting)

### Detective investigating a case (Andrea, Medium)
The model alternates between:
- **Thought**: "I need to find out X" (reasoning about what to do)
- **Action**: Actually doing it (calling a tool, searching, computing)
- **Observation**: Processing the result

Framed as a detective who doesn't just think about the crime — they think, then go collect evidence, then update their theory based on what they find. The interleaving of reasoning and action is what distinguishes ReAct from pure CoT.

> Source: https://medium.com/@andreaerisme/a-deep-dive-into-llm-prompting-techniques-744d93decebb

---

## Summary: Most Effective Teaching Patterns

| Pattern | Used For | Why It Works |
|---------|----------|--------------|
| **Tiny worked example** | BPE, KV cache | Walk through every step on a minimal case — no abstraction, just mechanics |
| **"What if we DON'T do this?"** | KV cache, speculative decoding | Show the inefficiency first, then the optimization feels obvious |
| **Concrete numbers** | Temperature, MoE | "T=0.2 for facts, T=0.9 for creativity" beats abstract explanation |
| **Role-based analogy** | MoE (hospital), self-consistency (panel of experts), attention (party guests) | Assign human roles to components |
| **Hardware analogy** | Speculative decoding (branch prediction), KV cache (memory reuse) | Maps directly to existing engineering intuition |
| **Taxonomy table** | CoT variants (zero/few/auto) | Side-by-side comparison clarifies when to use what |
| **Emergent ability framing** | CoT | "This only works on large models" prevents misapplication |

---

## Sources

1. **BPE Tokenizer Training & Tokenization** — Langformers Blog (Rabindra Lamsal)
   https://blog.langformers.com/bpe-tokenizer-explained/

2. **KV Cache, Attention, Embeddings** — Saad Ahmed Siddiqui, Medium (Jul 2025)
   https://medium.com/@saad.ahmed1926q/kv-cache-explained-intuitively-2b425a36dfc7

3. **Speculative Decoding** — Google Research (Yaniv Leviathan, Matan Kalman, Yossi Matias, Dec 2024)
   https://research.google/blog/looking-back-at-speculative-decoding/

4. **CoT Prompting Explained** — Codecademy Team
   https://www.codecademy.com/article/chain-of-thought-cot-prompting

5. **LLM Prompting Techniques Deep Dive** — Andrea, Medium (Apr 2025)
   https://medium.com/@andreaerisme/a-deep-dive-into-llm-prompting-techniques-744d93decebb

6. **CoT Prompting Definitive Guide** — DEV Community (Oliver Bennet, 2025)
   https://dev.to/oliver_bennet_e005ad93e7/chain-of-thought-cot-prompting-the-definitive-guide-2025-3m5o

7. **Understanding MoE in LLMs** — Gregory Zem, Medium
   https://medium.com/@gregory.zem/understanding-mixture-of-experts-moe-in-large-language-models-b0a1bab9e40d

8. **Understanding LLM Temperature** — Kelsey Wang, Medium
   https://medium.com/@kelseywang19/understanding-llm-temperature-a-deep-dive-into-how-it-shapes-ai-output-3d9ca8f3a3a
