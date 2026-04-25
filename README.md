# DistributedInference

My goal from this project is to be able to do Prefill on a single GPU and pass it for Decode on another GPU -> this is to understand the memory heavy and compute heavy stages during inference.
There are libraries already available that does this - Im trying to do this project to be able to clearly understand the process and try to make it efficient for the Decode phase and then take latency into account.


Plan that walks from a single attention head all the way to disaggregated serving and Dynamo. 

## Setup assumptions
Aat minimum: 1 GPU for Phase 0–3, 2 GPUs (single node) for Phase 4–5, and 2 nodes with NCCL or RDMA between them for Phase 6+. 

Tools: `torch`, `torch.distributed`, `transformers`, `vllm`, `nvtx` / `torch.profiler`, `nsight systems` for traces.

---

## Phase 0 — Build the forward pass 
Goal: own every tensor that flows through one decoder layer. 

Exercises:
1. Implement scaled dot-product attention in raw PyTorch (no `nn.MultiheadAttention`). Inputs Q, K, V of shape `[B, H, S, D]`, output `[B, H, S, D]`.
2. Wrap it in a single transformer block (RMSNorm → attention → residual → SwiGLU MLP → residual). Match the Llama architecture so later phases plug in cleanly.
3. Load Llama-3.2-1B weights into hand-rolled block and verify outputs match HuggingFace within `1e-3` for one prompt. This will catch every subtle bug for free.

What to internalize: the shapes. Write `[B, H, S, D]` on a sticky note.

---

## Phase 1 — Attention variants
Goal: feel why MQA/GQA exist by *measuring* KV cache size.

Concept: in MHA have `H` query heads and `H` KV heads. In MQA have `H` Q heads and `1` KV head. In GQA have `H` Q heads and `G` KV heads where `1 < G < H` (Llama 3 uses 32 Q / 8 KV).

Exercises:
1. Take Phase 0 attention. Add a flag `kv_heads` and reshape so MHA, GQA, MQA all flow through one code path (broadcast K/V across grouped Q heads).
2. Compute KV cache memory analytically: `2 * num_layers * seq_len * kv_heads * head_dim * dtype_bytes`. Plot it for MHA vs GQA vs MQA at `seq_len = 8k, 32k, 128k`. This is the punchline: long context is unaffordable without GQA.
3. Replace attention with `flash_attn_func` from the `flash-attn` package. Profile both with `torch.profiler` at `seq_len = 4096`. Look at the trace: vanilla attention launches a kernel that allocates `[B, H, S, S]`; FlashAttention does not. That `S²` allocation is the entire reason FA exists.

What to internalize: FlashAttention isn't a different math — same softmax(QK^T)V — it's a memory-access reordering (tiling + online softmax) that keeps the `S×S` matrix in SRAM. MQA/GQA reduce *KV cache*, not compute. These solve different problems.

---

## Phase 2 — Prefill vs decode
Goal: see the compute-bound / memory-bound split.

Concept: prefill processes the full prompt in one forward pass — that's a `[B, S_prompt, D]` tensor through every matmul, lots of arithmetic per byte loaded → **compute-bound**. Decode generates one token at a time — `[B, 1, D]` through the same weights, so load gigabytes of weights to do one token's worth of math → **memory-bandwidth-bound**.

Exercises:
1. Implement greedy generation with an explicit KV cache: `past_k`, `past_v` lists per layer, append on each step.
2. Measure separately:
   - prefill throughput: `tokens/sec` for prompt of length 512, batch 1.
   - decode throughput: `tokens/sec` for generating 256 tokens after that prompt.
   see prefill is maybe 10–50× faster per token. That gap is the entire motivation for disaggregation.
3. Use `nvidia-smi dmon` or `nsys` during decode. Watch SM utilization — it'll be low. Watch HBM bandwidth — it'll be near peak. That's a memory-bound workload's fingerprint.
4. Now try batch size 32 in decode. Throughput per token goes way up because amortize the weight-loading across more tokens. This is *why* batching matters for decode but barely matters for prefill.

What to internalize: prefill wants big tensor cores and low latency. Decode wants high HBM bandwidth and big batches. They benefit from *different* hardware and *different* parallelism. That's the whole reason to disaggregate.

---

## Phase 3 — Batching strategies
Goal: understand why static batching is bad and what continuous batching actually does.

Exercises:
1. Static batching: take 8 requests of varying length, pad to longest, run together. Measure GPU idle time during decode of the short ones.
2. Continuous (inflight) batching: maintain a "running" batch, evict finished sequences, admit new ones every step. Implement a toy version— write maybe 200 lines and finally understand vLLM's scheduler.
3. Read the PagedAttention paper and the `vllm` `block_manager.py`. Run vLLM on the same workload and compare throughput to toy version.

What to internalize: KV cache fragmentation is the bottleneck of naive batching. Paging fixes that, exactly the way virtual memory fixed external fragmentation in OSes.

---

## Phase 4 — Parallelism on a single node (multi-GPU)
Goal: actually move tensors between GPUs and see communication cost.

Concepts:
- **TP (tensor parallel)**: split weight matrices across GPUs. Column-parallel for the first matmul of a block, row-parallel for the second, all-reduce in between. For attention split *heads* across GPUs.
- **DP (data parallel)**: full model replica per GPU, different batch shards. Cheap communication, but each GPU needs the full model. For inference, DP usually means independent replicas behind a load balancer.
- **PP (pipeline parallel)**: split *layers* across GPUs. Microbatches flow through. Bubbles hurt latency-bound decode.
- **EP (expert parallel)**: only relevant for MoE — experts live on different GPUs, all-to-all dispatches tokens to their chosen experts.

Exercises:
1. Two GPUs, one node. Use `torch.distributed` with NCCL. Implement 2-way TP for a single attention layer: split QKV projection column-wise, split output projection row-wise, all-reduce. Verify numerical equivalence to non-TP.
2. Extend to a full Llama block, then full model. Run prefill at TP=1 and TP=2 — measure latency and per-GPU memory. Then do the same for decode. see TP helps prefill latency and lets fit bigger models, but for small-batch decode the all-reduce eats wins.
3. (Optional, MoE) Spin up Mixtral or DeepSeek-MoE in vLLM with `--tensor-parallel-size 2 --enable-expert-parallel` and look at the all-to-all in `nsys`. See why EP is communication-heavy.

What to internalize: TP communication happens *every layer*. It only works when GPUs are connected by NVLink. Across nodes (slow Ethernet/IB), TP collapses — that's why cross-node parallelism is usually PP or DP or expert-parallel, not TP.

---

## Phase 5 — Serving stack: frontend and backend
Goal: separate the "engine" from the "server" in head.

Concepts:
- Backend / engine: the thing that runs forward passes — vLLM, TRT-LLM, SGLang, raw PyTorch. Owns the model, the KV cache, the scheduler.
- Frontend: HTTP/gRPC server, tokenization, request validation, streaming, OpenAI-compatible API. Owns nothing about the model.

Exercises:
1. Write a tiny FastAPI server that wraps `transformers.generate` and exposes `/v1/chat/completions`. Stream tokens via SSE. This is the "PyTorch backend" baseline — slow but control everything.
2. Replace the backend with `vllm.LLMEngine` (the offline class) called from FastAPI server. Same API surface, ~10× throughput. Notice what changed: continuous batching, paged KV, CUDA graphs.
3. Now run vLLM's built-in OpenAI server (`python -m vllm.entrypoints.openai.api_server`) and benchmark with `vllm bench serve`. Compare to custom frontend → vLLM engine setup. Identical numbers. The point: frontend is interchangeable; engine is the substance.

What to internalize: Dynamo, Triton, and vLLM's server are all *frontends + routers* — they orchestrate engines. Engines like vLLM-core or TRT-LLM are the actual GPU workhorses.

---

## Phase 6 — Disaggregated prefill/decode across two nodes
This is the centerpiece. Goal: prefill on node A, ship the KV cache to node B, decode on B.

Exercises:
1. **KV transfer primitive.** On 2 nodes with NCCL set up, write a script: rank 0 generates a tensor shaped like a real KV cache (`[layers, 2, S, kv_heads, head_dim]`), rank 1 receives it via `dist.send` / `dist.recv`. Measure transfer time vs tensor size. Compare TCP vs IB. This number — KV transfer ms — is the central number in disaggregated serving.
2. **Toy disaggregation.** On node A, run prefill only: tokenize prompt, forward pass, capture KV cache, send to node B. On node B, receive KV, run decode loop with that KV as `past_key_values`, stream tokens back to A, A returns to client. Now reimplementing Dynamo's core flow at toy scale.
3. **Where it breaks.** Now try with a real model (say Llama-3-8B). The KV for one 8k-token prompt is hundreds of MB. Transfer time becomes meaningful. Try sending layer-by-layer as prefill computes them (overlap transfer with compute) — this is exactly what NIXL / LMCache do. Measure the savings.
4. **Routing.** Add a simple load balancer in front: 2 prefill workers, 4 decode workers. Round-robin first. Then try "send to the decode worker that already has cached the prompt prefix" — that's KV-aware routing, the Dynamo smart-router idea, in 50 lines.

What to internalize: disaggregation buys the right to scale prefill and decode *independently* (different counts, different parallelism configs, different GPU SKUs even). The cost is KV transfer. Dynamo exists to make that cost manageable.

---

## Phase 7 — Now read Dynamo
With Phases 0–6 done, Dynamo's docs and code stop being mysterious. Recognize every piece.

Exercises:
1. Read the Dynamo architecture doc and map every component to something built: smart router → Phase 6 step 4; KV manager → Phase 6 step 1; engine adapter → Phase 5 step 2.
2. Run Dynamo's reference example with vLLM as the backend. Trace one request end-to-end with logging. Identify: where does prefill happen? When does KV transfer? Where does the decode worker get scheduled?
3. Swap the backend to TRT-LLM. Notice that the *frontend and routing* don't change — that's the abstraction win.
4. Try a failure: kill a decode worker mid-stream. Watch what the router does. This is where production-readiness lives.

