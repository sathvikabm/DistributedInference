# DistributedInference

My goal from this project is to be able to do Prefill on a single GPU and pass it for Decode on another GPU -> this is to understand the memory heavy and compute heavy stages during inference.
There are libraries already available that does this - Im trying to do this project to be able to clearly understand the process and try to make it efficient for the Decode phase and then take latency into account.
 


1) Qwen2.5-0.5B -> 2 H100 NVL 
-> Ran time check for different prompt length(prefill) and ctx_len(decode) on the CUDA:0 


Findings 

a) When GQA was used num_KV_head = 2 -> the num_attention_head did not show up in the shape
K shape: [1, 2, 5, 64] = [batch=1, num_kv_heads=2, seq_len=5, head_dim=64]

b) Prefill time vs Decode time 
warmup time excluded 
Prefill is compute-bound. 128 tokens took 11.8 ms, 2048 tokens took 12.9 ms. 16x'd the work and the time barely moved. 

```   prompt_len   prefill_ms   tokens/sec
         128        11.82        10833
         512        12.13        42221
        1024        12.28        83388
        2048        12.90       158760
        4096        24.56       166806
```

        
Decode is memory-bound. 10.6 ms per token, flat across all context lengths. Doesn't matter if  context is 128 or 4096 — same speed. Why? Because the bottleneck isn't math, it's reading ~1 GB of weights from HBM every single step. The GPU's compute units are mostly waiting around. ~94 tokens/sec.
``` ctx_len   decode_ms/tok   tokens/sec
         128           10.57           95
         512           10.70           93
        1024           10.69           94
        2048           10.58           94
        4096           10.58           94
```

c) Prefill on CUDA:0, ship KV cache to CUDA:1, Decode on CUDA:1 matched Single GPU-output

The DynamicCache object isn't directly .to()-able, so we move the underlying tensors and rebuild it on the other side.
.to() requires both tensors to live in one process's address space (one Python process owning both GPUs). Doesn't scale.


move_cache_to_device
```
from transformers import DynamicCache

def move_cache_to_device(cache, device):
    """Move a DynamicCache from one GPU to another by moving each layer's K and V tensors."""
    new_cache = DynamicCache()
    for i in range(len(cache)):
        k, v = get_kv(cache, i)
        k_new = k.to(device, non_blocking=True)
        v_new = v.to(device, non_blocking=True)
        new_cache.update(k_new, v_new, i)
    torch.cuda.synchronize()
    return new_cache
```


check for NVLink
```
import time

def bench_transfer(size_mb):
    n_floats = (size_mb * 1024 * 1024) // 2  # fp16 = 2 bytes
    x = torch.randn(n_floats, dtype=torch.float16, device="cuda:0")
    
    # Warmup
    for _ in range(3):
        y = x.to("cuda:1")
    torch.cuda.synchronize()
    
    # Time it
    n_runs = 20
    start = time.perf_counter()
    for _ in range(n_runs):
        y = x.to("cuda:1", non_blocking=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs
    
    bytes_moved = n_floats * 2
    bandwidth_gbs = bytes_moved / elapsed / 1e9
    print(f"  {size_mb:>5} MB: {elapsed*1000:>7.3f} ms  -> {bandwidth_gbs:>6.1f} GB/s")
    return bandwidth_gbs

print("Transfer bandwidth GPU 0 -> GPU 1:")
for sz in [1, 10, 100, 500, 1000]:
    bench_transfer(sz)
```

Output
```
Transfer bandwidth GPU 0 -> GPU 1:
      1 MB:   0.020 ms  ->   52.3 GB/s
     10 MB:   0.055 ms  ->  189.2 GB/s
    100 MB:   0.411 ms  ->  255.4 GB/s
    500 MB:   1.985 ms  ->  264.2 GB/s
   1000 MB:   3.954 ms  ->  265.2 GB/s
```

    
d) NCCL with two process
```torchrun --nproc_per_node=2 --master_port=29500 disagg_nccl.py```

Real systems have two processes (or two servers), each owning one GPU, talking via NCCL over NVLink. Same wire, different API.
# Send each layer's K and V
```
send
#PREFILL SERVER
for i in range(len(kv)):
   dist.send(kv.layers[i].keys.contiguous(), dst=1)
   dist.send(kv.layers[i].values.contiguous(), dst=1)
#Send first token
dist.send(first_token, dst=1)

receive
#DECODE SERVER
meta = torch.zeros(5, dtype=torch.long, device="cuda:1")
dist.recv(meta, src=0)
n_layers, B, H, S, D = meta.tolist()

kv = DynamicCache()
for i in range(n_layers):
   k = torch.empty((B, H, S, D), dtype=torch.float16, device="cuda:1")
   v = torch.empty((B, H, S, D), dtype=torch.float16, device="cuda:1")
   dist.recv(k, src=0)
   dist.recv(v, src=0)
   kv.update(k, v, i)
# Receive first token
next_token = torch.zeros((1, 1), dtype=torch.long, device="cuda:1")
dist.recv(next_token, src=0)

```

Rank 0 sent in 8.79 ms
Rank 1 received in 4.88 ms
For a 5-token prompt, KV cache is ~60 KB total

60 KB at NVLink speed should take microseconds, not milliseconds. So why 8 ms? 
Two reasons:
First-time NCCL warmup. First send/recv after init_process_group does buffer allocation, kernel JIT, connection setup. Subsequent transfers will be ~100× faster. We'll see that in pipelining.
24 layers × 2 tensors = 48 separate dist.send calls. Each has launch overhead. Real systems batch this — concatenate all K's into one tensor, send once. Optimization for later. 

disagg_nccl.py
```
import os
import time
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=f"cuda:{rank}"
    )
    model.eval()
    
    if rank == 0:
        # PREFILL SERVER
        inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        kv = out.past_key_values
        first_token = out.logits[:, -1:].argmax(dim=-1)
        
        # Send shape metadata first (one tensor, fixed size)
        k0 = kv.layers[0].keys
        meta = torch.tensor(
            [len(kv), k0.shape[0], k0.shape[1], k0.shape[2], k0.shape[3]],
            dtype=torch.long, device="cuda:0",
        )
        dist.send(meta, dst=1)
        
        # Send each layer's K and V
        t0 = time.perf_counter()
        for i in range(len(kv)):
            dist.send(kv.layers[i].keys.contiguous(), dst=1)
            dist.send(kv.layers[i].values.contiguous(), dst=1)
        torch.cuda.synchronize()
        print(f"[rank 0] sent KV cache in {(time.perf_counter()-t0)*1000:.2f} ms")
        
        # Send first token
        dist.send(first_token, dst=1)
    
    else:
        # DECODE SERVER
        meta = torch.zeros(5, dtype=torch.long, device="cuda:1")
        dist.recv(meta, src=0)
        n_layers, B, H, S, D = meta.tolist()
        print(f"[rank 1] receiving cache: layers={n_layers}, shape=[{B},{H},{S},{D}]")
        
        # Receive into a fresh DynamicCache
        kv = DynamicCache()
        t0 = time.perf_counter()
        for i in range(n_layers):
            k = torch.empty((B, H, S, D), dtype=torch.float16, device="cuda:1")
            v = torch.empty((B, H, S, D), dtype=torch.float16, device="cuda:1")
            dist.recv(k, src=0)
            dist.recv(v, src=0)
            kv.update(k, v, i)
        torch.cuda.synchronize()
        print(f"[rank 1] received in {(time.perf_counter()-t0)*1000:.2f} ms")
        
        # Receive first token
        next_token = torch.zeros((1, 1), dtype=torch.long, device="cuda:1")
        dist.recv(next_token, src=0)
        
        # Decode loop
        generated = [next_token]
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS - 1):
                out = model(next_token, past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                next_token = out.logits[:, -1:].argmax(dim=-1)
                generated.append(next_token)
        
        full = torch.cat(generated, dim=1)
        print(f"[rank 1] generated: {tokenizer.decode(full[0])!r}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

e) Pipeline - Plan: throw N requests at the system. Rank 0 prefills request i+1 while rank 1 decodes request i. Compare throughput vs single-GPU sequential.
disagg_pipeline.py
```
import os, time, torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B"
N_REQUESTS = 20
PROMPT_LEN = 512
DECODE_LEN = 64

def send_cache(kv, dst):
    k0 = kv.layers[0].keys
    meta = torch.tensor(
        [len(kv), k0.shape[0], k0.shape[1], k0.shape[2], k0.shape[3]],
        dtype=torch.long, device=k0.device,
    )
    dist.send(meta, dst=dst)
    for i in range(len(kv)):
        dist.send(kv.layers[i].keys.contiguous(), dst=dst)
        dist.send(kv.layers[i].values.contiguous(), dst=dst)

def recv_cache(src, device):
    meta = torch.zeros(5, dtype=torch.long, device=device)
    dist.recv(meta, src=src)
    n_layers, B, H, S, D = meta.tolist()
    kv = DynamicCache()
    for i in range(n_layers):
        k = torch.empty((B, H, S, D), dtype=torch.float16, device=device)
        v = torch.empty((B, H, S, D), dtype=torch.float16, device=device)
        dist.recv(k, src=src)
        dist.recv(v, src=src)
        kv.update(k, v, i)
    return kv

def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=f"cuda:{rank}"
    )
    model.eval()
    
    # Warmup NCCL with a small send/recv
    if rank == 0:
        warmup = torch.ones(1, device="cuda:0")
        dist.send(warmup, dst=1)
    else:
        warmup = torch.zeros(1, device="cuda:1")
        dist.recv(warmup, src=0)
    torch.cuda.synchronize()
    
    if rank == 0:
        # === DISAGGREGATED RUN ===
        prompts = [torch.randint(0, 1000, (1, PROMPT_LEN), device="cuda:0") 
                   for _ in range(N_REQUESTS)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for ids in prompts:
            with torch.no_grad():
                out = model(ids, use_cache=True)
            send_cache(out.past_key_values, dst=1)
            first = out.logits[:, -1:].argmax(dim=-1)
            dist.send(first, dst=1)
        
        torch.cuda.synchronize()
        t_disagg = time.perf_counter() - t0
        print(f"[rank 0] disaggregated: {N_REQUESTS} reqs in {t_disagg:.2f}s "
              f"= {N_REQUESTS*DECODE_LEN/t_disagg:.0f} tok/s")
        
        # === SINGLE-GPU BASELINE (rank 0 does everything) ===
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for ids in prompts:
            with torch.no_grad():
                out = model(ids, use_cache=True)
                kv = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)
                for _ in range(DECODE_LEN - 1):
                    out = model(next_tok, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_single = time.perf_counter() - t0
        print(f"[rank 0] single-GPU:    {N_REQUESTS} reqs in {t_single:.2f}s "
              f"= {N_REQUESTS*DECODE_LEN/t_single:.0f} tok/s")
        
        print(f"\nSpeedup (disagg / single): {t_single/t_disagg:.2f}x")
    
    else:
        # Rank 1: receive and decode in a loop
        for _ in range(N_REQUESTS):
            kv = recv_cache(src=0, device="cuda:1")
            next_tok = torch.zeros((1, 1), dtype=torch.long, device="cuda:1")
            dist.recv(next_tok, src=0)
            with torch.no_grad():
                for _ in range(DECODE_LEN - 1):
                    out = model(next_tok, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```
Output
[rank 0] disaggregated: 20 reqs in 13.76s = 93 tok/s
[rank 0] single-GPU:    20 reqs in 13.28s = 96 tok/s
Speedup (disagg / single): 0.97x
```

Compare:

Single-GPU baseline: prefill+decode on one H100, sequential requests
Disaggregated pipelined: prefill on rank 0, decode on rank 1, overlapped

case1:

This version runs requests sequentially through the pipeline — meaning rank 0 sends a request, then waits before sending the next one (because dist.send is synchronous). It's not yet truly pipelined.

disaggregation without overlap is just adding network overhead. The whole point of disagg is to do prefill and decode in parallel on different GPUs.

```
GPU 0: [prefill][send KV]------------idle for 670 ms----------[next req]
GPU 1: --------idle--------[recv KV][decode 64 tokens]---idle---
```

case2:
While rank 1 decodes request K, rank 0 should be prefilling request K+1. Both GPUs busy at the same time. That requires non-blocking sends (isend) so rank 0 doesn't wait for rank 1 to receive before starting the next prefill.

non-blocking and overlapping prefill-i+1 with decode-i
disagg_pipelined_nonblocking.py

```
import os, time, torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B"
N_REQUESTS = 20
PROMPT_LEN = 512
DECODE_LEN = 64

def isend_cache(kv, dst):
    """Non-blocking send. Returns list of work handles."""
    k0 = kv.layers[0].keys
    meta = torch.tensor(
        [len(kv), k0.shape[0], k0.shape[1], k0.shape[2], k0.shape[3]],
        dtype=torch.long, device=k0.device,
    )
    works = [dist.isend(meta, dst=dst)]
    # Keep references to tensors so they don't get freed before send completes
    tensors = [meta]
    for i in range(len(kv)):
        k = kv.layers[i].keys.contiguous()
        v = kv.layers[i].values.contiguous()
        tensors.append(k); tensors.append(v)
        works.append(dist.isend(k, dst=dst))
        works.append(dist.isend(v, dst=dst))
    return works, tensors

def recv_cache(src, device):
    meta = torch.zeros(5, dtype=torch.long, device=device)
    dist.recv(meta, src=src)
    n_layers, B, H, S, D = meta.tolist()
    kv = DynamicCache()
    for i in range(n_layers):
        k = torch.empty((B, H, S, D), dtype=torch.float16, device=device)
        v = torch.empty((B, H, S, D), dtype=torch.float16, device=device)
        dist.recv(k, src=src)
        dist.recv(v, src=src)
        kv.update(k, v, i)
    return kv

def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map=f"cuda:{rank}"
    )
    model.eval()
    
    # Warmup NCCL
    if rank == 0:
        dist.send(torch.ones(1, device="cuda:0"), dst=1)
    else:
        dist.recv(torch.zeros(1, device="cuda:1"), src=0)
    torch.cuda.synchronize()
    
    if rank == 0:
        prompts = [torch.randint(0, 1000, (1, PROMPT_LEN), device="cuda:0") 
                   for _ in range(N_REQUESTS)]
        
        # === PIPELINED DISAGGREGATED ===
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        pending = []  # keep tensors alive until their send completes
        for ids in prompts:
            with torch.no_grad():
                out = model(ids, use_cache=True)
            works, tensors = isend_cache(out.past_key_values, dst=1)
            first = out.logits[:, -1:].argmax(dim=-1).contiguous()
            works.append(dist.isend(first, dst=1))
            tensors.append(first)
            pending.append((works, tensors))
            
            # Reap completed sends so memory can be reused
            pending = [(w, t) for (w, t) in pending if not all(x.is_completed() for x in w)]
        
        # Wait for all remaining sends
        for works, _ in pending:
            for w in works:
                w.wait()
        
        # Wait for rank 1 to finish all decodes
        done = torch.zeros(1, device="cuda:0")
        dist.recv(done, src=1)
        
        torch.cuda.synchronize()
        t_disagg = time.perf_counter() - t0
        print(f"[rank 0] PIPELINED disagg: {N_REQUESTS} reqs in {t_disagg:.2f}s "
              f"= {N_REQUESTS*DECODE_LEN/t_disagg:.0f} tok/s")
        
        # === SINGLE-GPU BASELINE ===
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for ids in prompts:
            with torch.no_grad():
                out = model(ids, use_cache=True)
                kv = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)
                for _ in range(DECODE_LEN - 1):
                    out = model(next_tok, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_single = time.perf_counter() - t0
        print(f"[rank 0] single-GPU:       {N_REQUESTS} reqs in {t_single:.2f}s "
              f"= {N_REQUESTS*DECODE_LEN/t_single:.0f} tok/s")
        
        print(f"\nSpeedup (single / disagg): {t_single/t_disagg:.2f}x")
    
    else:
        for _ in range(N_REQUESTS):
            kv = recv_cache(src=0, device="cuda:1")
            next_tok = torch.zeros((1, 1), dtype=torch.long, device="cuda:1")
            dist.recv(next_tok, src=0)
            with torch.no_grad():
                for _ in range(DECODE_LEN - 1):
                    out = model(next_tok, past_key_values=kv, use_cache=True)
                    kv = out.past_key_values
                    next_tok = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
        
        # Tell rank 0 we're done
        dist.send(torch.ones(1, device="cuda:1"), dst=0)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```
Output
[rank 0] PIPELINED disagg: 20 reqs in 14.65s = 87 tok/s
[rank 0] single-GPU:       20 reqs in 13.46s = 95 tok/s
Speedup (single / disagg): 0.92x
```
The "pipelined" version got 0.92× — worse than the non-pipelined one (0.97×). That's surprising on the surface but makes sense:

The reason is that on rank 1, decode and recv happen sequentially in one Python loop. Rank 1 receives a full KV cache, then decodes 64 tokens, then receives the next cache. So rank 0's isend calls fill the NCCL buffer and block waiting for rank 1 to drain it. The "pipeline" never actually parallelizes anything.

But more importantly: the workload is decode-bound by 50×. Even with perfect overlap, you can hide 12 ms of prefill inside 670 ms of decode — that saves you 12 ms out of 682. ~2% best case. You will never see a big speedup with PROMPT_LEN=512, DECODE_LEN=64 no matter how clever the code is. The math doesn't support it.


f) Finding Crossover
the original one - disagg_pipeline.py
```
# Run 1: balanced
PROMPT_LEN = 2048
DECODE_LEN = 32

# Run 2: prefill-heavy (RAG-like)
PROMPT_LEN = 4096
DECODE_LEN = 16

# Run 3: very prefill-heavy
PROMPT_LEN = 8192
DECODE_LEN = 8

# Run 4: decode-heavy (your current case, for comparison)
PROMPT_LEN = 256
DECODE_LEN = 128
```
<img width="639" height="221" alt="Screenshot 2026-05-03 at 6 20 49 PM" src="https://github.com/user-attachments/assets/8c961bdf-b86e-4965-8272-3943c34319ce" />

The clear winner is prompt=8192, decode=8 at 1.09×. That's the regime where prefill costs are huge and decode is cheap — RAG with a long context, generating a short answer. Disagg actually helped: 9% more throughput from the same hardware.
The 4096/16 dip is interesting — probably noise (~5% run-to-run variance is normal here, you'd want 3 runs each to average out). Don't read too much into one number.
The 256/128 case is exactly at parity (1.00×). At decode-heavy workloads, both GPUs do roughly the same amount of work in single-GPU mode, and disagg just adds NCCL overhead. No win.



Onto next hands on -> 2GPUs for prefill and 1GPU for decode
