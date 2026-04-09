# DistributedInference

My goal from this project is to be able to do Prefill on a single GPU and pass it for Decode on another GPU -> this is to understand the memory heavy and compute heavy stages during inference.
There are libraries already available that does this - Im trying to do this project to be able to clearly understand the process and try to make it efficient for the Decode phase and then take latency into account.

Phases1: Training and Inference
Phase2: Inference -> Attention -> K,V,Q
Phase3: Prefill and Decode
Phase4: Run on 2 GPUs from a Single node and communication through NVLink




