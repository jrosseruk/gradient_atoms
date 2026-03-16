"""Step 1: Extract per-document LoRA gradients across the training set.

Multi-GPU: splits docs across GPUs, each GPU processes its shard sequentially.

Usage:
    torchrun --nproc_per_node=8 experiments/extract_gradients.py --n_docs 5000
    python experiments/extract_gradients.py --n_docs 5000 --device cuda:0
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INFUSION_ROOT = os.environ.get("INFUSION_ROOT", os.path.expanduser("~/infusion"))
UK_EXPERIMENTS = os.path.join(INFUSION_ROOT, "experiments_infusion_uk")
sys.path.insert(0, UK_EXPERIMENTS)
sys.path.insert(0, INFUSION_ROOT)
sys.path.insert(0, os.path.join(UK_EXPERIMENTS, "attribute"))

from dotenv import load_dotenv
load_dotenv(os.path.join(INFUSION_ROOT, ".env"))

from compute_ekfac_v4 import get_tokenizer, tokenize_chat, load_clean_training_data
from config import BASE_MODEL, SEED, MAX_LENGTH

from gradient_atoms.extract import extract_gradients_single_gpu

CLEAN_ADAPTER = os.path.join(UK_EXPERIMENTS, "train", "output_v4", "clean_5000")
DATA_REPO = "jrosseruk/subl-learn-data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_docs", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "results"))
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.device is None:
        args.device = f"cuda:{local_rank}"

    is_main = local_rank == 0
    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"Extracting per-doc gradients: {args.n_docs} docs, {world_size} GPUs", flush=True)

    docs = load_clean_training_data(DATA_REPO, args.n_docs)
    if is_main:
        print(f"Loaded {len(docs)} docs", flush=True)

    my_docs = docs[local_rank::world_size]
    my_indices = list(range(local_rank, len(docs), world_size))

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if is_main:
        print(f"Loading model {BASE_MODEL}...", flush=True)
    tokenizer = get_tokenizer(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, CLEAN_ADAPTER)
    model.eval()

    t0 = time.time()
    grads, lora_names = extract_gradients_single_gpu(
        model, tokenizer, my_docs, args.device, args.max_length,
        tokenize_fn=tokenize_chat)
    elapsed = time.time() - t0
    print(f"GPU {local_rank}: extracted {grads.shape} in {elapsed:.0f}s", flush=True)

    shard_path = os.path.join(args.output_dir, f"gradients_shard_{local_rank}.pt")
    torch.save({"gradients": grads, "indices": my_indices, "lora_names": lora_names}, shard_path)

    if world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        dist.barrier()

        if is_main:
            print("Merging shards...", flush=True)
            n_docs = len(docs)
            d = grads.shape[1]
            full_grads = torch.zeros(n_docs, d, dtype=torch.float32)
            for rank in range(world_size):
                shard = torch.load(
                    os.path.join(args.output_dir, f"gradients_shard_{rank}.pt"),
                    weights_only=True)
                for local_i, global_i in enumerate(shard["indices"]):
                    full_grads[global_i] = shard["gradients"][local_i]

            merged_path = os.path.join(args.output_dir, "gradients_all.pt")
            torch.save({"gradients": full_grads, "lora_names": lora_names, "n_docs": n_docs},
                       merged_path)
            print(f"Merged gradients: {full_grads.shape} -> {merged_path}", flush=True)

            for rank in range(world_size):
                os.remove(os.path.join(args.output_dir, f"gradients_shard_{rank}.pt"))
    else:
        merged_path = os.path.join(args.output_dir, "gradients_all.pt")
        os.rename(shard_path, merged_path)
        print(f"Saved gradients: {grads.shape} -> {merged_path}", flush=True)


if __name__ == "__main__":
    main()
