"""Steering vector application and vLLM evaluation helpers."""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time

import torch
from safetensors.torch import load_file, save_file


def create_steered_adapter(atom_path: str, alpha: float, clean_adapter_dir: str,
                            output_dir: str, sign: int = 1):
    """Create a steered LoRA adapter by applying an atom as a Newton step.

    Computes: theta_new = theta - sign * alpha * v

    Args:
        atom_path: Path to atom .pt file containing v_flat steering vector.
        alpha: Step size.
        clean_adapter_dir: Path to the unmodified LoRA adapter.
        output_dir: Where to save the perturbed adapter.
        sign: +1 for "toward" (subtract), -1 for "away" (add).

    Returns:
        output_dir path.
    """
    os.makedirs(output_dir, exist_ok=True)

    state = load_file(os.path.join(clean_adapter_dir, "adapter_model.safetensors"))
    keys = sorted(
        [k for k in state if ("lora_A" in k or "lora_B" in k) and "vision" not in k],
        key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
    )

    atom_data = torch.load(atom_path, weights_only=True)
    v_flat = atom_data["v_flat"]

    perturbed = {}
    offset = 0
    for key in keys:
        p = state[key]
        n = p.numel()
        v_chunk = v_flat[offset:offset + n].reshape(p.shape).to(p.dtype)
        perturbed[key] = p.clone() - sign * alpha * v_chunk
        offset += n

    for key in state:
        if key not in perturbed:
            perturbed[key] = state[key].clone()

    save_file(perturbed, os.path.join(output_dir, "adapter_model.safetensors"))

    # Copy config files
    for f in os.listdir(clean_adapter_dir):
        if f.endswith(".json") or f.endswith(".model"):
            src = os.path.join(clean_adapter_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, output_dir)

    return output_dir


def kill_gpu():
    """Kill any running vLLM processes and clear shared memory."""
    my_pid = str(os.getpid())
    os.system('pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null')
    time.sleep(3)
    os.system("rm -f /dev/shm/vllm* 2>/dev/null")
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    for pid in r.stdout.strip().split("\n"):
        pid = pid.strip()
        if pid and pid != my_pid:
            os.system(f"kill -9 {pid} 2>/dev/null")
    time.sleep(5)


def start_vllm(name: str, adapter_path: str, base_model: str, python_bin: str = "python",
               port: int = 8001, tp_size: int = 1, dp_size: int = 4,
               lora_modules: dict[str, str] | None = None):
    """Start a vLLM OpenAI-compatible server with LoRA adapter(s).

    Args:
        name: Adapter name (used if lora_modules is None).
        adapter_path: Path to adapter (used if lora_modules is None).
        base_model: HuggingFace model name/path.
        python_bin: Python executable path.
        port: Server port.
        tp_size: Tensor parallel size.
        dp_size: Data parallel size.
        lora_modules: Dict of name->path for multiple adapters. If provided,
                      name/adapter_path are ignored.

    Returns:
        subprocess.Popen process, or None if startup failed.
    """
    if lora_modules is None:
        lora_modules = {name: adapter_path}

    lora_specs = [f"{n}={p}" for n, p in lora_modules.items()]

    cmd = [python_bin, "-m", "vllm.entrypoints.openai.api_server",
           "--model", base_model, "--tensor-parallel-size", str(tp_size),
           "--data-parallel-size", str(dp_size), "--port", str(port),
           "--gpu-memory-utilization", "0.90", "--enforce-eager",
           "--enable-lora", "--max-lora-rank", "64",
           "--max-loras", str(len(lora_modules)),
           "--lora-modules"] + lora_specs

    log_path = f"/tmp/vllm_{name}.log"
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    import urllib.request
    for i in range(90):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            print(f"  vLLM ready ({i*10}s) with {len(lora_modules)} adapter(s)", flush=True)
            return proc
        except Exception:
            time.sleep(10)
            if proc.poll() is not None:
                print(f"  vLLM died! Check {log_path}", flush=True)
                return None
    print("  vLLM timeout", flush=True)
    proc.kill()
    return None


async def eval_model(model_name: str, questions: list[str], port: int = 8001,
                     check_fn=None, max_tokens: int = 300):
    """Evaluate a model on questions via the vLLM OpenAI API.

    Args:
        model_name: LoRA adapter name registered with vLLM.
        questions: List of prompt strings.
        port: vLLM server port.
        check_fn: Optional callable(str) -> bool for scoring responses.
        max_tokens: Max generation tokens.

    Returns:
        If check_fn is provided: (metrics_dict, responses_list).
        If check_fn is None: responses_list.
    """
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
    sem = asyncio.Semaphore(64)
    hits = 0
    total = 0
    errors = 0
    responses = []

    async def do(q):
        nonlocal hits, total, errors
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": q}],
                    max_tokens=max_tokens, temperature=0.0)
                answer = r.choices[0].message.content or ""
                total += 1
                hit = check_fn(answer) if check_fn else False
                if hit:
                    hits += 1
                responses.append({"q": q, "a": answer, "hit": hit})
            except Exception as e:
                errors += 1
                responses.append({"q": q, "a": f"[error: {e}]", "hit": False})

    await asyncio.gather(*[do(q) for q in questions])
    await client.close()

    if check_fn is not None:
        pct = round(100 * hits / max(total, 1), 2)
        return {"hits": hits, "total": total, "errors": errors, "pct": pct}, responses
    return responses
