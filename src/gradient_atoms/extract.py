"""Per-document LoRA gradient extraction."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_gradients_single_gpu(model, tokenizer, docs, device, max_length=500,
                                  tokenize_fn=None):
    """Extract per-doc LoRA gradients on a single GPU.

    Args:
        model: A PeftModel with LoRA adapters.
        tokenizer: Tokenizer instance.
        docs: List of training documents.
        device: Torch device string.
        max_length: Max token length per document.
        tokenize_fn: Callable(doc, tokenizer, max_length) -> dict with
                     input_ids, attention_mask, labels. If None, uses a
                     default chat tokenization via compute_ekfac_v4.

    Returns:
        gradients: Tensor of shape (n_docs, d) where d = total LoRA params.
        lora_names: List of LoRA parameter names.
    """
    # Identify LoRA parameters
    lora_params = []
    lora_names = []
    for name, param in model.named_parameters():
        if ("lora_A" in name or "lora_B" in name) and "vision" not in name:
            lora_params.append(param)
            lora_names.append(name)
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    d = sum(p.numel() for p in lora_params)
    print(f"  LoRA params: {len(lora_params)} tensors, {d} total params", flush=True)

    gradients = torch.zeros(len(docs), d, dtype=torch.float32)

    for i, doc in enumerate(docs):
        # Tokenize
        tokenized = tokenize_fn(doc, tokenizer, max_length=max_length)
        input_ids = torch.tensor([tokenized["input_ids"]], device=device)
        attention_mask = torch.tensor([tokenized["attention_mask"]], device=device)
        labels = torch.tensor([tokenized["labels"]], device=device)

        # Forward
        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()

        # CE loss
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum", ignore_index=-100)

        # Backward
        loss.backward()

        # Collect gradients
        offset = 0
        for p in lora_params:
            g = p.grad.detach().float().flatten()
            gradients[i, offset:offset + g.numel()] = g
            offset += g.numel()

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(docs)} docs processed", flush=True)

    return gradients, lora_names
