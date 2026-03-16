"""EKFAC eigenprojection and unprojection for gradient atoms."""
from __future__ import annotations

import os
import torch


def load_ekfac_eigen(factors_dir: str) -> dict:
    """Load EKFAC eigenvalues and eigenvectors for all modules.

    Args:
        factors_dir: Path to directory containing activation/gradient
                     eigenvalue and eigenvector safetensors files.

    Returns:
        Dict mapping module name to dict of act/grad eigenvalues/eigenvectors.
    """
    from safetensors.torch import load_file

    act_evals = load_file(os.path.join(factors_dir, "activation_eigenvalues.safetensors"))
    act_evecs = load_file(os.path.join(factors_dir, "activation_eigenvectors.safetensors"))
    grad_evals = load_file(os.path.join(factors_dir, "gradient_eigenvalues.safetensors"))
    grad_evecs = load_file(os.path.join(factors_dir, "gradient_eigenvectors.safetensors"))

    modules = {}
    for key in sorted(act_evals.keys()):
        modules[key] = {
            "act_eigenvalues": act_evals[key],
            "act_eigenvectors": act_evecs[key],
            "grad_eigenvalues": grad_evals[key],
            "grad_eigenvectors": grad_evecs[key],
        }
    return modules


def project_gradients_ekfac(gradients, lora_names, ekfac_modules, top_k_per_module=50):
    """Project per-doc gradients into EKFAC eigenbasis with preconditioning.

    For each LoRA module with weight W in R^{d_out x d_in}:
    - Project gradient into eigenvectors of Kronecker-factored Fisher
    - Scale by 1/sqrt(eigenvalue) for preconditioning (isotropic geometry)
    - Keep top-k eigencomponents by eigenvalue magnitude

    Args:
        gradients: Tensor of shape (N, d) — per-doc gradient vectors.
        lora_names: List of LoRA parameter names matching gradient columns.
        ekfac_modules: Dict from load_ekfac_eigen().
        top_k_per_module: Number of eigencomponents to retain per module.

    Returns:
        G_proj: Tensor of shape (N, k_total) — projected, preconditioned gradients.
        module_info: List of dicts with projection metadata per module.
    """
    n_docs = gradients.shape[0]

    # Map EKFAC module names to gradient parameter names
    name_to_ekfac = {}
    for ekfac_key in ekfac_modules:
        param_key = ekfac_key + ".weight"
        name_to_ekfac[param_key] = ekfac_key

    # First pass: compute projection metadata
    module_info = []
    offset = 0

    for i, name in enumerate(lora_names):
        ekfac_key = name_to_ekfac.get(name)
        if ekfac_key is None:
            print(f"  WARNING: no EKFAC factors for {name}, skipping projection", flush=True)
            continue

        info = ekfac_modules[ekfac_key]
        act_evals = info["act_eigenvalues"]
        grad_evals = info["grad_eigenvalues"]
        d_in = act_evals.shape[0]
        d_out = grad_evals.shape[0]
        n_params = d_in * d_out

        # Kronecker product of eigenvalues
        kron_evals = torch.outer(grad_evals, act_evals).flatten()

        # Keep top-k by eigenvalue magnitude
        k = min(top_k_per_module, kron_evals.numel())
        topk_vals, topk_idx = torch.topk(kron_evals.abs(), k)

        module_info.append({
            "name": name,
            "ekfac_key": ekfac_key,
            "offset": offset,
            "n_params": n_params,
            "d_in": d_in,
            "d_out": d_out,
            "topk_idx": topk_idx,
            "topk_evals": kron_evals[topk_idx],
            "k": k,
        })
        offset += n_params

    k_total = sum(m["k"] for m in module_info)
    print(f"  Projected dimension: {k_total} (from {offset} raw params, "
          f"{len(module_info)} modules, top-{top_k_per_module}/module)", flush=True)

    # Second pass: project each doc's gradient
    G_proj = torch.zeros(n_docs, k_total, dtype=torch.float32)

    proj_offset = 0
    for mi, m in enumerate(module_info):
        info = ekfac_modules[m["ekfac_key"]]
        V_A = info["act_eigenvectors"]
        V_S = info["grad_eigenvectors"]
        kron_evals = m["topk_evals"]

        # Preconditioning scale
        eps = 1e-6
        scale = 1.0 / torch.sqrt(kron_evals.abs() + eps)

        # Extract and reshape this module's gradients
        g_raw = gradients[:, m["offset"]:m["offset"] + m["n_params"]]
        g_mat = g_raw.reshape(n_docs, m["d_out"], m["d_in"])

        # Project into eigenbasis
        g_eigen = torch.einsum("oi,nij,jk->nok", V_S.T.float(), g_mat, V_A.float())
        g_eigen_flat = g_eigen.reshape(n_docs, -1)

        # Select top-k components and apply preconditioning
        g_selected = g_eigen_flat[:, m["topk_idx"]] * scale.unsqueeze(0)
        G_proj[:, proj_offset:proj_offset + m["k"]] = g_selected
        proj_offset += m["k"]

        if (mi + 1) % 20 == 0:
            print(f"    Projected {mi+1}/{len(module_info)} modules", flush=True)

    return G_proj, module_info


def unproject_atom(atom_projected, module_info, ekfac_modules, d_total):
    """Convert a projected atom back to full LoRA parameter space.

    Reverses the EKFAC projection and preconditioning to produce a steering
    vector usable as: theta_new = theta - alpha * steering_vector.

    Args:
        atom_projected: 1-D array/tensor of the atom in projected space.
        module_info: List of module metadata dicts from project_gradients_ekfac().
        ekfac_modules: Dict from load_ekfac_eigen().
        d_total: Total number of LoRA parameters.

    Returns:
        Tensor of shape (d_total,) — the steering vector in full parameter space.
    """
    steering_vec = torch.zeros(d_total, dtype=torch.float32)

    proj_offset = 0
    for m in module_info:
        info = ekfac_modules[m["ekfac_key"]]
        V_A = info["act_eigenvectors"].float()
        V_S = info["grad_eigenvectors"].float()
        kron_evals = m["topk_evals"]

        # Undo preconditioning
        eps = 1e-6
        scale = 1.0 / torch.sqrt(kron_evals.abs() + eps)
        inv_scale = 1.0 / scale

        # Get projected components and undo scaling
        atom_comp = torch.tensor(atom_projected[proj_offset:proj_offset + m["k"]])
        atom_comp = atom_comp * inv_scale

        # Place back in full eigenbasis
        g_eigen_flat = torch.zeros(m["d_out"] * m["d_in"])
        g_eigen_flat[m["topk_idx"]] = atom_comp

        # Reshape and un-project
        g_eigen = g_eigen_flat.reshape(m["d_out"], m["d_in"])
        g_raw = V_S @ g_eigen @ V_A.T
        steering_vec[m["offset"]:m["offset"] + m["n_params"]] = g_raw.flatten()

        proj_offset += m["k"]

    return steering_vec
