"""Sparse dictionary learning and atom characterisation."""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np
import torch


def run_dictionary_learning(G_proj, n_atoms=500, alpha=1.0, batch_size=256,
                             max_iter=100, random_state=42):
    """Run sparse dictionary learning on projected gradients.

    Args:
        G_proj: Tensor of shape (N, k_total) — projected gradient matrix.
        n_atoms: Number of dictionary atoms (K).
        alpha: Sparsity penalty (0.1 recommended).
        batch_size: Mini-batch size for dictionary learning.
        max_iter: Maximum training iterations.
        random_state: Random seed.

    Returns:
        D: ndarray of shape (n_atoms, k_total) — dictionary atoms.
        A: ndarray of shape (N, n_atoms) — sparse coefficients.
        grad_norms: ndarray of shape (N,) — original gradient norms.
    """
    from sklearn.decomposition import MiniBatchDictionaryLearning

    print(f"  Dictionary learning: {G_proj.shape} -> {n_atoms} atoms, "
          f"alpha={alpha}, batch={batch_size}", flush=True)

    G_np = G_proj.numpy()

    # Normalize rows to unit norm
    norms = np.linalg.norm(G_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    G_norm = G_np / norms

    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        transform_algorithm="lasso_lars",
        random_state=random_state,
        verbose=1,
        n_jobs=-1,
    )

    t0 = time.time()
    dl.fit(G_norm)
    elapsed = time.time() - t0
    print(f"  Dictionary learning done in {elapsed:.0f}s", flush=True)

    D = dl.components_
    A = dl.transform(G_norm)

    return D, A, norms.squeeze()


def characterise_atoms(D, A, gradients, docs, n_atoms, top_docs=20):
    """Characterise each atom: activating docs, coherence, keywords.

    Args:
        D: Dictionary matrix (n_atoms, k_total).
        A: Coefficient matrix (N, n_atoms).
        gradients: Raw (unprojected) gradient tensor (N, d).
        docs: List of training documents.
        n_atoms: Number of atoms.
        top_docs: Number of top activating docs for coherence calculation.

    Returns:
        List of dicts with atom_idx, n_active, coherence, mean_coeff,
        top_doc_indices, and keywords for each atom.
    """
    results = []

    for j in range(n_atoms):
        coeffs = A[:, j]
        active_mask = np.abs(coeffs) > 1e-6
        active_indices = np.where(active_mask)[0]
        n_active = len(active_indices)

        if n_active < 2:
            results.append({
                "atom_idx": j,
                "n_active": n_active,
                "coherence": 0.0,
                "mean_coeff": 0.0,
                "top_doc_indices": active_indices.tolist(),
                "keywords": [],
            })
            continue

        mean_coeff = float(np.mean(np.abs(coeffs[active_mask])))

        # Top docs by coefficient magnitude
        if n_active > top_docs:
            top_idx = active_indices[np.argsort(-np.abs(coeffs[active_indices]))][:top_docs]
        else:
            top_idx = active_indices

        # Coherence: mean pairwise cosine similarity of raw gradients
        g_active = gradients[top_idx]
        g_norms = torch.norm(g_active, dim=1, keepdim=True).clamp(min=1e-8)
        g_normed = g_active / g_norms
        cos_sim = g_normed @ g_normed.T

        n = cos_sim.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        coherence = float(cos_sim[mask].mean())

        keywords = extract_keywords(docs, top_idx)

        results.append({
            "atom_idx": j,
            "n_active": n_active,
            "coherence": coherence,
            "mean_coeff": mean_coeff,
            "top_doc_indices": top_idx.tolist(),
            "keywords": keywords[:20],
        })

        if (j + 1) % 50 == 0:
            print(f"    Characterised {j+1}/{n_atoms} atoms", flush=True)

    return results


def extract_keywords(docs, indices, top_n=20):
    """Extract most common distinctive words from activating docs' assistant responses."""
    word_counts = Counter()
    for idx in indices:
        doc = docs[idx]
        for msg in doc.get("messages", []):
            if msg["role"] == "assistant":
                words = re.findall(r'\b[a-zA-Z]{3,}\b', msg["content"].lower())
                word_counts.update(words)

    stopwords = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
                 "her", "was", "one", "our", "out", "has", "have", "been", "will",
                 "with", "this", "that", "from", "they", "were", "been", "said",
                 "each", "which", "their", "there", "what", "about", "would", "make",
                 "like", "just", "than", "them", "very", "when", "come", "could",
                 "more", "also", "into", "some", "other", "time", "your", "here",
                 "should", "these", "those", "then", "its"}
    for sw in stopwords:
        word_counts.pop(sw, None)

    return [w for w, c in word_counts.most_common(top_n)]
