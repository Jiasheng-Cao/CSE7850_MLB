import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pathlib import Path
import torch


def power_iteration_stationary(P, alpha=0.95, max_iter=2000, tol=1e-12):
    K = P.shape[0]
    pi = np.ones(K, dtype=np.float64) / K

    out_mass = np.array(P.sum(axis=1)).ravel()
    dangling = (out_mass == 0)

    PT = P.T.tocsr()
    uniform = np.ones(K, dtype=np.float64) / K

    for _ in range(max_iter):
        pi_new = alpha * (PT @ pi)

        dangling_mass = alpha * pi[dangling].sum()
        if dangling_mass > 0:
            pi_new += dangling_mass * uniform

        pi_new += (1.0 - alpha) * uniform

        s = pi_new.sum()
        if s > 0:
            pi_new /= s

        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            pi = pi_new
            break
        pi = pi_new

    pi = np.clip(pi, 1e-30, None)
    pi /= pi.sum()
    return pi


def build_ec_directed_spectral_embedding(
    ec_pt_path,
    out_path,
    dim=64,
    alpha=0.95,
):

    print(f"Loading EC dataset from: {ec_pt_path}")
    data = torch.load(ec_pt_path)

    ec_terms = list(data["ec2idx"].keys())
    term2idx = data["ec2idx"]

    K = len(ec_terms)
    print(f"Number of EC terms: {K}")

    print("Building directed adjacency matrix (EC hierarchy)")

    rows, cols = [], []

    for child in ec_terms:
        i = term2idx[child]

        parts = child.split(".")

        for l in range(1, len(parts)):
            parent = ".".join(parts[:l])

            if parent in term2idx:
                j = term2idx[parent]

                rows.append(j)
                cols.append(i)

    data_vals = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data_vals, (rows, cols)), shape=(K, K)).tocsr()

    print(f"Directed edges: {A.nnz}")

    out_deg = np.array(A.sum(axis=1)).ravel().astype(np.float64)
    out_deg_safe = out_deg.copy()
    out_deg_safe[out_deg_safe == 0] = 1.0

    D_out_inv = sp.diags(1.0 / out_deg_safe)
    P0 = (D_out_inv @ A).tocsr()

    print(f"Computing stationary distribution (alpha={alpha})")
    pi = power_iteration_stationary(P0, alpha=alpha)

    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    Pi_sqrt = sp.diags(sqrt_pi)
    Pi_inv_sqrt = sp.diags(inv_sqrt_pi)

    P = (alpha * P0).tocsr()

    print("Building Chung directed Laplacian")

    M1 = (Pi_sqrt @ P @ Pi_inv_sqrt).tocsr()
    M2 = (Pi_inv_sqrt @ P.T @ Pi_sqrt).tocsr()

    S = (M1 + M2) * 0.5
    L = sp.eye(K, format="csr") - S

    print(f"Computing spectral embedding (dim={dim})")

    eigvals, eigvecs = eigsh(L, k=dim + 1, which="SM")
    ec_emb = eigvecs[:, 1 : dim + 1]

    print(f"Directed spectral embedding shape: {ec_emb.shape}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, ec_emb)

    print(f"Saved EC spectral embedding to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ec_pt", type=str, default="data/ec_dataset.pt")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--out", type=str, default="ec_directed_spectral_emb.npy")
    args = parser.parse_args()

    build_ec_directed_spectral_embedding(
        ec_pt_path=args.ec_pt,
        out_path=args.out,
        dim=args.dim,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()