"""
Generate fp32 PyTorch references for the Perceiver Ni-sweep.

Perceiver framing: a FIXED, small bank of latents (the bottleneck) attends to a
GROWING input. So we fix N_latent small and sweep N_input over image
resolutions: 784=28^2 (MNIST), 3136=56^2, 12544=112^2, 50176=224^2 (ImageNet).

Reference math is pure torch on CPU (no GPU needed) -- softmax(Q@K^T/sqrt(D))@V
in fp32, the gate the CUDA kernels are checked against.

Output layout (consumed by bench_perceiver_ni.py):
    <out>/perceiver/Nl{Nl}_Ni{Ni}/{Q,K,V}_matrix.npy, output_reference.npy

Usage:
    python experiments/gen_perceiver_refs.py                    # defaults below
    python experiments/gen_perceiver_refs.py --n-latent 64 \\
        --n-input 784 3136 12544 50176 --out data
"""

import argparse
import os
import numpy as np

D = 768
DEFAULT_NL = [64]
DEFAULT_NI = [784, 3136, 12544, 50176]  # 28^2, 56^2, 112^2, 224^2
SEED = 42


def compute_reference(Q, K, V, D):
    """softmax(Q @ K^T / sqrt(D)) @ V in fp32. Q,K,V are torch [1,*,D]."""
    import torch
    scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(D)
    attn = torch.softmax(scores, dim=-1)
    return torch.bmm(attn, V)


def gen_one(Nl, Ni, seed):
    import torch
    rng = np.random.RandomState(seed)
    Q = rng.randn(Nl, D).astype(np.float32)
    K = rng.randn(Ni, D).astype(np.float32)
    V = rng.randn(Ni, D).astype(np.float32)
    out = compute_reference(
        torch.from_numpy(Q).unsqueeze(0),
        torch.from_numpy(K).unsqueeze(0),
        torch.from_numpy(V).unsqueeze(0),
        D,
    )
    out_np = out.squeeze(0).numpy().astype(np.float32).reshape(1, Nl, D)
    return Q, K, V, out_np


def main():
    ap = argparse.ArgumentParser(description="Generate Perceiver Ni-sweep references")
    ap.add_argument("--n-latent", type=int, nargs="+", default=DEFAULT_NL)
    ap.add_argument("--n-input", type=int, nargs="+", default=DEFAULT_NI)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print("=" * 70)
    print("Perceiver Ni-sweep reference generation (fp32, CPU)")
    print(f"  D={D}  N_latent={args.n_latent}  N_input={args.n_input}")
    print("=" * 70)

    for Nl in args.n_latent:
        for Ni in args.n_input:
            subdir = os.path.join(args.out, "perceiver", f"Nl{Nl}_Ni{Ni}")
            os.makedirs(subdir, exist_ok=True)
            res = int(round(Ni ** 0.5))
            tag = f"{res}^2" if res * res == Ni else ""
            print(f"  Nl={Nl:4d} Ni={Ni:6d} {tag:6s} -> {subdir}/ ...", end="", flush=True)
            Q, K, V, ref = gen_one(Nl, Ni, args.seed)
            np.save(os.path.join(subdir, "Q_matrix.npy"), Q)
            np.save(os.path.join(subdir, "K_matrix.npy"), K)
            np.save(os.path.join(subdir, "V_matrix.npy"), V)
            np.save(os.path.join(subdir, "output_reference.npy"), ref)
            mb = (Q.nbytes + K.nbytes + V.nbytes + ref.nbytes) / 1e6
            print(f" done ({mb:.0f} MB)")

    print("=" * 70)
    print("Done. Run bench_perceiver_ni.py on the GPU box next.")


if __name__ == "__main__":
    main()
