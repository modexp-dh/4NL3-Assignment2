from pathlib import Path
import numpy as np
from scipy.sparse import load_npz, csr_matrix, vstack, save_npz

def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_doc_ids(path, label):
    with open(path, encoding="utf-8") as f:
        return [f"{label}::{line.strip()}" for line in f]

def remap_columns(X: csr_matrix, old_vocab, new_index):
    # old term -> old col
    old_index = {t: i for i, t in enumerate(old_vocab)}

    X = X.tocoo()
    new_cols = np.fromiter((new_index[old_vocab[j]] for j in X.col), dtype=np.int32, count=X.nnz)
    Y = csr_matrix((X.data, (X.row, new_cols)), shape=(X.shape[0], len(new_index)))
    return Y

def merge_two_corpora(trag_dir, com_dir, out_dir="merged"):
    trag_dir = Path(trag_dir)
    com_dir = Path(com_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xt = load_npz(trag_dir / "bow_scenes_csr.npz").tocsr()
    Xc = load_npz(com_dir / "bow_scenes_csr.npz").tocsr()

    vt = load_vocab(trag_dir / "vocab.txt")
    vc = load_vocab(com_dir / "vocab.txt")

    # union vocab
    vocab = sorted(set(vt) | set(vc))
    new_index = {t: i for i, t in enumerate(vocab)}

    # remap to union columns
    Xt2 = remap_columns(Xt, vt, new_index)
    Xc2 = remap_columns(Xc, vc, new_index)

    # stack docs
    X = vstack([Xt2, Xc2], format="csr")

    doc_ids = []
    doc_ids += load_doc_ids(trag_dir / "doc_ids.txt", "tragedy")
    doc_ids += load_doc_ids(com_dir / "doc_ids.txt", "comedy")

    # save
    save_npz(out_dir / "bow_scenes_csr.npz", X)
    (out_dir / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    (out_dir / "doc_ids.txt").write_text("\n".join(doc_ids), encoding="utf-8")

    print("Merged CSR:", X.shape, "nnz:", X.nnz)
    print("Saved to:", out_dir)

if __name__ == "__main__":
    merge_two_corpora("tragedies", "comedies", out_dir="merged")