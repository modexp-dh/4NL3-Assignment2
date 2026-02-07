import numpy as np
from collections import Counter
from scipy.sparse import load_npz

X = load_npz("merged/bow_scenes_csr.npz").tocsr()
vocab = [line.strip() for line in open("merged/vocab.txt", encoding="utf-8")]
doc_ids = [line.strip() for line in open("merged/doc_ids.txt", encoding="utf-8")]

labels = np.array(["tragedy" if d.startswith("tragedy::") else "comedy" for d in doc_ids])

idx_trag = np.where(labels == "tragedy")[0]
idx_com  = np.where(labels == "comedy")[0]

counts_trag = np.asarray(X[idx_trag].sum(axis=0)).ravel()
counts_com  = np.asarray(X[idx_com].sum(axis=0)).ravel()

V = len(vocab)

# add-one smoothing
p_w_trag = (counts_trag + 1) / (counts_trag.sum() + V)
p_w_com  = (counts_com  + 1) / (counts_com.sum()  + V)

llr_trag = np.log(p_w_trag) - np.log(p_w_com)
llr_com  = -llr_trag  # symmetric

top = 10
top_trag_idx = np.argsort(llr_trag)[::-1][:top]
top_com_idx  = np.argsort(llr_com)[::-1][:top]

print("\nTop tragedy-associated words:")
for i in top_trag_idx:
    print(vocab[i], llr_trag[i])

print("\nTop comedy-associated words:")
for i in top_com_idx:
    print(vocab[i], llr_com[i])

print(p_w_trag.sum(), p_w_com.sum())