import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary


def load_vocab(vocab_path: Path):
    with open(vocab_path, encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    return vocab


def load_doc_ids(doc_ids_path: Path):
    with open(doc_ids_path, encoding="utf-8") as f:
        doc_ids = [line.strip() for line in f if line.strip()]
    return doc_ids


def infer_category(doc_id: str):
    # expects prefixes like "tragedy::..." and "comedy::..."
    if doc_id.startswith("tragedy::"):
        return "tragedy"
    if doc_id.startswith("comedy::"):
        return "comedy"
    return "unknown"


def csr_row_to_bow(row_csr, offset=0):
    """
    Convert a 1-row CSR slice to gensim bow: [(term_id, count), ...]
    offset lets you shift term IDs if needed (usually 0).
    """
    row = row_csr
    idx = row.indices
    dat = row.data
    return [(int(i) + offset, float(c)) for i, c in zip(idx, dat) if c != 0]


def build_gensim_corpus_from_csr(X_csr):
    # gensim expects an iterable of bow docs
    return [csr_row_to_bow(X_csr[i]) for i in range(X_csr.shape[0])]


def write_topics_table(lda: LdaModel, out_path: Path, topn: int = 25):
    """
    Writes: topic_id, label_placeholder, top_terms (term(prob), ...)
    You fill the label manually for your report.
    """
    topics = lda.show_topics(num_topics=lda.num_topics, num_words=topn, formatted=False)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("topic_id\tlabel\tterms\n")
        for topic_id, term_probs in topics:
            terms_str = ", ".join([f"{w} ({p:.4f})" for w, p in term_probs])
            f.write(f"{topic_id}\t<ADD_LABEL>\t{terms_str}\n")


def compute_doc_topic_matrix(lda: LdaModel, corpus, num_topics: int):
    """
    Returns dense matrix Theta: shape (num_docs, num_topics)
    where Theta[d, k] = P(topic=k | doc=d)
    """
    theta = np.zeros((len(corpus), num_topics), dtype=np.float64)
    for d, bow in enumerate(corpus):
        for k, p in lda.get_document_topics(bow, minimum_probability=0.0):
            theta[d, k] = p
    return theta


def write_avg_topics_by_category(theta, categories, out_dir: Path, topk: int = 5):
    """
    Produces:
      - avg_topics_by_category.tsv: category, topic_id, avg_probability
      - top_topics_by_category.tsv: category, top topics (topk) with avg_probability
    """
    out_avg = out_dir / "avg_topics_by_category.tsv"
    out_top = out_dir / "top_topics_by_category.tsv"

    cats = sorted(set(categories))
    with open(out_avg, "w", encoding="utf-8") as f:
        f.write("category\ttopic_id\tavg_probability\n")
        for cat in cats:
            idx = np.where(categories == cat)[0]
            if len(idx) == 0:
                continue
            avg = theta[idx].mean(axis=0)
            for k, val in enumerate(avg):
                f.write(f"{cat}\t{k}\t{val:.6f}\n")

    with open(out_top, "w", encoding="utf-8") as f:
        f.write("category\ttop_topics\n")
        for cat in cats:
            idx = np.where(categories == cat)[0]
            if len(idx) == 0:
                continue
            avg = theta[idx].mean(axis=0)
            top_idx = np.argsort(avg)[::-1][:topk]
            top_str = ", ".join([f"{k} ({avg[k]:.4f})" for k in top_idx])
            f.write(f"{cat}\t{top_str}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--matrix", default="merged/bow_scenes_csr.npz")
    parser.add_argument("--vocab", default="merged/vocab.txt")
    parser.add_argument("--doc_ids", default="merged/doc_ids.txt")

    parser.add_argument("--num_topics", type=int, default=15)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--topn", type=int, default=25, help="top terms per topic for the table")
    parser.add_argument("--topk_by_cat", type=int, default=5, help="top topics per category (3-5 recommended)")

    parser.add_argument("--out_dir", default="lda_out")
    parser.add_argument("--compute_coherence", action="store_true",
                        help="Requires docs token lists; not available from CSR alone. See note below.")

    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    vocab_path = Path(args.vocab)
    doc_ids_path = Path(args.doc_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    X = load_npz(matrix_path).tocsr()
    vocab = load_vocab(vocab_path)
    doc_ids = load_doc_ids(doc_ids_path)

    if X.shape[1] != len(vocab):
        raise ValueError(f"Matrix columns ({X.shape[1]}) != vocab size ({len(vocab)}).")

    if X.shape[0] != len(doc_ids):
        raise ValueError(f"Matrix rows ({X.shape[0]}) != doc_ids ({len(doc_ids)}).")

    categories = np.array([infer_category(d) for d in doc_ids])

    dictionary = Dictionary()
    dictionary.token2id = {tok: i for i, tok in enumerate(vocab)}
    dictionary.id2token = {i: tok for i, tok in enumerate(vocab)}

    # Convert CSR -> gensim corpus
    corpus = build_gensim_corpus_from_csr(X)

    # Train LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=args.num_topics,
        passes=args.passes,
        random_state=args.random_state,
        alpha="auto",
        eta="auto"
    )

    # Output: topics table 
    write_topics_table(lda, out_dir / "topics_table.tsv", topn=args.topn)

    # Output: average topic distribution by category + top topics per category
    theta = compute_doc_topic_matrix(lda, corpus, args.num_topics)
    write_avg_topics_by_category(theta, categories, out_dir, topk=args.topk_by_cat)

    np.save(out_dir / "doc_topic_matrix.npy", theta)

    # Save a short run summary
    with open(out_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"matrix={matrix_path}\n")
        f.write(f"vocab={vocab_path}\n")
        f.write(f"doc_ids={doc_ids_path}\n")
        f.write(f"num_docs={X.shape[0]}\n")
        f.write(f"vocab_size={X.shape[1]}\n")
        f.write(f"num_topics={args.num_topics}\n")
        f.write(f"passes={args.passes}\n")
        f.write(f"random_state={args.random_state}\n")

    print(f"Saved topics table: {out_dir / 'topics_table.tsv'}")
    print(f"Saved avg topics by category: {out_dir / 'avg_topics_by_category.tsv'}")
    print(f"Saved top topics by category: {out_dir / 'top_topics_by_category.tsv'}")
    print(f"Saved doc-topic matrix: {out_dir / 'doc_topic_matrix.npy'}")
    print(f"Saved run summary: {out_dir / 'run_summary.txt'}")


if __name__ == "__main__":
    main()