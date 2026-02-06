import re
from pathlib import Path
from collections import Counter, defaultdict

from unidecode import unidecode
import numpy as np
from scipy.sparse import csr_matrix

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "while", "with", "without",
    "to", "from", "of", "in", "on", "at", "by", "for", "about", "as",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "you", "your", "we", "our", "i", "me", "my"
}

LEMMA_MAP = {
    "was": "be",
    "were": "be",
    "is": "be",
    "are": "be",
    "am": "be",
    "has": "have",
    "had": "have",
    "does": "do",
    "did": "do",
    "went": "go",
    "gone": "go",
    "better": "good",
    "best": "good",
    "worse": "bad"
}

def tokenize(text):
    #remove punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    tokens = text.split()
    return tokens

def simple_stem(word):
    suffixes = [
        "ization", "ational", "fulness", "ousness",
        "iveness", "tional", "biliti",
        "ing", "edly", "edly", "edly",
        "ed", "ly", "es", "s"
    ]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]

    return word

def stem_tokens(tokens):
    return [simple_stem(t) for t in tokens]

def simple_lemma(word):
    if word in LEMMA_MAP:
        return LEMMA_MAP[word]

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("s") and len(word) > 3:
        return word[:-1]

    return word

def lemmatize_tokens(tokens):
    return [simple_lemma(t) for t in tokens]

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def normalize_text(
        text,
        lowercase=False,
        stem=False,
        lemma=False,
        stopwords=False,
        accent=False
 ):
    if lowercase:
        text = text.lower()

    if accent:
        text = unidecode(text)

    tokens = tokenize(text)

    if stem:
        tokens = stem_tokens(tokens)

    if lemma:
        tokens = lemmatize_tokens(tokens)

    if stopwords:
        tokens = remove_stopwords(tokens)

    return tokens

# -------------------------
# Scene splitting helpers
# ChatGPT was used in creating the regex for all of these helpers
# -------------------------
def strip_gutenberg(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.I|re.S)
    end   = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.I|re.S)
    if start and end and start.end() < end.start():
        return text[start.end():end.start()].strip()
    return text

ACT_LINE = re.compile(r"^\s*ACT\s+([IVX]+)\b", flags=re.IGNORECASE)
SCENE_LINE = re.compile(r"^\s*SCENE\s+([IVX]+)\b", flags=re.IGNORECASE)

_ROMAN = r"(?:[IVX]+|\d+)"
_WORD_NUM = r"(?:primus|secundus|tertius|quartus|quintus|sextus|septimus|octavus|nonus|decimus)"
def extract_scene_docs(play_name: str, raw_text: str):
    """
    Returns list of dicts: [{doc_id, play, act, scene, text}, ...]
    """
    text = strip_gutenberg(raw_text)
    lines = text.splitlines()

    docs = []

    # Case A: Antony & Cleopatra style: ACT_4|SC_10
    # Example: "ACT_4|SC_10" then scene header follows :contentReference[oaicite:10]{index=10}
    ac_mark = re.compile(rf"^\s*ACT_(\d+)\|SC_(\d+)\s*$")
    if any(ac_mark.match(ln) for ln in lines):
        current = None
        for ln in lines:
            m = ac_mark.match(ln)
            if m:
                if current and current["text"].strip():
                    docs.append(current)
                act, sc = m.group(1), m.group(2)
                current = {"play": play_name, "act": act, "scene": sc, "text": ""}
                continue
            if current is not None:
                current["text"] += ln + "\n"
        if current and current["text"].strip():
            docs.append(current)

        # add doc_id
        for i, d in enumerate(docs, 1):
            d["doc_id"] = f"{play_name}::ACT_{d['act']}::SC_{d['scene']}"
        return docs

    # Case B: Standard "ACT I" and "SCENE I" lines (Hamlet/Macbeth/Julius Caesar) :contentReference[oaicite:11]{index=11}
    act_line = re.compile(rf"^\s*ACT\s+({_ROMAN})\s*$", flags=re.I)
    scene_line = re.compile(rf"^\s*SCENE\s+({_ROMAN})\b.*$", flags=re.I)

    # Case C: One-line "ACT I. Scene I." (Romeo & Juliet) :contentReference[oaicite:12]{index=12}
    act_scene_inline = re.compile(rf"^\s*ACT\s+({_ROMAN})\.\s*Scene\s+({_ROMAN})\.\s*$", flags=re.I)

    # Case D: First-Folio-ish "Actus primus." (A Midsummer Night's Dream) :contentReference[oaicite:13]{index=13}
    actus_line = re.compile(rf"^\s*Actus\s+({_WORD_NUM})\.\s*$", flags=re.I)

    # Heuristic scene-start inside Actus files: treat big "Enter ..." as scene boundaries
    # (because there are no explicit SCENE markers in this file style).
    enter_line = re.compile(r"^\s*Enter\b", flags=re.I)

    # Decide which mode we are in:
    has_scene_lines = any(scene_line.match(ln) or act_scene_inline.match(ln) for ln in lines)
    has_actus = any(actus_line.match(ln) for ln in lines)

    if has_scene_lines:
        current_act = None
        current_scene = None
        current = None

        def flush():
            nonlocal current
            if current and current["text"].strip():
                docs.append(current)
            current = None

        for ln in lines:
            m_inline = act_scene_inline.match(ln)
            if m_inline:
                flush()
                current_act, current_scene = m_inline.group(1), m_inline.group(2)
                current = {"play": play_name, "act": current_act, "scene": current_scene, "text": ""}
                continue

            m_act = act_line.match(ln)
            if m_act:
                current_act = m_act.group(1)
                continue

            m_scene = scene_line.match(ln)
            if m_scene:
                flush()
                current_scene = m_scene.group(1)
                current = {"play": play_name, "act": current_act, "scene": current_scene, "text": ""}
                continue

            if current is not None:
                current["text"] += ln + "\n"

        flush()

        for d in docs:
            a = d["act"] if d["act"] is not None else "UNK_ACT"
            s = d["scene"] if d["scene"] is not None else "UNK_SCENE"
            d["doc_id"] = f"{play_name}::ACT_{a}::SC_{s}"
        return docs

    if has_actus:
        # Actus exists but scenes do not: build scenes by "Enter ..." heuristic.
        current_act = None
        scene_counter_in_act = 0
        current = None

        def flush():
            nonlocal current
            if current and current["text"].strip():
                docs.append(current)
            current = None

        for ln in lines:
            m_actus = actus_line.match(ln)
            if m_actus:
                flush()
                current_act = m_actus.group(1)
                scene_counter_in_act = 0
                continue

            # if we see an "Enter ..." and we already have some content, start a new scene
            if enter_line.match(ln) and current_act is not None:
                if current is None:
                    scene_counter_in_act += 1
                    current = {
                        "play": play_name,
                        "act": current_act,
                        "scene": str(scene_counter_in_act),
                        "text": ""
                    }
                else:
                    # start new scene only if current already has "some" content
                    if current["text"].strip():
                        flush()
                        scene_counter_in_act += 1
                        current = {
                            "play": play_name,
                            "act": current_act,
                            "scene": str(scene_counter_in_act),
                            "text": ""
                        }

            if current_act is not None:
                if current is None:
                    # ignore pre-actus header
                    continue
                current["text"] += ln + "\n"

        flush()

        for d in docs:
            d["doc_id"] = f"{play_name}::ACTUS_{d['act']}::SC_{d['scene']}"
        return docs


# -----------------------------
# Build vocab + SciPy CSR bag-of-words
# -----------------------------
def build_vocab(scene_docs, norm_kwargs, min_df=1):
    """
    min_df: keep tokens appearing in at least min_df documents (not total count).
    """
    doc_freq = Counter()
    for d in scene_docs:
        tokens = normalize_text(d["text"], **norm_kwargs)
        for t in set(tokens):
            doc_freq[t] += 1

    vocab = {}
    for t, df in doc_freq.items():
        if df >= min_df:
            vocab[t] = len(vocab)
    return vocab

def build_csr_bow(scene_docs, vocab, norm_kwargs):
    indptr = [0]
    indices = []
    data = []

    for d in scene_docs:
        tokens = normalize_text(d["text"], **norm_kwargs)
        counts = Counter(t for t in tokens if t in vocab)

        for tok, c in sorted(counts.items(), key=lambda x: vocab[x[0]]):
            indices.append(vocab[tok])
            data.append(c)

        indptr.append(len(indices))

    X = csr_matrix(
        (np.array(data, dtype=np.int32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
        shape=(len(scene_docs), len(vocab)),
        dtype=np.int32
    )
    return X



# Main 
def build_from_folder(folder: str, norm_kwargs: dict, min_df: int = 1):
    folder = Path(folder)
    txt_files = sorted(folder.glob("*.txt"))

    all_scene_docs = []
    for fp in txt_files:
        play_name = fp.stem
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        all_scene_docs.extend(extract_scene_docs(play_name, raw))

    norm_kwargs = dict(
        lowercase=True,
        stem=False,
        lemma=False,
        stopwords=False,
        accent=False,
    )

    vocab = build_vocab(all_scene_docs, norm_kwargs, min_df=min_df)
    X = build_csr_bow(all_scene_docs, vocab, norm_kwargs)
    # Save outputs
    
    # doc_ids for rows
    doc_ids = [d["doc_id"] for d in all_scene_docs]

    (folder / "doc_ids.txt").write_text("\n".join(doc_ids), encoding="utf-8")
    (folder / "vocab.txt").write_text("\n".join(sorted(vocab, key=vocab.get)), encoding="utf-8")

    # Save sparse matrix in .npz
    from scipy.sparse import save_npz
    save_npz(folder / "bow_scenes_csr.npz", X)

    print(f"Plays read: {len(txt_files)}")
    print(f"Scene-documents: {len(all_scene_docs)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"CSR shape: {X.shape}, nnz={X.nnz}")
    print("Saved: doc_ids.txt, vocab.txt, bow_scenes_csr.npz")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build scene-level bag-of-words (CSR) with normalization options"
    )

    parser.add_argument("input_dir", help="Folder containing Shakespeare .txt files")

    # normalization
    parser.add_argument("-lowercase", action="store_true")
    parser.add_argument("-stem", action="store_true")
    parser.add_argument("-lemmatize", action="store_true")
    parser.add_argument("-stopwords", action="store_true")
    parser.add_argument("-myopt", action="store_true", help="Strip accents (unidecode)")

    parser.add_argument("--min_df", type=int, default=1,
                        help="Minimum document frequency for vocab terms")

    args = parser.parse_args()

    build_from_folder(
        folder=args.input_dir,
        norm_kwargs=dict(
            lowercase=args.lowercase,
            stem=args.stem,
            lemma=args.lemmatize,
            stopwords=args.stopwords,
            accent=args.myopt
        ),
        min_df=args.min_df
    )