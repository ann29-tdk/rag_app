"""
rag/engine.py
=============
Generalised RAG Retrieval Engine.

Works on ANY document — research papers, books, history texts,
wildlife reports, legal documents, Soviet Union archives, anything.

Architecture
------------
This engine handles RETRIEVAL ONLY:
  1. Load any PDF / DOCX / TXT
  2. Split into overlapping chunks
  3. Build TF-IDF + LSA embeddings
  4. Hybrid search: FAISS semantic + exhaustive BM25 keyword scan
  5. Return the top-k most relevant chunks as raw text

Answer GENERATION is handled by the Anthropic API in the browser
(app_flask.py sends retrieved chunks to Claude, which writes the answer).
This separation means answers are ALWAYS accurate and coherent English,
regardless of document domain or question type.
"""

import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import faiss
from pypdf import PdfReader

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CHUNK_SIZE     = 500    # characters per chunk
CHUNK_OVERLAP  = 80     # overlap between consecutive chunks
TOP_K          = 8      # chunks returned per query
TFIDF_FEATURES = 8000   # vocabulary size for TF-IDF
LSA_COMPONENTS = 150    # LSA latent dimensions

CACHE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

# Greek / math characters common in academic PDFs — replace with text
_GREEK = {
    'θ':'theta', 'α':'alpha',  'β':'beta',    'γ':'gamma',   'δ':'delta',
    'ε':'epsilon','η':'eta',   'λ':'lambda',  'μ':'mu',      'σ':'sigma',
    'φ':'phi',   'ω':'omega',  'ℓ':'l',       '∆':'Delta',   '∈':'in',
    '×':'x',     '→':'->',    '≤':'<=',       '≥':'>=',
    '−':'-',     '–':'-',     '—':'-',
    'ﬁ':'fi',    'ﬀ':'ff',    'ﬃ':'ffi',     'ﬂ':'fl',
}

# Words that look like proper names but are NOT author names
_NAME_BLOCKLIST = {
    "Abstract","Generative","Adversarial","Networks","Learning","Conference",
    "International","Proceedings","Machine","Deep","Neural","Natural","Images",
    "Latent","Space","Convolutional","Optimizing","Copyright","Figure","Table",
    "Section","Appendix","Stochastic","Gradient","Descent","Research","Based",
    "Training","Image","Model","Using","Loss","Enhancement","Framework",
    "Contrast","Medical","Automated","Brightness","Preserving","Department",
    "Soviet","Union","Russian","American","British","European","Asian",
}

# Keywords that indicate an institutional affiliation in a document
_AFFIL_KEYWORDS = [
    "university","institute","research","laboratory","lab","department",
    "facebook","google","deepmind","openai","microsoft","mit","stanford",
    "harvard","cambridge","oxford","ai research","correspondence",
    "birla","bitmesra","proceedings","conference","copyright",
    "academy","college","school","faculty","center","centre",
]

# Question words that indicate an author/institution query
_META_TRIGGERS = {
    "author","authors","wrote","written","who","name","names",
    "affiliation","affiliated","institution","organisation","organization",
    "published","year",
}

# Common words that add no signal for retrieval scoring
_STOPWORDS = {
    "what","how","why","which","when","where","does","did","the","and","for",
    "are","was","this","that","with","from","have","its","you","paper","about",
    "who","name","tell","explain","describe","give","can","could","please",
    "list","also","is","a","an","of","in","on","at","by","to","or","but",
    "not","do","be","if","as","so","we","they","it","were","been","has",
    "had","their","our","your","he","she","his","her","them","these","those",
    "than","then","into","each","every","both","few","more","most","other",
    "such","no","nor","only","same","too","very","just","during","before",
    "after","above","between","through","used","use","using","uses",
}


# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────

def _clean(text: str) -> str:
    """
    Clean raw PDF/text content:
    - Translate Greek & math symbols to ASCII text
    - Rejoin words broken across lines by hyphens
    - Remove non-printable / non-ASCII characters
    - Collapse multiple spaces
    """
    for ch, rep in _GREEK.items():
        text = text.replace(ch, rep)
    # "pro-\njected" → "projected"
    text = re.sub(r'(\w+)-\n(\w)', lambda m: m.group(1) + m.group(2), text)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _normalise(w: str) -> str:
    """
    Light stemmer for retrieval matching.
    Handles British/American spelling (initialise/initialize)
    and strips common suffixes (running→runn, methods→method).
    """
    w = w.lower()
    if len(w) > 5:
        w = re.sub(r'ised$', 'ized', w)
        w = re.sub(r'ise$',  'ize',  w)
    if w.endswith('izing') and len(w) > 6: return w[:-3]
    if w.endswith('ized')  and len(w) > 5: return w[:-2]
    if w.endswith('ize')   and len(w) > 5: return w[:-1]
    if w.endswith('ing')   and len(w) > 5: return w[:-3]
    if w.endswith('tion')  and len(w) > 5: return w[:-3]
    if w.endswith('ed')    and len(w) > 4: return w[:-2]
    if w.endswith('s')     and len(w) > 4: return w[:-1]
    return w


def _qwords(question: str) -> set:
    """Normalised content words from a question, stopwords removed."""
    raw = set(re.findall(r'\b\w{2,}\b', question.lower()))
    return {_normalise(w) for w in raw} - _STOPWORDS - {_normalise(s) for s in _STOPWORDS}


def _is_meta_query(question: str) -> bool:
    """True if the question is asking about authors, institution, year, etc."""
    words = set(re.findall(r'\b\w{3,}\b', question.lower())) - _STOPWORDS
    return bool(words & _META_TRIGGERS)


# ─────────────────────────────────────────
# AUTHOR / AFFILIATION EXTRACTION
# (used for metadata queries — author name, institution, etc.)
# ─────────────────────────────────────────

def _extract_names_from_chunk(chunk: str) -> List[str]:
    """
    Extract proper-name pairs and triples from the first 600 chars of a chunk.
    Strips LaTeX superscript digits (Bojanowski1 → Bojanowski).
    Filters against _NAME_BLOCKLIST to avoid false positives.
    """
    c = re.sub(r'[,;˚*†‡]', ' ', chunk[:600])
    c = re.sub(r'([A-Z][a-z]+)[0-9]+', r'\1', c)   # strip superscripts
    c = re.sub(r' {2,}', ' ', c)
    seen, results = [], []

    # 3-word names first (more specific, avoids partial matches)
    for a, b, d in re.findall(
            r'\b([A-Z][a-zA-Z\-]+)\s+([A-Z][a-zA-Z\-]+)\s+([A-Z][a-zA-Z\-]+)\b', c):
        if any(w in _NAME_BLOCKLIST for w in [a, b, d]): continue
        if not all(w.isalpha() for w in [a, b, d]): continue
        name = f"{a} {b} {d}"
        if name not in seen: seen.append(name); results.append(name)

    # 2-word names
    for a, b in re.findall(r'\b([A-Z][a-zA-Z\-]+)\s+([A-Z][a-zA-Z\-]+)\b', c):
        if a in _NAME_BLOCKLIST or b in _NAME_BLOCKLIST: continue
        if not a.isalpha() or not b.isalpha(): continue
        name = f"{a} {b}"
        if name not in seen and not any(name in n for n in seen):
            seen.append(name); results.append(name)
    return results


def _extract_author_lines(all_doc_chunks: List[str]) -> str:
    """
    Scans the first 5 chunks (title-page area) for author names and emails.
    Then scans all chunks for affiliation keywords.
    Returns a formatted string for the /ask metadata route.
    """
    collected, seen = [], set()

    def add(item: str):
        k = item[:60]
        if k not in seen: seen.add(k); collected.append(item)

    # Title-page chunks: emails and names
    for chunk in all_doc_chunks[:5]:
        emails = re.findall(r'[\w.+-]+@[\w.-]+\.\w{2,}', chunk)
        if emails: add("Email: " + ", ".join(emails))
        names = _extract_names_from_chunk(chunk)
        if names: add("Authors: " + ", ".join(names[:8]))

    # All chunks: affiliation keywords
    for chunk in all_doc_chunks:
        cl = chunk.lower()
        for kw in _AFFIL_KEYWORDS:
            idx = cl.find(kw)
            if idx != -1:
                add("Affiliation: " + chunk[max(0, idx-30): idx+100].strip()[:150])
                break

    return " || ".join(collected[:6]) if collected else ""


# ─────────────────────────────────────────
# DOCUMENT LOADERS
# ─────────────────────────────────────────

def load_pdf(path: str) -> Tuple[str, Dict[int, str]]:
    """Load a PDF, clean each page, return (full_text_with_markers, page_dict)."""
    reader, pages, ft = PdfReader(path), {}, ""
    for i, page in enumerate(reader.pages):
        t = _clean(page.extract_text() or "")
        pages[i + 1] = t
        ft += f"\n[Page {i+1}]\n{t}\n"
    return ft, pages


def load_txt(path: str) -> Tuple[str, Dict[int, str]]:
    """Load a plain-text file, trying multiple encodings."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(path, encoding=enc) as f:
                text = _clean(f.read())
            return text, {1: text}
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode text file: {path}")


def load_docx(path: str) -> Tuple[str, Dict[int, str]]:
    """Load a Word document (.docx)."""
    try:
        import docx
    except ImportError:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    doc  = docx.Document(path)
    text = _clean("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
    return text, {1: text}


def load_document(path: str) -> Tuple[str, Dict[int, str]]:
    """Dispatch loader by file extension. Supports: .pdf, .txt, .docx"""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":  return load_pdf(path)
    if ext == ".txt":  return load_txt(path)
    if ext == ".docx": return load_docx(path)
    raise ValueError(f"Unsupported file type: '{ext}'. Use PDF, TXT, or DOCX.")


# ─────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────

def smart_chunk(ft: str, size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> Tuple[List[str], List[int]]:
    """
    Split document text into overlapping fixed-size chunks.
    Tracks which page number each chunk came from.

    Each chunk is a flat, space-joined string (no internal newlines),
    which makes BM25 matching and regex cleaning straightforward.

    Returns: (chunks, page_numbers)
    """
    lines, chunks, cpages = ft.split('\n'), [], []
    buf, cur = "", 1

    for line in lines:
        # Page marker inserted by load_pdf
        pm = re.match(r'\[Page (\d+)\]', line)
        if pm:
            cur = int(pm.group(1))
            continue
        line = line.strip()
        if not line:
            continue

        if len(buf) + len(line) + 1 <= size:
            buf = buf + " " + line if buf else line
        else:
            if buf:
                chunks.append(buf.strip())
                cpages.append(cur)
            # Overlap: carry last 15 words into next chunk
            words = buf.split()
            ot    = " ".join(words[-15:]) if len(words) > 15 else ""
            buf   = (ot + " " + line).strip() if ot else line

    if buf.strip():
        chunks.append(buf.strip())
        cpages.append(cur)

    return chunks, cpages


# ─────────────────────────────────────────
# EMBEDDINGS — TF-IDF + LSA
# ─────────────────────────────────────────

def build_embeddings(chunks: List[str]) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    """
    Build L2-normalised LSA embeddings from document chunks.
    Pipeline: raw text → TF-IDF (bigrams) → SVD → L2-normalise.
    """
    vec = TfidfVectorizer(
        max_features   = TFIDF_FEATURES,
        ngram_range    = (1, 2),
        sublinear_tf   = True,
        min_df         = 1,
        strip_accents  = "unicode",
    )
    X      = vec.fit_transform(chunks)
    n_comp = min(LSA_COMPONENTS, X.shape[1] - 1, X.shape[0] - 1)
    svd    = TruncatedSVD(n_components=n_comp, random_state=42)
    E      = svd.fit_transform(X).astype(np.float32)
    norms  = np.linalg.norm(E, axis=1, keepdims=True)
    return E / (norms + 1e-10), vec, svd


def embed_query(q: str, vec: TfidfVectorizer, svd: TruncatedSVD) -> np.ndarray:
    """Embed a query using the document's fitted TF-IDF + SVD pipeline."""
    qv = vec.transform([q])
    qe = svd.transform(qv).astype(np.float32)
    return qe / (np.linalg.norm(qe) + 1e-10)


def build_faiss_index(E: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index (cosine sim on normalised vectors)."""
    idx = faiss.IndexFlatIP(E.shape[1])
    idx.add(E)
    return idx


# ─────────────────────────────────────────
# HYBRID RETRIEVAL
# ─────────────────────────────────────────

def _bm25(chunk: str, q_norms: set) -> float:
    """
    BM25-style keyword score.
    Counts how many normalised query terms appear in the chunk,
    with a small density bonus to prefer focused chunks.
    """
    c_norms = {_normalise(w) for w in re.findall(r'\b\w{2,}\b', chunk.lower())}
    hits    = len(q_norms & c_norms)
    return hits + hits / (len(chunk.split()) + 1) * 8


def hybrid_search(
    query        : str,
    chunks       : List[str],
    chunk_pages  : List[int],
    chunk_sources: List[str],
    vectorizer   : TfidfVectorizer,
    svd          : TruncatedSVD,
    faiss_index  : faiss.IndexFlatIP,
    top_k        : int = TOP_K,
    source_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Two-pass hybrid retrieval — works on ANY document domain.

    Pass 1 — FAISS semantic search (TF-IDF + LSA cosine similarity).
              Finds chunks that are semantically related to the query.
              Good for paraphrase, synonyms, related concepts.

    Pass 2 — Exhaustive BM25 keyword scan on all remaining chunks.
              Catches specific numbers, names, terms that semantic search
              may miss (e.g. "0.584", "Bojanowski", "collectivisation").

    Both passes are domain-agnostic: the same code handles a paper about
    image processing, a history book about the Soviet Union, or anything else.

    Returns top_k chunks, merged, deduplicated, and ranked by combined score.
    """
    q_norms = _qwords(query)

    def make(idx, sem, kw):
        return {
            "text"  : chunks[idx],
            "source": chunk_sources[idx],
            "page"  : chunk_pages[idx],
            "sem"   : sem,
            "kw"    : kw,
        }

    # ── Pass 1: semantic (FAISS) ──────────────────────────────────────
    qe      = embed_query(query, vectorizer, svd)
    fetch_k = min(top_k * 10, len(chunks))
    D, I    = faiss_index.search(qe, fetch_k)
    pool    = {}
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks): continue
        if source_filter and chunk_sources[idx] != source_filter: continue
        key = chunks[idx][:80]
        if key not in pool:
            pool[key] = make(idx, float(score), _bm25(chunks[idx], q_norms))

    # ── Pass 2: exhaustive BM25 on all remaining chunks ───────────────
    for idx, chunk in enumerate(chunks):
        if source_filter and chunk_sources[idx] != source_filter: continue
        key = chunk[:80]
        if key in pool: continue
        ks = _bm25(chunk, q_norms)
        if ks >= 1.5:
            pool[key] = make(idx, 0.0, ks)

    # ── Combine scores and rank ───────────────────────────────────────
    all_e   = list(pool.values())
    max_sem = max((e["sem"] for e in all_e), default=1) or 1
    max_kw  = max((e["kw"]  for e in all_e), default=1) or 1
    for e in all_e:
        # 55% semantic weight, 45% keyword weight
        e["combined"] = 0.55 * e["sem"] / max_sem + 0.45 * e["kw"] / max_kw
    all_e.sort(key=lambda x: x["combined"], reverse=True)

    seen, final = set(), []
    for e in all_e:
        key = e["text"][:80]
        if key in seen: continue
        seen.add(key)
        final.append({
            "text"     : e["text"],
            "source"   : e["source"],
            "page"     : e["page"],
            "score"    : round(e["combined"], 3),
            "score_pct": int(e["combined"] * 100),
        })
        if len(final) >= top_k: break
    return final


# ─────────────────────────────────────────
# RAG ENGINE
# ─────────────────────────────────────────

class RAGEngine:
    """
    Generalised RAG Retrieval Engine.

    Handles: loading, chunking, indexing, and retrieving relevant text
    from any document. Answer generation is handled externally by the
    Anthropic API (in app_flask.py) to ensure accurate, fluent English.

    Supports: PDF, DOCX, TXT
    Supports: multi-document indexing with per-document scope filtering

    Usage
    -----
        engine = RAGEngine()
        engine.index_file("soviet_history.pdf")
        chunks = engine.search("What caused the collapse of the Soviet Union?")
        # -> returns 8 most relevant text chunks
        # -> pass these to Claude API for the final answer

    Multi-document with scope
    -------------------------
        engine.index_file("doc1.pdf")
        engine.index_file("doc2.pdf")
        # Search only doc1:
        chunks = engine.search("Who is the author?", source_filter="doc1.pdf")
        # Search all docs:
        chunks = engine.search("Compare both approaches", source_filter=None)
    """

    def __init__(self):
        self.chunks        : List[str]                   = []
        self.chunk_pages   : List[int]                   = []
        self.chunk_sources : List[str]                   = []
        self.vectorizer    : Optional[TfidfVectorizer]   = None
        self.svd           : Optional[TruncatedSVD]      = None
        self.index         : Optional[faiss.IndexFlatIP] = None
        self.embeddings    : Optional[np.ndarray]        = None
        self.documents     : List[Dict]                  = []
        self.chat_history  : List[Dict]                  = []

    # ── INDEXING ────────────────────────────────────────────────────
    def index_file(self, filepath: str) -> Dict:
        """
        Load a document file, split into chunks, build embeddings, index.
        Safe to call multiple times — re-indexing replaces old chunks.

        Returns {"success": True, "filename": ..., "chunks": N, "pages": N}
        """
        path, filename = Path(filepath), Path(filepath).name
        try:
            full_text, page_map = load_document(filepath)
        except Exception as e:
            return {"success": False, "error": str(e), "filename": filename}

        new_chunks, new_pages = smart_chunk(full_text)
        if not new_chunks:
            return {"success": False,
                    "error": "No text could be extracted from this file.",
                    "filename": filename}

        # Remove any stale chunks from a previous index of this same file
        keep = [i for i, s in enumerate(self.chunk_sources) if s != filename]
        self.chunks        = [self.chunks[i]        for i in keep]
        self.chunk_pages   = [self.chunk_pages[i]   for i in keep]
        self.chunk_sources = [self.chunk_sources[i] for i in keep]

        self.chunks        += new_chunks
        self.chunk_pages   += new_pages
        self.chunk_sources += [filename] * len(new_chunks)
        self._rebuild_index()

        doc_info = {
            "filename" : filename,
            "filepath" : str(filepath),
            "pages"    : len(page_map),
            "chunks"   : len(new_chunks),
            "size_mb"  : round(path.stat().st_size / (1024 * 1024), 2),
        }
        self.documents = [d for d in self.documents if d["filename"] != filename]
        self.documents.append(doc_info)
        return {"success": True, "filename": filename,
                "chunks": len(new_chunks), "pages": len(page_map)}

    def _rebuild_index(self):
        """Rebuild TF-IDF + LSA + FAISS from all current chunks."""
        if not self.chunks:
            self.vectorizer = self.svd = self.index = self.embeddings = None
            return
        self.embeddings, self.vectorizer, self.svd = build_embeddings(self.chunks)
        self.index = build_faiss_index(self.embeddings)

    def remove_document(self, filename: str) -> bool:
        """Remove all chunks for one document and rebuild the index."""
        keep = [i for i, s in enumerate(self.chunk_sources) if s != filename]
        if len(keep) == len(self.chunk_sources):
            return False  # document not found
        self.chunks        = [self.chunks[i]        for i in keep]
        self.chunk_pages   = [self.chunk_pages[i]   for i in keep]
        self.chunk_sources = [self.chunk_sources[i] for i in keep]
        self.documents     = [d for d in self.documents if d["filename"] != filename]
        self._rebuild_index()
        return True

    def reset(self):
        """Wipe all documents, chunks, and index. Start fresh."""
        self.__init__()

    # ── SEARCH ──────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = TOP_K,
               source_filter: Optional[str] = None) -> List[Dict]:
        """
        Find the most relevant document chunks for a query.

        Parameters
        ----------
        query         : Natural language question or search string.
        top_k         : Maximum chunks to return (default 8).
        source_filter : Restrict to one document filename, or None for all.

        Returns
        -------
        List of dicts: [{text, source, page, score, score_pct}, ...]
        The 'text' field contains the raw document chunk.
        These chunks are sent to the Anthropic API to generate the answer.
        """
        if self.index is None or not self.chunks:
            return []
        return hybrid_search(
            query, self.chunks, self.chunk_pages, self.chunk_sources,
            self.vectorizer, self.svd, self.index,
            top_k=top_k, source_filter=source_filter,
        )

    # ── UTILITIES ───────────────────────────────────────────────────
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []

    def get_stats(self) -> Dict:
        """Return current index statistics."""
        return {
            "total_chunks"  : len(self.chunks),
            "documents"     : [d["filename"] for d in self.documents],
            "doc_details"   : self.documents,
            "history_turns" : len(self.chat_history) // 2,
        }
