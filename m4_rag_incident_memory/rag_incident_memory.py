# -*- coding: utf-8 -*-
"""
VIGIL — M4 RAG Incident Memory
=================================
A searchable memory of past industrial accidents and OISD safety regulations.
When M3 fires a RiskEvent, this module finds the most similar past incidents
and the relevant regulatory clauses, and returns them as grounding context.

Think of it as a safety officer who has read every DGFASLI accident report
ever published. Show them a current situation and they immediately say:
"This looks like the Vizag incident from 2025 — here's what was missed."

WHAT THIS MODULE DOES
-----------------------
1. Loads the incident + regulation corpus (incident_corpus.json)
2. Embeds every document into a vector using sentence embeddings
3. Builds a FAISS index for fast similarity search (falls back to numpy
   brute-force cosine similarity if FAISS is not installed)
4. Given a RiskEvent (or raw query text), embeds the query and retrieves
   the top-k most similar incidents/regulations
5. Also does exact pattern-signature matching (rules_fired -> pattern_signature)
   as a second retrieval path that doesn't depend on embedding quality
6. Returns ranked results with similarity scores + the applicable OISD clause

OUTPUT — RetrievalResult
--------------------------
{
    "query_summary": "Zone C3: Gas + Hot Work Permit, Gas + Non-Isolated Maintenance...",
    "retrieved_at": "2026-01-15T08:42:05Z",
    "top_incidents": [
        {
            "doc_id": "INC-001",
            "title": "Visakhapatnam Steel Plant Coke Oven Explosion (January 2025)",
            "similarity_score": 0.94,
            "match_type": "pattern_signature",   # or "embedding"
            "summary": "...",
            "fatalities": 8,
            "root_causes": [...],
            "oisd_clauses_violated": [...]
        },
        ... (top 3)
    ],
    "applicable_regulations": [
        {
            "doc_id": "REG-001",
            "title": "OISD-116 Clause 8.4 — Hot Work Permit Requirements...",
            "clause": "8.4",
            "summary": "...",
            "relevance": "Directly applicable — hot work permit active with gas above threshold"
        }
    ],
    "headline_match": "This matches the Visakhapatnam Steel Plant pattern (Jan 2025, 8 fatalities) — gas warning + hot work permit + non-isolated maintenance, the exact combination that preceded that explosion."
}

ALGORITHMS & LOGIC USED
-------------------------
1. DUAL RETRIEVAL STRATEGY
   a) PATTERN SIGNATURE MATCHING (deterministic, high precision)
      Every incident has a pattern_signature like "gas_warning+hot_work_permit+non_isolated_maintenance"
      M3's fired rule_ids are mapped to the same vocabulary (CR-001 -> gas_warning+hot_work_permit, etc.)
      Jaccard similarity between current condition-set and each incident's pattern_signature set.
      This is what catches "this IS the Vizag pattern" with high confidence — not just semantically
      similar text, but the actual same combination of structural conditions.

   b) EMBEDDING SIMILARITY (semantic, broad recall)
      Each document's title+summary+keywords is embedded into a vector.
      Cosine similarity between query embedding and corpus embeddings.
      Catches incidents that are conceptually related even without identical
      structural patterns (e.g. different gas, same "stale sensor" theme).

   Final ranking blends both: pattern_score * 0.6 + embedding_score * 0.4
   Pattern match is weighted higher because it's the more rigorous signal.

2. EMBEDDING METHOD
   PRIMARY: sentence-transformers (all-MiniLM-L6-v2) if installed — real semantic embeddings.
   FALLBACK: TF-IDF style bag-of-words vectorisation using numpy — no internet/model
   download required. This guarantees the module runs in any environment, including
   offline hackathon demo machines, without silently failing.

3. VECTOR INDEX
   PRIMARY: FAISS IndexFlatIP (inner product on normalised vectors = cosine similarity).
   FALLBACK: numpy brute-force matrix multiply — for a corpus this size (12 docs),
   brute force is actually just as fast as FAISS and avoids the dependency entirely.
   Both paths are exposed through the same VectorIndex interface so M3/M4 integration
   code never needs to know which backend is active.

4. PATTERN SIGNATURE VOCABULARY MAPPING
   M3's rule_ids (CR-001, CR-002, etc.) are mapped to canonical condition tokens.
   This shared vocabulary is what lets M3's structured output talk directly to
   M4's incident corpus without re-parsing natural language.

TECHNOLOGIES
------------
- faiss-cpu (optional): production-grade vector similarity search
- sentence-transformers (optional): real semantic embeddings
- numpy: fallback vectorisation + brute-force search (always available)
- json: corpus loading
- dataclasses: typed retrieval results
"""

import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Windows PowerShell defaults to a legacy codepage (cp1252/cp437) that cannot
# encode characters like em-dashes (—) or arrows (→) used throughout this
# module's print statements. Without this, the script crashes silently mid-run
# with UnicodeEncodeError the instant it hits the first such character —
# exactly the symptom of output stopping right after corpus/index build logs.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # Python <3.7 fallback — not expected in this environment

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation
# ---------------------------------------------------------------------------
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "rag_incident_memory.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("VIGIL.M4.RAGIncidentMemory")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K_INCIDENTS = 3
TOP_K_REGULATIONS = 3
PATTERN_WEIGHT = 0.6
EMBEDDING_WEIGHT = 0.4
RELEVANCE_FLOOR = 0.25   # Incidents scoring below this are not shown — a 0.147
                          # match diluted alongside a 0.566 match makes the
                          # strongest match look weaker than it is

# Maps M3 rule_ids to canonical condition tokens used in pattern_signature
RULE_ID_TO_CONDITION = {
    "CR-001": "gas_warning+hot_work_permit",
    "CR-002": "gas_warning+non_isolated_maintenance",
    "CR-003": "hot_work_permit+shift_changeover",
    "CR-004": "confined_space_permit+oxygen_depletion",
    "CR-005": "gas_warning+uncertified_workers",
    "CR-006": "multi_zone_permit",
    "CR-007": "rapid_gas_escalation",
    "CR-008": "stale_sensor+active_maintenance",
    "CR-009": "hot_work_permit+no_fire_watch",
    "CR-010": "gas_warning+hot_work_permit+non_isolated_maintenance",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class IncidentMatch:
    doc_id: str
    title: str
    similarity_score: float
    match_percent: int            # similarity_score as 0-100 integer for display
    match_label: str              # "Strong" / "Moderate" / "Weak" qualitative tag
    match_type: str             # "pattern_signature" | "embedding" | "blended"
    summary: str
    facility: str
    fatalities: int
    root_causes: list
    oisd_clauses_violated: list


@dataclass
class RegulationMatch:
    doc_id: str
    title: str
    clause: str
    source_document: str
    summary: str
    relevance: str
    similarity_score: float


@dataclass
class RetrievalResult:
    query_summary: str
    retrieved_at: str
    top_incidents: list
    applicable_regulations: list
    headline_match: str
    embedding_backend: str       # "sentence-transformers" | "numpy-bow"
    index_backend: str           # "faiss" | "numpy-bruteforce"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _similarity_label(score: float) -> str:
    """Convert a raw 0-1 similarity score into a qualitative tag for display."""
    if score >= 0.55:
        return "Strong"
    if score >= 0.35:
        return "Moderate"
    return "Weak"


# ---------------------------------------------------------------------------
# Embedding backend — sentence-transformers primary, numpy BoW fallback
# ---------------------------------------------------------------------------
class EmbeddingBackend:
    """
    Produces vector embeddings for text.
    Falls back to a simple TF-IDF-style bag-of-words vectoriser if
    sentence-transformers is not installed — guarantees the module
    works in any environment without internet access for model download.
    """

    def __init__(self):
        self._model = None
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self.backend_name = "numpy-bow"

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self.backend_name = "sentence-transformers"
                logger.info("Embedding backend: sentence-transformers (all-MiniLM-L6-v2)")
            except Exception as e:
                logger.warning(f"sentence-transformers failed to load model: {e}. Using BoW fallback.")
        else:
            logger.info("sentence-transformers not installed. Using numpy bag-of-words fallback.")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit_vocabulary(self, documents: list[str]) -> None:
        """
        Build vocabulary + IDF weights from the corpus.
        Only used by the BoW fallback path.
        """
        if self._model is not None:
            return  # sentence-transformers doesn't need a fitted vocab

        doc_freq: Counter = Counter()
        tokenized_docs = [self._tokenize(d) for d in documents]

        for tokens in tokenized_docs:
            for term in set(tokens):
                doc_freq[term] += 1

        vocab_terms = sorted(doc_freq.keys())
        self._vocab = {term: i for i, term in enumerate(vocab_terms)}

        n_docs = len(documents)
        idf = np.zeros(len(vocab_terms))
        for term, idx in self._vocab.items():
            idf[idx] = np.log((n_docs + 1) / (doc_freq[term] + 1)) + 1
        self._idf = idf

        logger.info(f"BoW vocabulary fitted: {len(self._vocab)} terms across {n_docs} documents")

    def embed(self, text: str) -> np.ndarray:
        """Return a normalised embedding vector for the given text."""
        if self._model is not None:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        return self._embed_bow(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return a matrix of normalised embeddings, one row per text."""
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._embed_bow(t) for t in texts])

    def _embed_bow(self, text: str) -> np.ndarray:
        """TF-IDF bag-of-words vector, L2-normalised."""
        if not self._vocab:
            raise RuntimeError("Vocabulary not fitted. Call fit_vocabulary() first.")
        vec = np.zeros(len(self._vocab), dtype=np.float32)
        tokens = self._tokenize(text)
        term_counts = Counter(tokens)
        for term, count in term_counts.items():
            idx = self._vocab.get(term)
            if idx is not None:
                tf = 1 + np.log(count)
                vec[idx] = tf * self._idf[idx]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


# ---------------------------------------------------------------------------
# Vector index — FAISS primary, numpy brute-force fallback
# ---------------------------------------------------------------------------
class VectorIndex:
    """
    Unified interface over FAISS IndexFlatIP or numpy brute-force cosine search.
    Both backends operate on L2-normalised vectors, so inner product == cosine similarity.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None
        self.backend_name = "numpy-bruteforce"

        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
            self.backend_name = "faiss"
            logger.info(f"Vector index backend: FAISS IndexFlatIP (dim={dim})")
        else:
            self._index = None
            logger.info(f"FAISS not installed. Using numpy brute-force search (dim={dim})")

    def add(self, vectors: np.ndarray) -> None:
        """Add a matrix of vectors (n_docs x dim) to the index."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if FAISS_AVAILABLE:
            self._index.add(vectors)
        self._vectors = vectors  # keep for fallback / debugging either way

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (scores, indices) for the top-k most similar vectors.
        scores: cosine similarity (higher = more similar)
        indices: row indices into the original document list
        """
        query_vec = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)

        if FAISS_AVAILABLE:
            scores, indices = self._index.search(query_vec, k)
            return scores[0], indices[0]

        # Numpy brute-force: cosine similarity via dot product (vectors pre-normalised)
        sims = (self._vectors @ query_vec.T).flatten()
        top_k_idx = np.argsort(-sims)[:k]
        top_k_scores = sims[top_k_idx]
        return top_k_scores, top_k_idx


# ---------------------------------------------------------------------------
# Core RAG module
# ---------------------------------------------------------------------------
class RAGIncidentMemory:
    """
    Loads the incident/regulation corpus, builds the index, and serves
    retrieval queries from RiskEvents.

    Usage:
        rag = RAGIncidentMemory()
        result = rag.retrieve_from_risk_event(risk_event_dict)
        # or
        result = rag.retrieve_from_text("gas leak near hot work during shift change")
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Path(__file__).parent / "data")
        self._corpus: list[dict] = []
        self._embedder = EmbeddingBackend()
        self._index: Optional[VectorIndex] = None

        self._load_corpus()
        self._build_index()

        logger.info(
            f"RAGIncidentMemory ready | corpus={len(self._corpus)} docs | "
            f"embedding={self._embedder.backend_name} | "
            f"index={self._index.backend_name}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_corpus(self) -> None:
        path = self.data_dir / "incident_corpus.json"
        if not path.exists():
            raise FileNotFoundError(f"Incident corpus not found: {path}")
        with open(path, encoding="utf-8") as f:
            self._corpus = json.load(f)
        logger.info(f"Loaded {len(self._corpus)} documents from {path.name}")

    def _doc_text(self, doc: dict) -> str:
        """Build the text representation used for embedding."""
        parts = [doc.get("title", ""), doc.get("summary", "")]
        parts.extend(doc.get("keywords", []))
        if doc.get("root_causes"):
            parts.extend(doc["root_causes"])
        return " ".join(parts)

    def _build_index(self) -> None:
        texts = [self._doc_text(d) for d in self._corpus]

        # Fit BoW vocabulary on the corpus if using fallback embedder
        self._embedder.fit_vocabulary(texts)

        embeddings = self._embedder.embed_batch(texts)
        self._index = VectorIndex(dim=embeddings.shape[1])
        self._index.add(embeddings)

        logger.info(f"Index built: {embeddings.shape[0]} vectors, dim={embeddings.shape[1]}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve_from_risk_event(self, risk_event) -> RetrievalResult:
        """
        Main entry point. Takes a RiskEvent (object or dict from M3) and
        retrieves the most relevant incidents and regulations.
        """
        if hasattr(risk_event, "to_dict"):
            event = risk_event.to_dict()
        else:
            event = risk_event

        zone = event.get("zone", "UNKNOWN")
        rules_fired = event.get("rules_fired", [])
        rule_ids = [r.get("rule_id") for r in rules_fired]

        # Build condition token set from fired rules
        condition_tokens = self._rule_ids_to_condition_set(rule_ids)

        # Build query text for embedding search
        rule_names = [r.get("name", "") for r in rules_fired]
        query_text = f"Zone {zone}: " + ", ".join(rule_names)

        return self._retrieve(query_text, condition_tokens)

    def retrieve_from_text(self, query_text: str) -> RetrievalResult:
        """Retrieve using a free-text query (no structured rule_ids available)."""
        return self._retrieve(query_text, condition_tokens=set())

    # ------------------------------------------------------------------
    # Internal: retrieval logic
    # ------------------------------------------------------------------
    def _rule_ids_to_condition_set(self, rule_ids: list[str]) -> set[str]:
        """Convert M3 rule_ids into the canonical condition token vocabulary."""
        tokens = set()
        for rid in rule_ids:
            mapped = RULE_ID_TO_CONDITION.get(rid)
            if mapped:
                tokens.update(mapped.split("+"))
        return tokens

    def _pattern_similarity(self, condition_tokens: set, doc: dict) -> float:
        """
        Jaccard similarity between current condition set and a document's
        pattern_signature. Returns 0 for documents without a pattern_signature
        (i.e. regulation docs use a different relevance mechanism).
        """
        sig = doc.get("pattern_signature")
        if not sig or not condition_tokens:
            return 0.0
        doc_tokens = set(sig.split("+"))
        if not doc_tokens:
            return 0.0
        intersection = condition_tokens & doc_tokens
        union = condition_tokens | doc_tokens
        return len(intersection) / len(union) if union else 0.0

    def _retrieve(self, query_text: str, condition_tokens: set) -> RetrievalResult:
        query_vec = self._embedder.embed(query_text)

        # Get embedding similarity for all docs (search full corpus, re-rank after)
        k = len(self._corpus)
        emb_scores, emb_indices = self._index.search(query_vec, k)

        # Build blended scores per document
        blended = []
        for score, idx in zip(emb_scores, emb_indices):
            doc = self._corpus[int(idx)]
            pattern_score = self._pattern_similarity(condition_tokens, doc)
            embedding_score = max(0.0, float(score))  # cosine can be slightly negative
            final_score = (PATTERN_WEIGHT * pattern_score) + (EMBEDDING_WEIGHT * embedding_score)
            match_type = (
                "pattern_signature" if pattern_score > 0.3
                else "embedding"
            )
            blended.append((final_score, match_type, doc, pattern_score, embedding_score))

        blended.sort(key=lambda x: -x[0])

        # Split into incidents and regulations
        incidents = [b for b in blended if b[2].get("type") == "incident"]
        regulations = [b for b in blended if b[2].get("type") == "regulation"]

        # Apply relevance floor — weak matches dilute the strongest match and
        # confuse a judge skimming 3 cards. Drop anything below the floor,
        # unless it would leave zero incidents (always show at least one).
        incidents_above_floor = [b for b in incidents if b[0] >= RELEVANCE_FLOOR]
        incidents_to_use = incidents_above_floor if incidents_above_floor else incidents[:1]

        top_incidents = [
            IncidentMatch(
                doc_id=doc["doc_id"],
                title=doc["title"],
                similarity_score=round(final_score, 3),
                match_percent=round(final_score * 100),
                match_label=_similarity_label(final_score),
                match_type=match_type,
                summary=doc["summary"],
                facility=doc.get("facility", ""),
                fatalities=doc.get("fatalities", 0),
                root_causes=doc.get("root_causes", []),
                oisd_clauses_violated=doc.get("oisd_clauses_violated", []),
            )
            for final_score, match_type, doc, p, e in incidents_to_use[:TOP_K_INCIDENTS]
        ]

        top_regulations = [
            RegulationMatch(
                doc_id=doc["doc_id"],
                title=doc["title"],
                clause=doc.get("clause", ""),
                source_document=doc.get("source_document", ""),
                summary=doc["summary"],
                relevance=self._explain_relevance(condition_tokens, doc),
                similarity_score=round(final_score, 3),
            )
            for final_score, match_type, doc, p, e in regulations[:TOP_K_REGULATIONS]
        ]

        headline = self._build_headline(top_incidents, condition_tokens)

        result = RetrievalResult(
            query_summary=query_text,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
            top_incidents=[asdict(i) for i in top_incidents],
            applicable_regulations=[asdict(r) for r in top_regulations],
            headline_match=headline,
            embedding_backend=self._embedder.backend_name,
            index_backend=self._index.backend_name,
        )

        logger.info(
            f"Retrieval complete | query='{query_text[:60]}...' | "
            f"top_incident={top_incidents[0].doc_id if top_incidents else 'none'} "
            f"(score={top_incidents[0].similarity_score if top_incidents else 0})"
        )

        return result

    def _explain_relevance(self, condition_tokens: set, reg_doc: dict) -> str:
        """Build a human-readable relevance explanation for a regulation match."""
        applies_when = set(reg_doc.get("applies_when", []))
        overlap = condition_tokens & applies_when
        if overlap:
            return f"Directly applicable — matches current condition(s): {', '.join(sorted(overlap))}"
        return "Related by semantic similarity to current situation"

    def _build_headline(self, top_incidents: list[IncidentMatch], condition_tokens: set) -> str:
        """Build the single-sentence headline match for the supervisor-facing alert."""
        if not top_incidents:
            return "No closely matching historical incident found in corpus."

        top = top_incidents[0]
        if top.similarity_score < 0.15:
            return (
                f"No strong historical match. Closest reference: {top.title} "
                f"(similarity {top.similarity_score})."
            )

        date_str = ""
        fatality_str = f", {top.fatalities} fatalities" if top.fatalities else ", no fatalities"

        return (
            f"This matches the {top.facility} pattern ({top.title.split('(')[-1].rstrip(')')}"
            f"{fatality_str}) — {top.summary[:120]}..."
        )

    def reload_corpus(self) -> None:
        """Hot-reload corpus from disk and rebuild index."""
        self._load_corpus()
        self._build_index()
        logger.info("Corpus reloaded and index rebuilt.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nVIGIL M4  -  RAG Incident Memory")
    print("Building index from incident corpus...\n")

    rag = RAGIncidentMemory()

    print(f"Embedding backend : {rag._embedder.backend_name}")
    print(f"Index backend     : {rag._index.backend_name}")
    print(f"Corpus size       : {len(rag._corpus)} documents\n")

    # Simulate M3's output for the Vizag-pattern scenario
    mock_risk_event = {
        "zone": "C3",
        "risk_score": 100,
        "severity": "CRITICAL",
        "rules_fired": [
            {"rule_id": "CR-001", "name": "Gas + Hot Work Permit"},
            {"rule_id": "CR-002", "name": "Gas + Non-Isolated Maintenance"},
            {"rule_id": "CR-003", "name": "Hot Work During Shift Changeover Window"},
            {"rule_id": "CR-010", "name": "Triple Compound: Gas + Hot Work + Non-Isolated Maintenance"},
        ],
    }

    print("="*70)
    print("  QUERY: Vizag-pattern RiskEvent from M3 (Zone C3, score=100)")
    print("="*70)

    result = rag.retrieve_from_risk_event(mock_risk_event)

    print(f"\nHEADLINE MATCH:\n  {result.headline_match}\n")

    print(f"TOP {len(result.top_incidents)} SIMILAR INCIDENTS:")
    for i, inc in enumerate(result.top_incidents, 1):
        print(f"\n  {i}. [{inc['doc_id']}] {inc['title']}")
        print(f"     Similarity: {inc['similarity_score']} ({inc['match_type']})")
        print(f"     Facility: {inc['facility']} | Fatalities: {inc['fatalities']}")
        print(f"     Root causes: {'; '.join(inc['root_causes'][:2])}")

    print(f"\nAPPLICABLE REGULATIONS:")
    for i, reg in enumerate(result.applicable_regulations, 1):
        print(f"\n  {i}. [{reg['doc_id']}] {reg['title']}")
        print(f"     Relevance: {reg['relevance']}")
        print(f"     {reg['summary'][:140]}...")

    print("\n" + "="*70)
    print("  Second test: free-text query (no structured rule_ids)")
    print("="*70)
    result2 = rag.retrieve_from_text("workers without certification near gas hazard")
    print(f"\nHEADLINE: {result2.headline_match}")
    print(f"Top match: {result2.top_incidents[0]['title'] if result2.top_incidents else 'none'}")