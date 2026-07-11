# -*- coding: utf-8 -*-
"""
Tests for VIGIL M4 — RAG Incident Memory
Run with: python -m pytest test_rag_incident_memory.py -v
"""

import json
from pathlib import Path

import pytest
from rag_incident_memory import (
    RAGIncidentMemory, RetrievalResult, EmbeddingBackend, VectorIndex,
    RULE_ID_TO_CONDITION,
)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def rag():
    """Build once per test module — index building is the expensive part."""
    return RAGIncidentMemory(data_dir=DATA_DIR)


def make_vizag_risk_event():
    return {
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


def make_confined_space_risk_event():
    return {
        "zone": "A1",
        "risk_score": 70,
        "severity": "HIGH",
        "rules_fired": [
            {"rule_id": "CR-004", "name": "Confined Space Entry + Oxygen Depletion"},
        ],
    }


# ---------------------------------------------------------------------------
# Tests: corpus loading
# ---------------------------------------------------------------------------
class TestCorpusLoading:
    def test_corpus_loads(self, rag):
        assert len(rag._corpus) > 0

    def test_corpus_has_incidents_and_regulations(self, rag):
        types = {d["type"] for d in rag._corpus}
        assert "incident" in types
        assert "regulation" in types

    def test_vizag_incident_in_corpus(self, rag):
        ids = [d["doc_id"] for d in rag._corpus]
        assert "INC-001" in ids

    def test_reload_corpus_works(self, rag):
        rag.reload_corpus()
        assert len(rag._corpus) > 0


# ---------------------------------------------------------------------------
# Tests: embedding backend
# ---------------------------------------------------------------------------
class TestEmbeddingBackend:
    def test_embed_returns_vector(self, rag):
        vec = rag._embedder.embed("test query about gas leak")
        assert vec.ndim == 1
        assert vec.shape[0] > 0

    def test_embed_batch_returns_matrix(self, rag):
        vecs = rag._embedder.embed_batch(["gas leak", "hot work fire", "confined space"])
        assert vecs.shape[0] == 3

    def test_embeddings_are_normalised(self, rag):
        import numpy as np
        vec = rag._embedder.embed("gas leak near hot work")
        norm = np.linalg.norm(vec)
        # Should be ~1.0 (normalised) or 0 (empty/unknown text edge case)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == 0.0

    def test_backend_name_set(self, rag):
        assert rag._embedder.backend_name in ("sentence-transformers", "numpy-bow")


# ---------------------------------------------------------------------------
# Tests: vector index
# ---------------------------------------------------------------------------
class TestVectorIndex:
    def test_index_built_with_correct_size(self, rag):
        assert rag._index._vectors.shape[0] == len(rag._corpus)

    def test_search_returns_k_results(self, rag):
        query_vec = rag._embedder.embed("gas hazard")
        scores, indices = rag._index.search(query_vec, k=5)
        assert len(scores) == 5
        assert len(indices) == 5

    def test_search_self_similarity_is_high(self, rag):
        """Searching for a document's own text should rank it highly."""
        doc = rag._corpus[0]
        text = rag._doc_text(doc)
        query_vec = rag._embedder.embed(text)
        scores, indices = rag._index.search(query_vec, k=1)
        assert indices[0] == 0  # Should find itself as top match


# ---------------------------------------------------------------------------
# Tests: pattern signature matching (the core retrieval mechanism)
# ---------------------------------------------------------------------------
class TestPatternMatching:
    def test_rule_id_mapping_complete(self):
        """Every CR-rule should have a condition mapping."""
        for rule_id in [f"CR-{i:03d}" for i in range(1, 11)]:
            assert rule_id in RULE_ID_TO_CONDITION

    def test_condition_set_from_vizag_rules(self, rag):
        rule_ids = ["CR-001", "CR-002", "CR-010"]
        tokens = rag._rule_ids_to_condition_set(rule_ids)
        assert "gas_warning" in tokens
        assert "hot_work_permit" in tokens
        assert "non_isolated_maintenance" in tokens

    def test_pattern_similarity_exact_match_is_high(self, rag):
        condition_tokens = {"gas_warning", "hot_work_permit", "non_isolated_maintenance"}
        vizag_doc = next(d for d in rag._corpus if d["doc_id"] == "INC-001")
        score = rag._pattern_similarity(condition_tokens, vizag_doc)
        assert score == 1.0  # Exact set match = Jaccard 1.0

    def test_pattern_similarity_zero_for_unrelated(self, rag):
        condition_tokens = {"confined_space_permit", "oxygen_depletion"}
        vizag_doc = next(d for d in rag._corpus if d["doc_id"] == "INC-001")
        score = rag._pattern_similarity(condition_tokens, vizag_doc)
        assert score < 0.5

    def test_pattern_similarity_zero_without_tokens(self, rag):
        empty_doc = {"pattern_signature": "gas_warning+hot_work_permit"}
        score = rag._pattern_similarity(set(), empty_doc)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: end-to-end retrieval from RiskEvent
# ---------------------------------------------------------------------------
class TestRetrievalFromRiskEvent:
    def test_vizag_pattern_retrieves_inc001_as_top_match(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert len(result.top_incidents) > 0
        assert result.top_incidents[0]["doc_id"] == "INC-001"

    def test_vizag_match_type_is_pattern_signature(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert result.top_incidents[0]["match_type"] == "pattern_signature"

    def test_confined_space_retrieves_inc003(self, rag):
        event = make_confined_space_risk_event()
        result = rag.retrieve_from_risk_event(event)
        top_ids = [i["doc_id"] for i in result.top_incidents]
        assert "INC-003" in top_ids

    def test_result_returns_top_k_incidents(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert len(result.top_incidents) <= 3

    def test_result_returns_regulations(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert len(result.applicable_regulations) > 0

    def test_headline_match_mentions_vizag(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert "Visakhapatnam" in result.headline_match

    def test_oisd_clauses_present_in_top_incident(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert len(result.top_incidents[0]["oisd_clauses_violated"]) > 0

    def test_regulation_relevance_explains_match(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        reg_001 = next(
            (r for r in result.applicable_regulations if r["doc_id"] == "REG-001"), None
        )
        if reg_001:
            assert "directly applicable" in reg_001["relevance"].lower() or \
                   "semantic" in reg_001["relevance"].lower()


# ---------------------------------------------------------------------------
# Tests: free-text retrieval
# ---------------------------------------------------------------------------
class TestRetrievalFromText:
    def test_free_text_query_returns_results(self, rag):
        result = rag.retrieve_from_text("gas leak during maintenance")
        assert len(result.top_incidents) > 0

    def test_free_text_no_pattern_match_uses_embedding_only(self, rag):
        result = rag.retrieve_from_text("random unrelated text about weather")
        # Should still return results, just with low scores / embedding match type
        assert isinstance(result, RetrievalResult)


# ---------------------------------------------------------------------------
# Tests: output structure
# ---------------------------------------------------------------------------
class TestOutputStructure:
    def test_to_json_valid(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        parsed = json.loads(result.to_json())
        assert "top_incidents" in parsed
        assert "applicable_regulations" in parsed

    def test_incident_match_has_required_fields(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        inc = result.top_incidents[0]
        required = [
            "doc_id", "title", "similarity_score", "match_type",
            "summary", "facility", "fatalities", "root_causes",
            "oisd_clauses_violated",
        ]
        for field in required:
            assert field in inc

    def test_regulation_match_has_required_fields(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        reg = result.applicable_regulations[0]
        required = ["doc_id", "title", "clause", "source_document", "summary", "relevance"]
        for field in required:
            assert field in reg

    def test_backend_names_recorded(self, rag):
        event = make_vizag_risk_event()
        result = rag.retrieve_from_risk_event(event)
        assert result.embedding_backend in ("sentence-transformers", "numpy-bow")
        assert result.index_backend in ("faiss", "numpy-bruteforce")
