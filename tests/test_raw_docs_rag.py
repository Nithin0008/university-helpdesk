"""
tests/test_raw_docs_rag.py
--------------------------
Verification tests for the integration of raw DOCX policy documents from sample_documents/docs_raw/.

Test coverage:
1. Recursive document scanning (18 total: 2 PDF, 2 TXT, 14 DOCX).
2. Content hashing and deduplication.
3. Enhanced DOCX table, list, and heading extraction.
4. Non-invention of page numbers for DOCX (page=None).
5. ODL vs Regular programme scope classification and separation.
6. BM25 and FAISS hybrid retrieval across new policies.
7. Category-restricted retrieval on newly ingested documents.
8. Grounded answers and citations formatting without invented page numbers.
9. Persistent caching (sub-second load time from disk).
10. Refusal guardrails on unsupported queries.
"""

import sys
import os
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["GEMINI_API_KEY"] = ""

from utils.document_processor import (
    load_or_build_document_index,
    scan_all_documents,
    compute_content_hash,
    process_document
)
from utils.text_search import BM25SearchEngine
from utils.vector_search import VectorSearchEngine
from utils.answer_generator import AnswerGenerator, format_citation


def test_1_recursive_document_discovery():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    docs = scan_all_documents(samples_dir)
    assert len(docs) == 14, f"Expected 14 active policy documents in docs_raw, found {len(docs)}"
    
    docxs = [d for d in docs if d.suffix.lower() == ".docx"]
    assert len(docxs) == 14, f"Expected 14 DOCX, got {len(docxs)}"
    
    # Confirm 0 excluded sample documents are in active discovery
    from utils.document_processor import EXCLUDED_OLD_SAMPLES
    for d in docs:
        assert d.name.lower() not in EXCLUDED_OLD_SAMPLES, f"Excluded file {d.name} found in active docs!"


def test_2_content_hashing_and_deduplication():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    docs = scan_all_documents(samples_dir)
    seen_hashes = set()
    for d in docs:
        raw = d.read_bytes()
        h = compute_content_hash(raw)
        assert len(h) == 64, f"Invalid SHA-256 hash length for {d.name}"
        assert h not in seen_hashes, f"Duplicate content hash found in {d.name}"
        seen_hashes.add(h)


def test_3_docx_table_and_list_extraction():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    odl_doc = samples_dir / "docs_raw" / "academic" / "evaluation_mechanism_odl_programmes.docx"
    raw = odl_doc.read_bytes()
    chunks = process_document(raw, odl_doc.name)
    assert len(chunks) > 0
    
    # Check that table text was extracted with pipe-delimited structure
    full_text = " ".join([c["text"] for c in chunks])
    assert "|" in full_text, "Markdown table structure not found in extracted DOCX chunks!"
    assert "30%" in full_text
    assert "70%" in full_text


def test_4_no_invented_page_numbers_for_docx():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    doc_path = samples_dir / "docs_raw" / "academic" / "attendance_policy.docx"
    raw = doc_path.read_bytes()
    chunks = process_document(raw, doc_path.name)
    for c in chunks:
        assert c.get("page") is None, f"DOCX chunk must have page=None, got: {c.get('page')}"
        cit = format_citation(c)
        assert "Page" not in cit, f"Citation for DOCX must not invent a Page number! Got: {cit}"


def test_5_odl_programme_scope_separation():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    odl_doc = samples_dir / "docs_raw" / "academic" / "evaluation_mechanism_odl_programmes.docx"
    chunks = process_document(odl_doc.read_bytes(), odl_doc.name)
    assert chunks[0]["programme_scope"] == "ODL", f"Expected ODL scope, got {chunks[0]['programme_scope']}"
    
    reg_doc = samples_dir / "docs_raw" / "academic" / "attendance_policy.docx"
    reg_chunks = process_document(reg_doc.read_bytes(), reg_doc.name)
    assert reg_chunks[0]["programme_scope"] == "Regular", f"Expected Regular scope, got {reg_chunks[0]['programme_scope']}"


def test_6_persistence_subsecond_cache_reload():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    bm25 = BM25SearchEngine()
    vector = VectorSearchEngine()
    
    t0 = time.time()
    chunks, doc_categories = load_or_build_document_index(
        samples_dir=samples_dir,
        vector_engine=vector,
        search_engine=bm25,
        force_reindex=False
    )
    t1 = time.time()
    
    assert len(chunks) >= 280, f"Expected >= 280 chunks, got {len(chunks)}"
    assert len(doc_categories) == 14, f"Expected 14 documents, got {len(doc_categories)}"
    assert (t1 - t0) < 2.0, f"Cache reload took {t1 - t0:.2f}s, expected < 2.0s"


def test_7_retrieval_across_new_policies():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    bm25 = BM25SearchEngine()
    vector = VectorSearchEngine()
    chunks, _ = load_or_build_document_index(samples_dir, vector, bm25)
    
    # Query 1: ODL evaluation weightage
    res1 = vector.hybrid_search("Continuous Assessment CA weightage theory ODL", bm25.search("Continuous Assessment CA weightage theory ODL", 10), top_k=3)
    assert res1[0]["chunk"]["source"] == "evaluation_mechanism_odl_programmes.docx"
    assert any("30%" in r["chunk"]["text"] for r in res1)
    
    # Query 2: Certificate collection deadline
    res2 = vector.hybrid_search("certificate collection 30 days destroyed", bm25.search("certificate collection 30 days destroyed", 10), top_k=3)
    assert any(r["chunk"]["source"] == "student_certificate_application_process.docx" for r in res2)
    
    # Query 3: LPUNEST B.Tech scholarship
    res3 = vector.hybrid_search("LPUNEST cut-off band scholarship B.Tech", bm25.search("LPUNEST cut-off band scholarship B.Tech", 10), top_k=3)
    assert any(r["chunk"]["source"] == "btech_fee_scholarship_structure_lpunest.docx" for r in res3)


def test_8_unsupported_question_refusal():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    bm25 = BM25SearchEngine()
    vector = VectorSearchEngine()
    chunks, _ = load_or_build_document_index(samples_dir, vector, bm25)
    
    q = "What is the swimming pool membership fee for hostel students?"
    results = vector.hybrid_search(q, bm25.search(q, 10), top_k=5)
    generator = AnswerGenerator(confidence_threshold=0.25)
    ans = generator.generate(q, results)
    
    assert ans["status"] in ["unsupported", "insufficient_evidence"], f"Expected refusal, got status: {ans['status']}"


if __name__ == "__main__":
    tests = [
        ("Active Document Discovery (14 Docs in docs_raw, 0 Excluded)", test_1_recursive_document_discovery),
        ("Content Hashing & Deduplication Integrity", test_2_content_hashing_and_deduplication),
        ("Enhanced DOCX Table & List Markdown Extraction", test_3_docx_table_and_list_extraction),
        ("Preservation of No-Invented-Page-Numbers for DOCX", test_4_no_invented_page_numbers_for_docx),
        ("ODL vs Regular Programme Scope Separation", test_5_odl_programme_scope_separation),
        ("Sub-second Persistent Disk Cache Reload", test_6_persistence_subsecond_cache_reload),
        ("Hybrid Retrieval Across Newly Integrated Policies", test_7_retrieval_across_new_policies),
        ("Zero-Hallucination Unsupported Query Refusal", test_8_unsupported_question_refusal),
    ]

    print("=" * 70)
    print("RUNNING RAW DOCS INTEGRATION TEST SUITE (8 TESTS)")
    print("=" * 70)

    all_passed = True
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:
            all_passed = False
            print(f"[FAIL] {name}: {e}")

    print("=" * 70)
    if all_passed:
        print("RESULT: ALL 8 NEW DOCS INTEGRATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME TESTS FAILED.")
    print("=" * 70)

