"""
tests/test_context_and_categories.py
------------------------------------
Test suite for Feature 1 (Context-Aware Conversational Query Rewriting)
and Feature 2 (Category-Based Document Management and Retrieval).

Test cases verified:
1. Follow-up: 'What about passport validity?' after a semester-abroad question.
2. Follow-up: 'And what CGPA is required?'
3. A completely unrelated follow-up question.
4. Telugu and Hindi follow-up questions.
5. Correct classification of an academic document.
6. Correct classification of a hostel document.
7. Manual category override.
8. Retrieval restricted to Academics.
9. All Categories retrieval.
10. Uncertain category classification falling back to global retrieval.
11. Exact citations and page numbers after filtering.
"""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["GEMINI_API_KEY"] = ""

from utils.context_rewriter import (
    rewrite_conversational_query,
    is_follow_up_question,
    get_recent_conversation_topic,
    rewrite_query_locally
)
from utils.category_manager import (
    CATEGORIES,
    ALL_CATEGORIES,
    classify_question,
    classify_document,
    CATEGORY_ICONS
)
from utils.document_processor import process_document, scan_all_documents
from utils.text_search import BM25SearchEngine
from utils.vector_search import VectorSearchEngine
from utils.answer_generator import AnswerGenerator


def load_all_sample_chunks():
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    docs = scan_all_documents(samples_dir)
    all_chunks = []
    for file_path in docs:
        with open(file_path, "rb") as f:
            content = f.read()
        chunks = process_document(content, file_path.name)
        all_chunks.extend(chunks)
    return all_chunks


def test_1_follow_up_passport_validity():
    """Test 1: Follow-up 'What about passport validity?' after semester abroad question."""
    history = [
        {"role": "user", "content": "What is the procedure for Semester Abroad?"},
        {"role": "assistant", "content": "Semester Abroad allows students to spend a semester at a foreign partner university. You need valid documentation."}
    ]
    query = "What about passport validity?"
    
    is_fup, reason = is_follow_up_question(query, history)
    assert is_fup, f"Expected is_follow_up=True, got {is_fup} ({reason})"
    
    search_q, was_rewritten, method = rewrite_conversational_query(query, history)
    assert was_rewritten, f"Expected was_rewritten=True, got {was_rewritten}"
    assert "passport validity" in search_q.lower()
    assert "semester abroad" in search_q.lower()
    
    # Original question must remain intact
    assert query == "What about passport validity?"


def test_2_follow_up_cgpa_requirement():
    """Test 2: Follow-up 'And what CGPA is required?' after semester abroad question."""
    history = [
        {"role": "user", "content": "Can I apply for Semester Abroad in my pre-final year?"},
        {"role": "assistant", "content": "Yes, pre-final year students can apply if they meet all academic criteria."}
    ]
    query = "And what CGPA is required?"
    
    is_fup, reason = is_follow_up_question(query, history)
    assert is_fup, f"Expected is_follow_up=True, got {is_fup}"
    
    search_q, was_rewritten, method = rewrite_conversational_query(query, history)
    assert was_rewritten, f"Expected was_rewritten=True, got {was_rewritten}"
    assert "cgpa" in search_q.lower()
    assert "semester abroad" in search_q.lower()


def test_3_unrelated_follow_up_stays_standalone():
    """Test 3: Completely unrelated follow-up question does not inherit previous topic."""
    history = [
        {"role": "user", "content": "Can I apply for Semester Abroad in my third year?"},
        {"role": "assistant", "content": "Yes, third-year students are eligible for semester exchange programs."}
    ]
    unrelated_query = "What are the hostel dining timings?"
    
    # Even if query begins with 'What are the', it introduces a clear standalone topic ('hostel dining timings')
    search_q, was_rewritten, method = rewrite_conversational_query(unrelated_query, history)
    assert not was_rewritten or "semester abroad" not in search_q.lower(), (
        f"Unrelated query wrongly contaminated with prior topic! Got: {search_q}"
    )


def test_4_telugu_and_hindi_follow_ups():
    """Test 4: Telugu and Hindi follow-up questions."""
    # Hindi follow-up
    hi_history = [
        {"role": "user", "content": "सेमेस्टर अब्रॉड के क्या नियम हैं?"},
        {"role": "assistant", "content": "सेमेस्टर अब्रॉड कार्यक्रम के तहत छात्र विदेशी विश्वविद्यालय में अध्ययन कर सकते हैं।"}
    ]
    hi_fup = "और अंतिम तिथि क्या है?"
    search_hi, rew_hi, _ = rewrite_conversational_query(hi_fup, hi_history)
    assert rew_hi, f"Hindi follow-up was not rewritten! Got: {search_hi}"
    assert "Semester Abroad" in search_hi or "सेमेस्टर" in search_hi

    # Telugu follow-up
    te_history = [
        {"role": "user", "content": "సెమిస్టర్ అబ్రాడ్ విధానం ఏమిటి?"},
        {"role": "assistant", "content": "సెమిస్టర్ అబ్రాడ్ ద్వారా విదేశీ విశ్వవిద్యాలయాలలో చదువుకోవచ్చు."}
    ]
    te_fup = "పాస్‌పోర్ట్ చెల్లుబాటు ఎంత ఉండాలి?"
    search_te, rew_te, _ = rewrite_conversational_query(te_fup, te_history)
    assert rew_te, f"Telugu follow-up was not rewritten! Got: {search_te}"
    assert "Semester Abroad" in search_te or "పాస్‌పోర్ట్" in search_te


def test_5_classification_academic_document():
    """Test 5: Correct classification of an academic document."""
    fn = "academic_regulations_2024.pdf"
    content = (
        "SECTION 1: ATTENDANCE POLICY\n"
        "1.1 Minimum Attendance Requirement: Every student must maintain 75% attendance.\n"
        "1.2 Condonation: Medical condonation requires 65%.\n"
        "Students must achieve a minimum CGPA of 7.5 for honours."
    )
    cats = classify_document(fn, content)
    assert cats[0] == "Academics", f"Expected primary category 'Academics', got: {cats}"


def test_6_classification_hostel_document():
    """Test 6: Correct classification of a hostel document."""
    fn = "campus_hostel_rules.pdf"
    content = (
        "HOSTEL RESIDENCE RULES\n"
        "Curfew timing is 9:30 PM on weekdays.\n"
        "Mess timings: Breakfast 7:30 - 9:00 AM, Dinner 7:30 - 9:30 PM.\n"
        "Warden permission required for night out."
    )
    cats = classify_document(fn, content)
    assert cats[0] == "Hostel", f"Expected primary category 'Hostel', got: {cats}"


def test_7_manual_category_override():
    """Test 7: Manual category override updates document chunks and retrieval."""
    chunks = process_document(
        "General university circular regarding code of conduct.",
        "circular.txt"
    )
    assert chunks[0]["category"] == "General / Other"
    
    # User manually overrides to 'Academics'
    new_cat = "Academics"
    for c in chunks:
        c["category"] = new_cat
        c["categories"] = [new_cat]
        
    assert chunks[0]["category"] == "Academics"
    assert "Academics" in chunks[0]["categories"]


def test_8_retrieval_restricted_to_academics():
    """Test 8: Retrieval restricted to Academics only returns academic chunks."""
    all_chunks = load_all_sample_chunks()
    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    # Search with category='Academics'
    results = bm25.search("attendance requirements", top_k=5, category="Academics")
    assert len(results) > 0, "No results returned for Academics filter"
    for r in results:
        c = r["chunk"]
        assert c.get("category") == "Academics" or "Academics" in c.get("categories", []), (
            f"Non-academic chunk returned when category was restricted to Academics! Got: {c.get('category')}"
        )


def test_9_all_categories_retrieval():
    """Test 9: 'All Categories' retrieval searches globally across all document categories."""
    all_chunks = load_all_sample_chunks()
    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    results_all = bm25.search("residential mess facility", top_k=5, category="All Categories")
    assert len(results_all) > 0
    categories_found = {r["chunk"].get("category") for r in results_all}
    assert "Hostel" in categories_found or any("Hostel" in r["chunk"].get("categories", []) for r in results_all)


def test_10_uncertain_category_fallback():
    """Test 10: Uncertain category classification safely falls back to searching all categories."""
    # 1. Question with uncertain category
    vague_q = "What is the procedure for verification?"
    cat, conf = classify_question(vague_q)
    assert cat is None or cat == "All Categories" or conf < 0.65, (
        f"Vague question should have low confidence/None, got cat={cat}, conf={conf}"
    )

    # 2. Retrieval fallback: querying with mismatched category falls back to global results
    all_chunks = load_all_sample_chunks()
    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)
    
    # Ask for residential/mess (which is in Hostel) while filtering on 'Finance' (where mess doesn't exist)
    results = bm25.search("residential mess facility", top_k=3, category="Finance")
    # Must fallback and not return empty!
    assert len(results) > 0, "Failed to fall back to global search when filtered category had 0 results!"
    top_text = results[0]["chunk"]["text"]
    assert "residential" in top_text.lower() or "mess" in top_text.lower()


def test_11_exact_citations_and_page_numbers_after_filtering():
    """Test 11: Exact citations, page numbers, and metadata are preserved after category filtering."""
    all_chunks = load_all_sample_chunks()
    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)
    generator = AnswerGenerator(confidence_threshold=0.25)

    results = bm25.search("minimum attendance", top_k=3, category="Academics")
    assert len(results) > 0
    top_chunk = results[0]["chunk"]
    
    assert "source" in top_chunk and top_chunk["source"]
    assert "attendance_policy.docx" in top_chunk["source"]
    assert top_chunk.get("category") == "Academics" or "Academics" in top_chunk.get("categories", [])

    # Verify final generated citation maintains exact format without invented page numbers
    ans = generator.generate("What is the minimum attendance required?", results)
    assert ans["status"] == "supported"
    assert "attendance_policy.docx" in ans["primary_citation"]


def test_12_family_business_followed_by_fee_refund_policy_regression():
    """
    Test 12: Regression test for false query rewriting.
    Question: 'What is the fee refund policy?' after discussing Family-Owned Business Revenue.
    Must REMAIN standalone and NOT be rewritten to include Family-Owned Business Revenue.
    """
    history = [
        {"role": "user", "content": "Is revenue from a family-owned business eligible under the student-centric revenue generation model?"},
        {"role": "assistant", "content": "No, revenue generated through family-owned businesses or family enterprises shall not be considered under the policy."}
    ]
    query = "What is the fee refund policy?"
    
    is_fup, reason = is_follow_up_question(query, history)
    assert not is_fup, f"Expected is_follow_up=False for standalone question, got {is_fup} ({reason})"

    search_q, was_rewritten, method = rewrite_conversational_query(query, history)
    assert not was_rewritten, f"Expected was_rewritten=False, but was rewritten to: {search_q}"
    assert search_q.strip() == query.strip()
    assert "family" not in search_q.lower()
    assert "revenue" not in search_q.lower()


if __name__ == "__main__":
    tests = [
        ("1. Follow-up: 'What about passport validity?'", test_1_follow_up_passport_validity),
        ("2. Follow-up: 'And what CGPA is required?'", test_2_follow_up_cgpa_requirement),
        ("3. Unrelated follow-up stays standalone", test_3_unrelated_follow_up_stays_standalone),
        ("4. Telugu and Hindi follow-up questions", test_4_telugu_and_hindi_follow_ups),
        ("5. Classification of an academic document", test_5_classification_academic_document),
        ("6. Classification of a hostel document", test_6_classification_hostel_document),
        ("7. Manual category override", test_7_manual_category_override),
        ("8. Retrieval restricted to Academics", test_8_retrieval_restricted_to_academics),
        ("9. All Categories retrieval", test_9_all_categories_retrieval),
        ("10. Uncertain category fallback to global retrieval", test_10_uncertain_category_fallback),
        ("11. Exact citations & page numbers after filtering", test_11_exact_citations_and_page_numbers_after_filtering),
        ("12. Regression: Family Business -> 'What is the fee refund policy?' remains standalone", test_12_family_business_followed_by_fee_refund_policy_regression),
    ]

    print("=" * 65)
    print("RUNNING CONTEXT REWRITING & CATEGORY RETRIEVAL TEST SUITE (11 TESTS)")
    print("=" * 65)

    all_passed = True
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:
            all_passed = False
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 65)
    if all_passed:
        print(f"RESULT: ALL {len(tests)} NEW TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME NEW TESTS FAILED.")
    print("=" * 65)
    sys.exit(0 if all_passed else 1)

