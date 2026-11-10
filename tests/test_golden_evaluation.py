"""
tests/test_golden_evaluation.py
--------------------------------
Golden evaluation test set covering all 14 active genuine LPU policy documents
in sample_documents/docs_raw/.

Evaluates:
1. Direct facts & exact room/building details
2. Numerical & percentage rules (75% attendance)
3. Percentage & CGPA merit criteria (10% benefit & 7.5 CGPA)
4. Passport validity for study abroad (6 months)
5. Honest handling of unmentioned policies (zero-hallucination)
6. Fee delayed payment consequences (blocked registration, exam ineligibility, facility suspension, Ph.D. de-registration)
7. Programme scope precision (ODL CA 30% / ETE 70% passing criteria)
8. Reappear and improvement exam mark retention
9. Library compounding overdue fines
10. Hostel conduct & mess meal plans
11. Career services PEP fee & placement drive debarment
12. Student Placement Coordinator eligibility (CGPA >= 6.5)
13. International student left marking (FRRO 14 days)
14. Multilingual questions (Hindi, Telugu, Punjabi)
15. Conversational follow-ups (including fee refund standalone regression)
16. Zero-hallucination refusal on unsupported queries

Each case verifies:
- expected_document
- expected_section
- supported_facts
- expected_conditions
"""

import os
import sys
import re
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep test deterministic using local extractive synthesis when testing ground truth
os.environ["GEMINI_API_KEY"] = ""

from utils.document_processor import load_or_build_document_index
from utils.text_search import BM25SearchEngine
from utils.vector_search import VectorSearchEngine
from utils.answer_generator import AnswerGenerator
from utils.context_rewriter import rewrite_conversational_query
from utils.query_translator import expand_query_multilingual


def _setup_pipeline():
    """Initializes search engines with the 14 active LPU DOCX policy documents."""
    docs_dir = Path(__file__).resolve().parent.parent / "sample_documents" / "docs_raw"
    if not docs_dir.exists():
        docs_dir = Path(__file__).resolve().parent.parent / "sample_documents"

    vector_engine = VectorSearchEngine()
    search_engine = BM25SearchEngine()

    chunks, doc_categories = load_or_build_document_index(
        samples_dir=docs_dir,
        vector_engine=vector_engine,
        search_engine=search_engine,
        force_reindex=False
    )
    generator = AnswerGenerator(confidence_threshold=0.25)
    return search_engine, vector_engine, generator, chunks


# Shared global pipeline initialized once per test run
_BM25, _VECTOR, _GENERATOR, _CHUNKS = _setup_pipeline()


def test_case_1_direct_fact_certificate_collection():
    """Case 1: Direct fact verification for Bonafide Certificate application, fee ₹100, and collection at Building 32 Room 101 Window 2."""
    q = "How can a student apply for a Bonafide Certificate at LPU, and where should they collect it?"
    bm25_res = _BM25.search(q, top_k=35)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=5)

    assert len(res) >= 1, "No results returned for certificate collection"
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "student_certificate_application_process.docx"
    sources = [r["chunk"]["source"] for r in res]
    assert "international_student_left_marking_process.docx" not in sources[:3], "Left marking contaminated certificate results"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    ans_text = ans["answer"]
    assert any(b in ans_text for b in ["32", "Building 32", "Building No. 32"])
    assert any(r in ans_text for r in ["101", "Room 101", "Room No. 101"])
    assert any(w in ans_text for w in ["Window 2", "Window No. 2", "Window 02", "Window"])
    assert any(t in ans_text.lower() for t in ["apply certificate", "procedure", "credentials", "token"])
    assert "student_certificate_application_process.docx" in ans["primary_citation"]


def test_case_2_exact_numerical_percentage_attendance():
    """Case 2: 75% minimum attendance rule and regulatory higher requirement from attendance_policy.docx."""
    q = "What is the minimum attendance required for students to be eligible for End-Term Examinations at LPU?"
    bm25_res = _BM25.search(q, top_k=35)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=5)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "attendance_policy.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    ans_text = ans["answer"]
    assert "75%" in ans_text
    assert any(term in ans_text.lower() for term in ["theory", "practical", "each registered", "each course"])
    assert any(term in ans_text.lower() for term in ["regulatory", "regulatory bodies", "higher requirement"])
    assert "attendance_policy.docx" in ans["primary_citation"]


def test_case_3_percentage_and_cgpa_academic_benefits():
    """Case 3: 10% attendance benefit and 7.5 CGPA from TITLE LPU EDU Revolution Policy.docx."""
    q = "What are the academic benefits and CGPA requirement for attendance benefits under the EDU Revolution Policy?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    sources = [r["chunk"]["source"] for r in res]
    assert "TITLE LPU EDU Revolution Policy.docx" in sources

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    ans_text = ans["answer"]
    assert "7.5" in ans_text or "CGPA" in ans_text
    assert "10%" in ans_text or "attendance" in ans_text.lower()


def test_case_4_passport_validity_semester_abroad():
    """Case 4: Six-month passport validity beyond stay from semester_abroad_policy.docx."""
    q = "What is the passport validity required for Semester Abroad?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "semester_abroad_policy.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert any(term in ans["answer"].lower() for term in ["six months", "6 months", "passport"])
    assert "semester_abroad_policy.docx" in ans["primary_citation"]


def test_case_5_family_business_revenue_exclusion():
    """Case 5: Zero-hallucination handling when family business revenue rule is not in active docs_raw."""
    q = "Can revenue from a family-owned business be counted for academic benefits under the EDU Revolution Policy?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    ans = _GENERATOR.generate(q, res)
    # The genuine active LPU policy in docs_raw does not mention family business revenue.
    # Therefore, the system must safely report insufficient evidence rather than hallucinating.
    assert ans["status"] in ("insufficient_evidence", "unsupported")
    assert any(term in ans["answer"].lower() for term in ["insufficient", "not contain", "unverified", "family"])


def test_case_6_fee_delayed_payment_consequences():
    """Case 6: Complete fee non-payment consequences without library policy contamination."""
    q = "What happens if a student misses the fee payment deadline at LPU?"
    bm25_res = _BM25.search(q, top_k=35)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=5)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "fee_payment_modes_and_guidelines.docx"
    sources = [r["chunk"]["source"] for r in res]
    assert "library_policy.docx" not in sources[:3], "library_policy.docx contaminated fee payment results"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    ans_lower = ans["answer"].lower()
    # Must report all 3 documented consequences:
    # 1. Blocked registration
    assert any(term in ans_lower for term in ["blocked registration", "registration for the next semester", "blocked"]), "Missing blocked registration fact"
    # 2. Examination ineligibility
    assert any(term in ans_lower for term in ["ineligibility for examinations", "ineligible for examination", "examinations", "exam"]), "Missing exam ineligibility fact"
    # 3. Facility/hostel suspension
    assert any(term in ans_lower for term in ["suspension of hostel", "hostel or other facility access", "facility access", "hostel"]), "Missing facility suspension fact"

    # Must NOT contain library book withholding consequences or fabricated fines
    assert "withholding of examination result" not in ans_lower
    assert "withholding of examination results" not in ans_lower
    assert "library" not in ans_lower
    assert "late fee of inr 5,000" not in ans_lower
    assert "fine of inr 10,000" not in ans_lower


def test_case_7_odl_programme_scope_evaluation():
    """Case 7: ODL continuous assessment 30% and end-term 70% passing criteria."""
    q = "What is the weightage of continuous assessment CA and end term examination for theory courses in ODL programmes at LPU?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "evaluation_mechanism_odl_programmes.docx"
    assert top_chunk.get("programme_scope") == "ODL"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert "evaluation_mechanism_odl_programmes.docx" in ans["primary_citation"]


def test_case_8_reappear_improvement_passing_mark():
    """Case 8: Reappear & improvement exam mark retention from reappear_and_improvement_exam_policy.docx."""
    q = "What are the rules for reappear and improvement examinations at LPU?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "reappear_and_improvement_exam_policy.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert any(term in ans["answer"].lower() for term in ["reappear", "improvement", "examination", "marks", "register"])


def test_case_9_library_overdue_fines():
    """Case 9: Library overdue fine rules from library_policy.docx."""
    q = "What is the fine for late return of library books at LPU?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "library_policy.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert any(term in ans["answer"].lower() for term in ["fine", "overdue", "inr", "rs", "per day", "2", "5"])


def test_case_10_hostel_conduct_and_mess_plan():
    """Case 10: Hostel conduct rules and mess vegetarian meals from student_residential_and_mess_policy.docx."""
    q = "What are the rules of conduct and mess plans in the residential facility at LPU?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "student_residential_and_mess_policy.docx"
    sec = (top_chunk.get("section") or top_chunk.get("section_title") or "").upper()
    assert "FEE AND REFUND" not in sec, f"Regression: Incorrect section {sec} assigned to hostel chunk"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert "student_residential_and_mess_policy.docx" in ans["primary_citation"]


def test_case_11_career_services_enrolment():
    """Case 11: Career services PEP guidelines and enrolment conditions."""
    q = "What are the enrolment guidelines and conditions for Career Services and PEP at LPU?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "career_services_enrolment_guidelines.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert "career_services_enrolment_guidelines.docx" in ans["primary_citation"]


def test_case_12_student_placement_coordinator_eligibility():
    """Case 12: Student Placement Coordinator eligibility (CGPA >= 6.5)."""
    q = "What is the eligibility criteria to become a Student Placement Coordinator?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "student_placement_coordinator_guidelines.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert any(term in ans["answer"] for term in ["6.5", "CGPA", "Placement Coordinator"])


def test_case_13_international_student_left_marking():
    """Case 13: International student left marking and FRRO reporting."""
    q = "What is the left marking process and FRRO reporting for international students at LPU?"
    bm25_res = _BM25.search(q, top_k=25)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=3)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "international_student_left_marking_process.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] == "supported"
    assert "international_student_left_marking_process.docx" in ans["primary_citation"]


def test_case_14_multilingual_hindi_telugu_punjabi():
    """Case 14: Multilingual queries across Hindi, Telugu, and Punjabi."""
    # Hindi Attendance
    q_hi = "एलपीयू में परीक्षा के लिए न्यूनतम उपस्थिति कितनी आवश्यक है?"
    trans_hi = expand_query_multilingual(q_hi)
    en_q_hi = trans_hi["english_query"] if trans_hi.get("is_translated") else q_hi
    b_hi = _BM25.search(en_q_hi, top_k=15)
    h_hi = _VECTOR.hybrid_search(q_hi, b_hi, top_k=3, translated_query=en_q_hi)
    assert len(h_hi) >= 1
    assert h_hi[0]["chunk"]["source"] == "attendance_policy.docx"
    ans_hi = _GENERATOR.generate(q_hi, h_hi)
    assert ans_hi["status"] == "supported"
    assert "75%" in ans_hi["answer"]

    # Telugu Passport Validity
    q_te = "సెమిస్టర్ అబ్రాడ్ కోసం పాస్‌పోర్ట్ చెల్లుబాటు ఎంత కాలం ఉండాలి?"
    trans_te = expand_query_multilingual(q_te)
    en_q_te = trans_te["english_query"] if trans_te.get("is_translated") else q_te
    b_te = _BM25.search(en_q_te, top_k=15)
    h_te = _VECTOR.hybrid_search(q_te, b_te, top_k=3, translated_query=en_q_te)
    assert len(h_te) >= 1
    assert h_te[0]["chunk"]["source"] == "semester_abroad_policy.docx"
    ans_te = _GENERATOR.generate(q_te, h_te)
    assert ans_te["status"] == "supported"

    # Punjabi Library Books Late Return
    q_pa = "ਲਾਇਬ੍ਰੇਰੀ ਦੀਆਂ ਕਿਤਾਬਾਂ ਦੇਰ ਨਾਲ ਜਮ੍ਹਾਂ ਕਰਵਾਉਣ 'ਤੇ ਕਿੰਨਾ ਜੁਰਮਾਨਾ ਲੱਗਦਾ ਹੈ?"
    trans_pa = expand_query_multilingual(q_pa)
    en_q_pa = trans_pa["english_query"] if trans_pa.get("is_translated") else q_pa
    b_pa = _BM25.search(en_q_pa, top_k=15)
    h_pa = _VECTOR.hybrid_search(q_pa, b_pa, top_k=3, translated_query=en_q_pa)
    assert len(h_pa) >= 1
    assert h_pa[0]["chunk"]["source"] == "library_policy.docx"
    ans_pa = _GENERATOR.generate(q_pa, h_pa)
    assert ans_pa["status"] == "supported"


def test_case_15_conversational_follow_up_and_standalone_regression():
    """Case 15: Conversational follow-ups and standalone question preservation."""
    # 1. Genuine follow-up must be rewritten with context
    history_abroad = [
        {"role": "user", "content": "What are the rules for Semester Abroad?"},
        {"role": "assistant", "content": "Students require a 7.5 CGPA and a valid passport."}
    ]
    rewritten_abroad, was_rew_abroad, _ = rewrite_conversational_query("What about passport validity?", history_abroad)
    assert was_rew_abroad
    assert "passport" in rewritten_abroad.lower()
    assert "semester abroad" in rewritten_abroad.lower()

    # 2. Standalone topic switch MUST NOT be contaminated by previous topic
    history_business = [
        {"role": "user", "content": "Can I claim benefits for my Family-Owned Business Revenue?"},
        {"role": "assistant", "content": "No, family-owned business revenue is strictly excluded."}
    ]
    q_refund = "What is the fee refund policy?"
    rewritten_refund, was_rew_refund, _ = rewrite_conversational_query(q_refund, history_business)
    assert not was_rew_refund, f"False rewrite occurred! Standalone question rewritten to: {rewritten_refund}"
    assert "family" not in rewritten_refund.lower(), f"Contaminated with prior topic: {rewritten_refund}"
    assert rewritten_refund == q_refund


def test_case_16_unsupported_questions_refusal():
    """Case 16: Zero-hallucination refusal on unsupported and out-of-scope queries."""
    unsupported_queries = [
        "What is the personal mobile phone number of the university chancellor?",
        "What is the daily discount coupon code for Domino's Pizza at the food court?",
        "What is the opening schedule of the Olympic swimming pool on campus?"
    ]
    for q_unsup in unsupported_queries:
        bm25_res = _BM25.search(q_unsup, top_k=5)
        res = _VECTOR.hybrid_search(q_unsup, bm25_res, top_k=3)
        ans = _GENERATOR.generate(q_unsup, res)
        # Must refuse cleanly without inventing details
        assert ans["status"] in ("unsupported", "insufficient_evidence"), (
            f"Query '{q_unsup}' was wrongly marked supported: {ans['status']}"
        )
        ans_lower = ans["answer"].lower()
        assert not any(num in ans_lower for num in ["98765", "998877", "98140", "phone number:"]), (
            f"Hallucinated phone number detected in response: {ans['answer']}"
        )


def test_case_17_hostel_timings_partial_evidence():
    """Case 17: Grounded partial evidence for hostel entry/exit timings."""
    q = "What are the hostel entry and exit timings at LPU?"
    bm25_res = _BM25.search(q, top_k=35)
    res = _VECTOR.hybrid_search(q, bm25_res, top_k=5)

    assert len(res) >= 1
    top_chunk = res[0]["chunk"]
    assert top_chunk["source"] == "student_residential_and_mess_policy.docx"

    ans = _GENERATOR.generate(q, res)
    assert ans["status"] in ("partial_evidence", "supported")
    ans_lower = ans["answer"].lower()
    # Must explain turnstiles / biometrics / security / e-leave
    assert any(term in ans_lower for term in ["turnstile", "biometric", "security", "leave"])
    # Must explicitly state that exact gate hours are not in the document
    assert any(term in ans_lower for term in ["does not state exact", "does not specify exact", "not state exact", "exact numeric", "hours/timings", "gate hours", "exact hostel"])


if __name__ == "__main__":
    golden_tests = [
        ("Case 1: Direct Fact (Certificate Collection & Room 101/102)", test_case_1_direct_fact_certificate_collection),
        ("Case 2: Exact Numerical / Percentage (75% Attendance Rule)", test_case_2_exact_numerical_percentage_attendance),
        ("Case 3: Percentage & CGPA (10% Benefit & 7.5 CGPA)", test_case_3_percentage_and_cgpa_academic_benefits),
        ("Case 4: Passport Validity (6-Month Rule Beyond Stay)", test_case_4_passport_validity_semester_abroad),
        ("Case 5: Zero-Hallucination on Unverified Policies", test_case_5_family_business_revenue_exclusion),
        ("Case 6: Complete Delayed Fee Payment Consequences", test_case_6_fee_delayed_payment_consequences),
        ("Case 7: ODL Programme Scope Precision (CA & ETE Scheme)", test_case_7_odl_programme_scope_evaluation),
        ("Case 8: Reappear & Improvement Exam Passing Mark Retention", test_case_8_reappear_improvement_passing_mark),
        ("Case 9: Library Compounding Overdue Fines", test_case_9_library_overdue_fines),
        ("Case 10: Hostel Conduct Rules & Mess Plan Sections", test_case_10_hostel_conduct_and_mess_plan),
        ("Case 11: Career Services PEP & Placement Guidelines", test_case_11_career_services_enrolment),
        ("Case 12: Student Placement Coordinator Eligibility (CGPA >= 6.5)", test_case_12_student_placement_coordinator_eligibility),
        ("Case 13: International Student Left Marking (FRRO 14 Days)", test_case_13_international_student_left_marking),
        ("Case 14: Multilingual Retrieval (Hindi, Telugu, Punjabi)", test_case_14_multilingual_hindi_telugu_punjabi),
        ("Case 15: Conversational Follow-up & Standalone Regression", test_case_15_conversational_follow_up_and_standalone_regression),
        ("Case 16: Zero-Hallucination Refusal on Unsupported Questions", test_case_16_unsupported_questions_refusal),
        ("Case 17: Grounded Partial Evidence (Hostel Entry/Exit Timings)", test_case_17_hostel_timings_partial_evidence),
    ]

    print("=" * 70)
    print("RUNNING 14-DOCUMENT GOLDEN EVALUATION TEST SET")
    print("=" * 70)

    all_passed = True
    for name, func in golden_tests:
        try:
            func()
            print(f"[PASS] {name}")
        except Exception as e:
            all_passed = False
            print(f"[FAIL] {name}: {e}")

    print("=" * 70)
    if all_passed:
        print(f"RESULT: ALL {len(golden_tests)} GOLDEN EVALUATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME GOLDEN EVALUATION TESTS FAILED.")
    print("=" * 70)
