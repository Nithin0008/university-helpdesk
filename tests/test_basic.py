"""
tests/test_basic.py
-------------------
Unit tests for Milestone 1: Document processing, chunking, and search.
"""

import os
import sys
import re
from pathlib import Path

# Add project root to Python module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Keep unit tests deterministic using local extractive synthesizer
os.environ["GEMINI_API_KEY"] = ""

from utils.document_processor import (
    extract_text_from_txt,
    chunk_text,
    process_document
)
from utils.text_search import BM25SearchEngine, tokenize


def test_tokenize():
    # Test English and multilingual Unicode words
    tokens = tokenize("What is attendance? नमस्ते ధన్యవాਦాలు ਸਤਿ ਸ਼੍ਰੀ ਅਕਾਲ")
    assert "attendance" in tokens
    assert "नमस्ते" in tokens
    assert len(tokens) >= 5


def test_txt_extraction_and_chunking():
    sample_text = (
        "Apex Global University Hostel Policy.\n"
        "Curfew timing is 9:30 PM on weekdays.\n"
        "Late entry requires permission from the warden.\n"
    ) * 10

    pages = extract_text_from_txt(sample_text, "policy.txt")
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["source"] == "policy.txt"

    chunks = chunk_text(pages, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["source"] == "policy.txt"
        assert chunk["page"] == 1
        assert "chunk_id" in chunk
        assert len(chunk["text"]) <= 350


def test_bm25_search_scoring():
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Every student must maintain at least 75% attendance for semester exams.",
            "source": "handbook.txt",
            "page": 1,
            "char_count": 71
        },
        {
            "chunk_id": "c2",
            "text": "The hostel mess serves breakfast between 7:30 AM and 9:15 AM every day.",
            "source": "hostel.txt",
            "page": 2,
            "char_count": 72
        },
        {
            "chunk_id": "c3",
            "text": "Library books can be borrowed for up to 14 days by undergraduate students.",
            "source": "library.txt",
            "page": 1,
            "char_count": 74
        }
    ]

    engine = BM25SearchEngine()
    engine.index_chunks(chunks)

    # Query 1: Attendance
    results = engine.search("attendance requirement for exams", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk"]["chunk_id"] == "c1"
    assert results[0]["chunk"]["source"] == "handbook.txt"
    assert "attendance" in results[0]["matched_terms"]

    # Query 2: Hostel mess
    results = engine.search("hostel breakfast timing", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk"]["chunk_id"] == "c2"

    # Query 3: Non-existent term
    results = engine.search("quantum astrophysics spaceship rocket", top_k=2)
    assert len(results) == 0


def get_legacy_samples_dir() -> Path:
    archive = Path(__file__).resolve().parent.parent / "sample_documents" / "archive_excluded"
    if archive.exists() and any(archive.glob("*.*")):
        return archive
    return Path(__file__).resolve().parent.parent / "sample_documents"


def test_pdf_extraction():
    """Verify PDF extraction preserves multi-page numbers."""
    pdf_path = get_legacy_samples_dir() / "sample_hostel_policy.pdf"
    assert pdf_path.exists(), f"Sample PDF missing at {pdf_path}"

    with open(pdf_path, "rb") as f:
        content = f.read()

    chunks = process_document(content, "sample_hostel_policy.pdf")
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    pages = {c["page"] for c in chunks}
    assert 1 in pages, "Page 1 missing in PDF extraction"
    assert 2 in pages, "Page 2 missing in PDF extraction"


def test_sample_documents_indexing():
    """Verify all files in sample_documents can be processed and indexed."""
    samples_dir = get_legacy_samples_dir()
    all_chunks = []

    for file_path in samples_dir.glob("*.*"):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    assert len(all_chunks) > 10, f"Expected >10 total chunks from sample files, got {len(all_chunks)}"

    engine = BM25SearchEngine()
    engine.index_chunks(all_chunks)

    # Test query across the indexed documents
    res_attendance = engine.search("attendance criteria 75 percent", top_k=1)
    assert len(res_attendance) == 1
    assert "sample_university_policy.txt" in res_attendance[0]["chunk"]["source"]

    res_library = engine.search("library book overdue fine", top_k=1)
    assert len(res_library) == 1
    assert "sample_library_and_fee_rules.txt" in res_library[0]["chunk"]["source"]


def test_streamlit_app_flow():
    """Verify Streamlit app initializes, loads samples, and searches via buttons and typing."""
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=240)
    at.run()
    assert not at.exception, "App crashed on startup"

    # 1. Click Load Samples button in sidebar
    load_btn = [b for b in at.button if "Load Samples" in b.label][0]
    load_btn.click().run()
    assert len(at.session_state.loaded_files) >= 3, "Sample documents not loaded into session state"
    assert len(at.session_state.indexed_chunks) >= 10, "Passages not indexed in session state"

    # 2. Test clicking example question button (btn_q1)
    btn_q1 = [b for b in at.button if b.key == "btn_q1"][0]
    btn_q1.click().run()
    assert not at.exception, "App crashed clicking example question button"
    assert at.session_state.search_input == "What is the minimum attendance required for exams?"
    search_input_widget = [t for t in at.text_input if t.key == "search_input"][0]
    assert search_input_widget.value == "What is the minimum attendance required for exams?"

    # Verify search result cards rendered from button click
    cards = [m.value for m in at.markdown if "result-card" in m.value and "Source:" in m.value]
    assert len(cards) >= 1, "Expected search result cards to be rendered on button click"
    assert "attendance" in cards[0].lower(), "Top result missing expected attendance policy"

    # 3. Test clicking another button (btn_q6: Hostel Curfew)
    btn_q6 = [b for b in at.button if b.key == "btn_q6"][0]
    btn_q6.click().run()
    assert at.session_state.search_input == "What are the hostel curfew timings on weekdays?"
    cards_curfew = [m.value for m in at.markdown if "result-card" in m.value and "Source:" in m.value]
    assert len(cards_curfew) >= 1, "Expected search result cards for curfew"


def test_multilingual_and_paraphrase_semantic_search():
    """Verify semantic vector search handles English paraphrasing, Hindi, Telugu, and Punjabi."""
    from utils.vector_search import VectorSearchEngine

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in samples_dir.glob("*.*"):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    assert v_engine.num_chunks == len(all_chunks)
    assert v_engine.embeddings is not None
    assert v_engine.backend in ["faiss", "numpy"]

    # 1. Paraphrase test: Keywords 'discount', 'top marks' never appear in document
    # Document contains: 'Merit Scholarship', '50% tuition fee waiver', 'CGPA 9.0'
    res_paraphrase = v_engine.search("Can top scoring students get a discount on tuition fees?", top_k=3)
    assert len(res_paraphrase) >= 1
    found_benefit = any(
        any(w in r["chunk"]["text"].lower() for w in ["scholarship", "tuition", "benefit"])
        for r in res_paraphrase
    )
    assert found_benefit, "No relevant scholarship, tuition, or academic benefit chunk found in top 3"
    assert res_paraphrase[0]["score"] > 0.4

    # 2. Hindi test: Attendance
    res_hindi = v_engine.search("परीक्षा में बैठने के लिए न्यूनतम उपस्थिति कितनी जरूरी है?", top_k=1)
    assert len(res_hindi) == 1
    assert "75%" in res_hindi[0]["chunk"]["text"] or "attendance" in res_hindi[0]["chunk"]["text"].lower()

    # 3. Telugu test: Hostel curfew (Must return actual curfew, not mess/dining)
    res_telugu = v_engine.search("హాస్టల్ రాత్రి కర్ఫ్యూ సమయం ఎన్ని గంటలకు?", top_k=1)
    assert len(res_telugu) == 1
    top_t_text = res_telugu[0]["chunk"]["text"].lower()
    assert "curfew" in top_t_text or "కర్ఫ్యూ" in top_t_text, "Telugu search returned passage missing curfew"
    assert "breakfast" not in top_t_text and "dining hall" not in top_t_text, "Telugu search wrongly returned dining timings"

    # 4. Punjabi test: Library fine
    res_punjabi = v_engine.search("ਲਾਇਬ੍ਰੇਰੀ ਦੀਆਂ ਕਿਤਾਬਾਂ ਦੇਰ ਨਾਲ ਦੇਣ 'ਤੇ ਕੀ ਜੁਰਮਾਨਾ ਲੱਗਦਾ ਹੈ?", top_k=1)
    assert len(res_punjabi) == 1
    assert "library" in res_punjabi[0]["chunk"]["source"].lower() or "fine" in res_punjabi[0]["chunk"]["text"].lower() or "book" in res_punjabi[0]["chunk"]["text"].lower()


def test_hybrid_search_combination():
    """Verify hybrid search correctly merges semantic and keyword rankings."""
    from utils.vector_search import VectorSearchEngine

    chunks = [
        {
            "chunk_id": "c1",
            "text": "The minimum attendance criteria is 75 percent for semester exams.",
            "source": "attendance.txt",
            "page": 1,
            "char_count": 68
        },
        {
            "chunk_id": "c2",
            "text": "Merit scholarships award a 50 percent tuition fee waiver to top academic performers.",
            "source": "scholarships.txt",
            "page": 1,
            "char_count": 84
        }
    ]

    bm25 = BM25SearchEngine()
    bm25.index_chunks(chunks)
    bm25_res = bm25.search("75 percent attendance", top_k=2)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(chunks)

    hybrid_res = v_engine.hybrid_search("75 percent attendance", bm25_res, top_k=2)
    assert len(hybrid_res) == 2
    assert hybrid_res[0]["chunk"]["chunk_id"] == "c1"
    assert hybrid_res[0]["method"] == "hybrid"
    assert "semantic_score" in hybrid_res[0]


def test_curfew_retrieval_quality():
    """Regression test: Verify curfew queries retrieve actual curfew rules, not meals."""
    from utils.vector_search import VectorSearchEngine

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    # 1. Semantic search tests for English, Hindi, Telugu, Paraphrase
    sem_curfew_queries = [
        ("English", "What time is the hostel curfew?"),
        ("Hindi", "छात्रावास का कर्फ्यू कितने बजे है?"),
        ("Telugu", "హాస్టల్ రాత్రి కర్ఫ్యూ సమయం ఎన్ని గంటలకు?"),
        ("English paraphrase", "When must students return to the hostel at night?")
    ]

    for lang, q in sem_curfew_queries:
        res = v_engine.search(q, top_k=1)
        assert len(res) >= 1, f"No semantic result for {lang}"
        text = res[0]["chunk"]["text"].lower()
        assert "curfew" in text or "కర్ఫ్యూ" in text, f"{lang} top semantic result missing curfew: {text[:100]}"
        assert "breakfast" not in text and "dining hall" not in text, f"{lang} wrongly returned dining: {text[:100]}"

    # 2. Hybrid search tests: All 5 queries (including Punjabi) must retrieve Curfew as #1
    all_curfew = sem_curfew_queries + [("Punjabi", "ਹੋਸਟਲ ਦਾ ਰਾਤ ਦਾ ਕਰਫਿਊ ਕਿੰਨੇ ਵਜੇ ਹੈ?")]
    for lang, q in all_curfew:
        bm25_res = bm25.search(q, top_k=5)
        hyb_res = v_engine.hybrid_search(q, bm25_res, top_k=1)
        assert len(hyb_res) >= 1, f"No hybrid result for {lang}"
        text = hyb_res[0]["chunk"]["text"].lower()
        assert "curfew" in text or "కర్ఫ్యూ" in text or "ਕਰਫਿਊ" in text, f"{lang} top hybrid result missing curfew: {text[:100]}"
        assert "breakfast" not in text and "dining hall" not in text, f"{lang} wrongly returned dining in hybrid: {text[:100]}"


def test_unanswerable_query_rejection():
    """Verify queries for nonexistent information have bounded, low hybrid scores."""
    from utils.vector_search import VectorSearchEngine

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    q = "What is the university principal's phone number?"
    bm25_res = bm25.search(q, top_k=3)
    hyb_res = v_engine.hybrid_search(q, bm25_res, top_k=3)

    # Score must be low (< 0.35) since principal's phone number does not exist
    if hyb_res:
        assert hyb_res[0]["score"] < 0.35, f"Unanswerable query got high score: {hyb_res[0]['score']}"


# ==============================================================================
# MILESTONE 3 TESTS: Grounded Answer Generation, Citations & Multilingual RAG
# ==============================================================================

def test_milestone3_citation_formatting_and_metadata():
    """Verify citation formatting adheres to [filename, Page X] and [filename, Section X.Y]."""
    from utils.answer_generator import format_citation

    # PDF Citation test
    chunk_pdf = {
        "source": "sample_hostel_policy.pdf",
        "page": 1,
        "text": "Hostel curfew is strictly 9:30 PM on weekdays."
    }
    cit_pdf = format_citation(chunk_pdf)
    assert cit_pdf == "[sample_hostel_policy.pdf, Page 1]", f"Unexpected PDF citation: {cit_pdf}"

    # TXT Section Citation test (Section 1.1)
    chunk_txt_sec = {
        "source": "sample_university_policy.txt",
        "page": 1,
        "text": "SECTION 1: ATTENDANCE POLICY\n1.1 Minimum Attendance Requirement:\nStudents must have 75% attendance"
    }
    cit_txt = format_citation(chunk_txt_sec)
    assert cit_txt == "[sample_university_policy.txt, Section 1.1]", f"Unexpected TXT citation: {cit_txt}"

    # TXT Section Citation test (Section 2.1)
    chunk_txt_sec2 = {
        "source": "sample_university_policy.txt",
        "page": 1,
        "text": "SECTION 2: HOSTEL ACCOMMODATION\n2.1 Hostel Curfew and Night Entry Timings:\nGates close at 9:30 PM"
    }
    cit_txt2 = format_citation(chunk_txt_sec2)
    assert cit_txt2 == "[sample_university_policy.txt, Section 2.1]", f"Unexpected TXT citation: {cit_txt2}"


def test_milestone3_grounded_answer_generation():
    """Verify answers are strictly grounded in retrieved passages with supporting facts and citations."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    # 1. Test Attendance Answer
    q_att = "What is the minimum attendance required for exams?"
    b_att = bm25.search(q_att, top_k=5)
    h_att = v_engine.hybrid_search(q_att, b_att, top_k=3)
    res_att = generator.generate(q_att, h_att)

    assert res_att["status"] == "supported"
    assert "75%" in res_att["answer"], "Answer missing critical 75% attendance fact"
    assert "[sample_university_policy.txt, Section 1.1]" in res_att["answer"]
    assert len(res_att["supporting_passage"]) > 20

    # 2. Test Hostel Curfew Answer
    q_curfew = "What are the hostel curfew timings on weekdays?"
    b_curfew = bm25.search(q_curfew, top_k=5)
    h_curfew = v_engine.hybrid_search(q_curfew, b_curfew, top_k=3)
    res_curfew = generator.generate(q_curfew, h_curfew)

    assert res_curfew["status"] == "supported"
    assert "9:30 PM" in res_curfew["answer"], "Answer missing curfew timing 9:30 PM"
    assert "sample_hostel_policy.pdf" in res_curfew["primary_citation"] or "sample_university_policy.txt" in res_curfew["primary_citation"]

    # 3. Test Merit Scholarship Paraphrase
    q_scholar = "Can top scoring students get a discount on tuition fees?"
    b_scholar = bm25.search(q_scholar, top_k=5)
    h_scholar = v_engine.hybrid_search(q_scholar, b_scholar, top_k=3)
    res_scholar = generator.generate(q_scholar, h_scholar)

    assert res_scholar["status"] == "supported"
    assert any(w in res_scholar["answer"].lower() for w in ["tuition", "fee", "40%", "50%", "financial assistance", "scholarship", "waiver"])


def test_milestone3_unsupported_questions_safety():
    """Verify unsupported questions (e.g. principal's phone number) are safely refused without hallucinating."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    # 1. Principal's phone number test
    q_phone = "What is the university principal's phone number?"
    b_phone = bm25.search(q_phone, top_k=3)
    h_phone = v_engine.hybrid_search(q_phone, b_phone, top_k=3)
    res_phone = generator.generate(q_phone, h_phone)

    assert res_phone["status"] == "unsupported", "Expected unsupported status for nonexistent principal phone number"
    assert "Information Not Available" in res_phone["answer"] or "do not contain information" in res_phone["answer"]
    assert "phone" not in res_phone["primary_citation"].lower()
    # Ensure no phone number pattern is hallucinated
    assert not re.search(r'\+?\d{10,12}', res_phone["answer"]), "Model hallucinated a phone number!"

    # 2. Non-existent flight training course
    q_flight = "How much is the flight training fee at the university?"
    b_flight = bm25.search(q_flight, top_k=3)
    h_flight = v_engine.hybrid_search(q_flight, b_flight, top_k=3)
    res_flight = generator.generate(q_flight, h_flight)

    assert res_flight["status"] == "unsupported"


def test_milestone3_multilingual_answers():
    """Verify multilingual questions in Hindi, Telugu, and Punjabi receive language-tailored grounded answers."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator, detect_language

    # Verify language detection
    assert detect_language("छात्रावास का समय क्या है?")["code"] == "hi"
    assert detect_language("హాస్టల్ రాత్రి కర్ఫ్యూ సమయం ఎన్ని గంటలకు?")["code"] == "te"
    assert detect_language("ਹੋਸਟਲ ਦਾ ਰਾਤ ਦਾ ਕਰਫਿਊ ਕਿੰਨੇ ਵਜੇ ਹੈ?")["code"] == "pa"

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    # 1. Telugu Hostel Curfew Query
    q_te = "హాస్టల్ రాత్రి కర్ఫ్యూ సమయం ఎన్ని గంటలకు?"
    b_te = bm25.search(q_te, top_k=5)
    h_te = v_engine.hybrid_search(q_te, b_te, top_k=3)
    res_te = generator.generate(q_te, h_te)

    assert res_te["status"] == "supported"
    assert "9:30 PM" in res_te["answer"]
    assert res_te["language"] == "Telugu (తెలుగు)"
    assert "విశ్వవిద్యాలయ" in res_te["answer"] or "ఆధార పత్రం" in res_te["answer"]

    # 2. Hindi Hostel Curfew Query
    q_hi = "छात्रावास का कर्फ्यू कितने बजे है?"
    b_hi = bm25.search(q_hi, top_k=5)
    h_hi = v_engine.hybrid_search(q_hi, b_hi, top_k=3)
    res_hi = generator.generate(q_hi, h_hi)

    assert res_hi["status"] == "supported"
    assert "9:30 PM" in res_hi["answer"]
    assert res_hi["language"] == "Hindi (हिंदी)"
    assert "विश्वविद्यालय" in res_hi["answer"] or "पुष्टि संदर्भ" in res_hi["answer"]

    # 3. Punjabi Library Fine Query
    q_pa = "ਲਾਇਬ੍ਰੇਰੀ ਦੀਆਂ ਕਿਤਾਬਾਂ ਦੇਰ ਨਾਲ ਦੇਣ 'ਤੇ ਕੀ ਜੁਰਮਾਨਾ ਲੱਗਦਾ ਹੈ?"
    b_pa = bm25.search(q_pa, top_k=5)
    h_pa = v_engine.hybrid_search(q_pa, b_pa, top_k=3)
    res_pa = generator.generate(q_pa, h_pa)

    assert res_pa["status"] == "supported"
    assert res_pa["language"] == "Punjabi (ਪੰਜਾਬੀ)"
    assert "INR 5" in res_pa["answer"] or "ਜੁਰਮਾਨਾ" in res_pa["answer"] or "library" in res_pa["answer"].lower()


def test_milestone3_attendance_and_curfew_citation_regression():
    """Regression test: Verify Hindi attendance query returns Section 1.1 (never 1.2) and curfew returns Section 2.1 or Page 1."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    # 1. Exact Hindi question from user: परीक्षा देने के लिए न्यूनतम उपस्थिति कितनी होनी चाहिए?
    q_hi = "परीक्षा देने के लिए न्यूनतम उपस्थिति कितनी होनी चाहिए?"
    b_hi = bm25.search(q_hi, top_k=5)
    h_hi = v_engine.hybrid_search(q_hi, b_hi, top_k=3)
    res_hi = generator.generate(q_hi, h_hi)

    assert res_hi["status"] == "supported", f"Expected supported status for Hindi attendance, got: {res_hi['status']}"
    assert "75%" in res_hi["answer"], "Missing 75% theory attendance fact in Hindi answer"
    assert "80%" in res_hi["answer"], "Missing 80% lab attendance fact in Hindi answer"
    
    # CRITICAL REGRESSION: Citation MUST be Section 1.1, NEVER Section 1.2!
    assert res_hi["primary_citation"] == "[sample_university_policy.txt, Section 1.1]", (
        f"Incorrect citation for minimum attendance! Expected Section 1.1, got: {res_hi['primary_citation']}"
    )
    assert "Section 1.2" not in res_hi["primary_citation"], "Regression: Condonation Section 1.2 wrongly cited for minimum attendance"
    assert "विश्वविद्यालय की आधिकारिक नीतियां अंग्रेजी में प्रकाशित हैं" in res_hi["answer"]

    # 2. English Attendance Query
    q_en = "What is the minimum attendance required for exams?"
    b_en = bm25.search(q_en, top_k=5)
    h_en = v_engine.hybrid_search(q_en, b_en, top_k=3)
    res_en = generator.generate(q_en, h_en)

    assert res_en["primary_citation"] == "[sample_university_policy.txt, Section 1.1]"
    assert "75%" in res_en["answer"]

    # 3. Telugu Hostel Curfew Query
    q_te = "హాస్టల్ రాత్రి కర్ఫ్యూ సమయం ఎన్ని గంటలకు?"
    b_te = bm25.search(q_te, top_k=5)
    h_te = v_engine.hybrid_search(q_te, b_te, top_k=3)
    res_te = generator.generate(q_te, h_te)

    assert "9:30 PM" in res_te["answer"]
    assert res_te["primary_citation"] in ["[sample_hostel_policy.pdf, Page 1]", "[sample_university_policy.txt, Section 2.1]"]
    assert "Section 2.3" not in res_te["primary_citation"], "Regression: Mess/dining section 2.3 wrongly cited for curfew"


def test_m31_telugu_passport_validity():
    """Milestone 3.1: Semester Abroad passport validity query in Telugu must retrieve Page 36, cite 6 months, and reject Page 41."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    q_te = "సెమిస్టర్ అబ్రాడ్ కోసం పాస్‌పోర్ట్ చెల్లుబాటు ఎంత కాలం ఉండాలి?"
    
    # Dual-path retrieval with auto query expansion
    bm25_res = bm25.search(q_te, top_k=5)
    hybrid_res = v_engine.hybrid_search(q_te, bm25_res, top_k=5)

    assert len(hybrid_res) > 0, "No search results returned for Telugu passport query"

    # Verify top result is from Academic Benefits.pdf, Page 36 (never Page 41)
    top_chunk = hybrid_res[0]["chunk"]
    assert top_chunk["source"] == "Academic Benefits.pdf", f"Expected Academic Benefits.pdf, got: {top_chunk['source']}"
    assert top_chunk["page"] == 36, f"Expected logical printed Page 36, got Page {top_chunk['page']}"
    assert "passport" in top_chunk["text"].lower(), "Missing passport keyword in top chunk"
    assert "six months" in top_chunk["text"].lower() or "6 months" in top_chunk["text"].lower(), "Missing six months requirement in top chunk"

    # Guardrail: Page 41 must never be accepted as the primary supporting evidence
    assert top_chunk["page"] != 41, "Page 41 social media passage wrongly retrieved as top result!"

    # Grounded answer synthesis check
    res = generator.generate(q_te, hybrid_res)
    assert res["status"] == "supported", f"Expected supported status, got: {res['status']}"
    assert res["primary_citation"] == "[Academic Benefits.pdf, Page 36]", f"Incorrect citation: {res['primary_citation']}"
    assert any(term in res["answer"].lower() for term in ["six months", "6 months"]), "Answer missing 6 months passport validity rule"
    assert "passport" in res["answer"].lower(), "Answer missing passport fact"
    assert "Page 41" not in res["primary_citation"], "Page 41 cited for passport query!"


def test_m31_telugu_family_business_revenue():
    """Milestone 3.1: Family business revenue query in Telugu must retrieve Page 5, extract not considered rule, and reject Page 41."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    q_te = "కుటుంబ వ్యాపారం నుండి వచ్చే రాబడి పరిగణించబడుతుందా?"
    
    bm25_res = bm25.search(q_te, top_k=5)
    hybrid_res = v_engine.hybrid_search(q_te, bm25_res, top_k=5)

    assert len(hybrid_res) > 0, "No search results returned for Telugu family business query"

    # Verify top result is from Academic Benefits.pdf, Page 5
    top_chunk = hybrid_res[0]["chunk"]
    assert top_chunk["source"] == "Academic Benefits.pdf", f"Expected Academic Benefits.pdf, got: {top_chunk['source']}"
    assert top_chunk["page"] == 5, f"Expected logical printed Page 5, got Page {top_chunk['page']}"
    assert "family-owned" in top_chunk["text"].lower() or "family enterprise" in top_chunk["text"].lower()
    assert "shall not be considered" in top_chunk["text"].lower() or "not be considered" in top_chunk["text"].lower()

    # Guardrail: Page 41 must never be accepted
    assert top_chunk["page"] != 41, "Page 41 social media passage wrongly retrieved as top result!"

    # Grounded answer synthesis check
    res = generator.generate(q_te, hybrid_res)
    assert res["status"] == "supported", f"Expected supported status, got: {res['status']}"
    assert res["primary_citation"] == "[Academic Benefits.pdf, Page 5]", f"Incorrect citation: {res['primary_citation']}"
    assert "shall not be considered" in res["answer"].lower() or "not be considered" in res["answer"].lower()
    # Explicit regression assertion: returned evidence actually mentions family-owned businesses or family enterprises
    assert "family-owned" in res["supporting_passage"].lower() or "family enterprise" in res["supporting_passage"].lower(), \
        "Returned evidence does not explicitly mention family-owned businesses or family enterprises!"
    assert "Page 41" not in res["primary_citation"], "Page 41 cited for family business query!"


def test_m31_telugu_attendance_cgpa_requirement():
    """Milestone 3.1: Attendance benefit and 7.5 CGPA query in Telugu must retrieve both facts and cite supporting page."""
    from utils.vector_search import VectorSearchEngine
    from utils.answer_generator import AnswerGenerator

    samples_dir = get_legacy_samples_dir()
    all_chunks = []
    for file_path in sorted(samples_dir.glob("*.*")):
        if file_path.suffix.lower() in [".txt", ".pdf"]:
            with open(file_path, "rb") as f:
                content = f.read()
            chunks = process_document(content, file_path.name)
            all_chunks.extend(chunks)

    bm25 = BM25SearchEngine()
    bm25.index_chunks(all_chunks)

    v_engine = VectorSearchEngine()
    v_engine.index_chunks(all_chunks)

    generator = AnswerGenerator(confidence_threshold=0.25)

    q_te = "10% హాజరు ప్రయోజనం కోసం 7.5 CGPA మరియు అర్హత ఏమిటి?"
    
    bm25_res = bm25.search(q_te, top_k=5)
    hybrid_res = v_engine.hybrid_search(q_te, bm25_res, top_k=5)

    assert len(hybrid_res) > 0, "No search results returned for Telugu attendance benefit query"

    # Verify answer contains BOTH the 10% benefit and the 7.5 CGPA condition
    res = generator.generate(q_te, hybrid_res)
    assert res["status"] == "supported", f"Expected supported status, got: {res['status']}"
    assert "7.5" in res["answer"], "Missing 7.5 CGPA condition in generated answer"
    assert "10%" in res["answer"] or "10 %" in res["answer"], "Missing 10% attendance benefit in generated answer"
    assert "Academic Benefits.pdf" in res["primary_citation"]
    assert "Page 30" in res["primary_citation"] or "Page 32" in res["primary_citation"]


def test_m31_unsupported_refusal_and_insufficient_evidence():
    """Milestone 3.1: Unanswerable query in Telugu is refused, and incomplete evidence returns insufficient_evidence."""
    from utils.answer_generator import AnswerGenerator

    generator = AnswerGenerator(confidence_threshold=0.25)

    # 1. Unanswerable question: Principal's personal phone number in Telugu
    q_unans = "ప్రిన్సిపాల్ గారి వ్యక్తిగత ఫోన్ నంబర్ ఎంత?"
    res_unans = generator.generate(q_unans, [])
    assert res_unans["status"] in ["unsupported", "insufficient_evidence"], f"Expected refusal, got: {res_unans['status']}"
    assert res_unans["status"] != "supported", "Refusal query was marked as supported!"
    assert "None" in res_unans["primary_citation"] or "Insufficient" in res_unans["primary_citation"] or "Information Absent" in res_unans["primary_citation"]

    # 2. Incomplete / Irrelevant evidence simulation: Page 41 social media passed for passport query
    fake_page41_chunk = {
        "chunk_id": "Academic_Benefits_pdf_p41_c1",
        "source": "Academic Benefits.pdf",
        "page": 41,
        "phys_page": 43,
        "text": "Apex Global University Official Social Media Channels. Follow us on Instagram, Twitter, Facebook, and LinkedIn. Post queries on WhatsApp student group."
    }
    fake_results = [{"chunk": fake_page41_chunk, "score": 0.35}]
    
    q_pass = "సెమిస్టర్ అబ్రాడ్ కోసం పాస్‌పోర్ట్ చెల్లుబాటు ఎంత కాలం ఉండాలి?"
    res_guard = generator.generate(q_pass, fake_results)
    assert res_guard["status"] == "insufficient_evidence", f"Expected insufficient_evidence, got: {res_guard['status']}"
    assert res_guard["status"] != "supported", "Irrelevant social media chunk was wrongly verified!"
    assert "Insufficient evidence" in res_guard["answer"] or "సేకరించిన పత్రాలలో తగిన ఆధారాలు లేవు" in res_guard["answer"]
    assert "Page 41" not in res_guard["primary_citation"], "Page 41 cited as verified evidence!"


def test_gemini_api_integration():
    """Verify Gemini API integration: prompt building, error classification, and graceful local fallback."""
    from utils.answer_generator import call_gemini_api, detect_language, AnswerGenerator

    q = "What is the minimum attendance required for semester exams?"
    fake_results = [{
        "chunk": {
            "source": "sample_university_policy.txt",
            "page": 1,
            "text": "Students must maintain at least 75% attendance in each course to be eligible to appear for the end-semester examinations."
        },
        "score": 0.85
    }]
    lang_info = detect_language(q)

    # 1. Test with empty API key: AnswerGenerator gracefully defaults to local extractive
    local_gen = AnswerGenerator(confidence_threshold=0.25, api_key="")
    res_local = local_gen.generate(q, fake_results)
    assert res_local["status"] == "supported"
    assert "75%" in res_local["answer"]
    assert "Extractive" in res_local["model"]

    # 2. Test call_gemini_api handles invalid key without raising an uncaught exception
    ans, model, err = call_gemini_api(q, fake_results, "INVALID_TEST_KEY_12345", lang_info)
    assert ans is None, "Expected invalid API key to return None"
    assert err is not None, "Expected an error message for invalid key"
    assert any(k in err.lower() for k in ["invalid", "expired", "http", "network", "connection", "ssl", "failed", "fallback"])

    # 3. Test with active environment key if available (verifies no crash and structured response)
    try:
        from dotenv import dotenv_values
        env_dict = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
        env_key = env_dict.get("GEMINI_API_KEY", "")
    except Exception:
        env_key = ""

    if env_key:
        ans_env, model_env, err_env = call_gemini_api(q, fake_results, env_key, lang_info)
        if ans_env:
            assert len(ans_env) > 5
            assert "Gemini" in (model_env or "")
        else:
            assert err_env is not None
            assert any(k in err_env.lower() for k in ["traffic", "fallback", "unavailable", "rate", "network"])


if __name__ == "__main__":
    tests = [
        ("Multilingual & Unicode Tokenization", test_tokenize),
        ("Plain Text (.txt) Extraction & Chunking", test_txt_extraction_and_chunking),
        ("PDF Multi-Page Extraction & Page Preservation", test_pdf_extraction),
        ("BM25 In-Memory Search & Scoring", test_bm25_search_scoring),
        ("Sample Documents Processing & Multi-Doc Search", test_sample_documents_indexing),
        ("Streamlit App End-to-End Simulation", test_streamlit_app_flow),
        ("Multilingual & Paraphrase Semantic Vector Search", test_multilingual_and_paraphrase_semantic_search),
        ("Hybrid (Semantic + Keyword) Search Fusion", test_hybrid_search_combination),
        ("Curfew vs Dining Retrieval Quality (Regression)", test_curfew_retrieval_quality),
        ("Unanswerable Query Score Bounding (Regression)", test_unanswerable_query_rejection),
        ("M3: Verifiable Citation Formatting & Metadata", test_milestone3_citation_formatting_and_metadata),
        ("M3: Grounded Answer Generation & Fact Extraction", test_milestone3_grounded_answer_generation),
        ("M3: Safe Handling of Unsupported Queries (Zero-Hallucination)", test_milestone3_unsupported_questions_safety),
        ("M3: Multilingual Query Grounding (Hindi, Telugu, Punjabi)", test_milestone3_multilingual_answers),
        ("M3: Attendance (Section 1.1) & Curfew Citation Regression", test_milestone3_attendance_and_curfew_citation_regression),
        ("M3.1: Telugu Passport Validity (Page 36 & 6-Month Rule)", test_m31_telugu_passport_validity),
        ("M3.1: Telugu Family Business Revenue (Page 5 & Exclusion Rule)", test_m31_telugu_family_business_revenue),
        ("M3.1: Telugu Attendance Benefit (10%) & 7.5 CGPA Requirement", test_m31_telugu_attendance_cgpa_requirement),
        ("M3.1: Refusal of Unanswerable & Insufficient Evidence Guardrails", test_m31_unsupported_refusal_and_insufficient_evidence),
        ("Gemini API Integration & Model Failover", test_gemini_api_integration),
    ]

    print("=" * 65)
    print("RUNNING MULTILINGUAL HELPDESK TEST SUITE (MILESTONE 1, 2, 3 & 3.1)")
    print("=" * 65)

    all_passed = True
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {test_name}")
        except Exception as e:
            all_passed = False
            print(f"[FAIL] {test_name}: {e}")

    print("=" * 65)
    if all_passed:
        print(f"RESULT: ALL {len(tests)} TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME TESTS FAILED.")
    print("=" * 65)

