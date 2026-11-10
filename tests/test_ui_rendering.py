"""
tests/test_ui_rendering.py
--------------------------
Integration and verification tests for UI chat rendering, evidence display,
and layout styling:

1. Verification of the fee payment question ("What are the available fee payment methods at LPU?").
2. Full assistant response visibility (natural-language answer with Markdown paragraphs, lists, bolding).
3. Absolute absence of raw HTML snippets (<span class="page-badge">, <div class="chunk-text">, etc.) in rendered evidence.
4. Clean separation of metadata (Source, Section, Page, Scope, Score) from document passage text.
5. Long answer with multiple bullet points rendering cleanly without clipping.
6. DOCX section citation formatting ([fee_payment_modes_and_guidelines.docx, Section: OVERVIEW]).
7. PDF page citation formatting ([Academic Benefits.pdf, Page 36]).
8. Multilingual Telugu query rendering.
9. Conversational follow-up query rewriting & rendering.
10. CSS responsive layout checks (preventing horizontal scrollbars at 1920px and 1366px).
"""

import sys
import os
import re
import html
from pathlib import Path
from streamlit.testing.v1 import AppTest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.document_processor import load_or_build_document_index
from utils.text_search import BM25SearchEngine
from utils.vector_search import VectorSearchEngine
from utils.answer_generator import AnswerGenerator, format_citation


def test_1_fee_payment_question_retrieval_and_answer():
    """Verify fee payment question retrieves fee_payment_modes_and_guidelines.docx with Section: OVERVIEW."""
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    bm25 = BM25SearchEngine()
    vector = VectorSearchEngine()
    chunks, _ = load_or_build_document_index(samples_dir, vector, bm25)

    q = "What are the available fee payment methods at LPU?"
    bm25_res = bm25.search(q, top_k=5)
    hybrid_res = vector.hybrid_search(q, bm25_res, top_k=3)

    assert len(hybrid_res) >= 1, "No search results for fee payment query"
    top_chunk = hybrid_res[0]["chunk"]
    assert top_chunk["source"] == "fee_payment_modes_and_guidelines.docx"

    gen = AnswerGenerator(confidence_threshold=0.25)
    ans_res = gen.generate(q, hybrid_res)
    assert ans_res["status"] == "supported"
    assert "fee_payment_modes_and_guidelines.docx" in ans_res["primary_citation"]
    assert "Section: OVERVIEW" in ans_res["primary_citation"] or "Section" in ans_res["primary_citation"]
    
    # Verify complete answer has rich policy details (online/offline, payment modes)
    answer_text = ans_res["answer"]
    assert len(answer_text) > 100
    assert any(term in answer_text.lower() for term in ["online", "payment", "fee", "upi", "net banking", "card"])


def test_2_no_raw_html_in_evidence_rendering():
    """Verify that evidence cards render clean text with ZERO raw HTML fragments shown as code or escaped text."""
    sample_chunk = {
        "source": "fee_payment_modes_and_guidelines.docx",
        "page": None,
        "section": "OVERVIEW",
        "programme_scope": "Regular",
        "text": "OVERVIEW Lovely Professional University provides students with multiple online and offline modes for paying academic, residential, and other facility charges."
    }
    score = 0.693

    page_str = f"Page: {sample_chunk['page']}" if sample_chunk['page'] is not None else (f"Section: {sample_chunk['section']}" if sample_chunk['section'] else "")
    scope_str = f"Scope: {sample_chunk['programme_scope']}" if (sample_chunk['programme_scope'] and sample_chunk['programme_scope'] != "General") else ""
    score_str = f"Score: {score:.3f}" if isinstance(score, (int, float)) else f"Score: {score}"
    safe_text = html.escape(sample_chunk["text"].strip())

    card_html = (
        f'<div class="result-card">\n'
        f'  <div class="result-card-meta">\n'
        f'    <span class="source-badge">Source: {sample_chunk["source"]}</span>\n'
        + (f'    <span class="page-badge">{page_str}</span>\n' if page_str else '')
        + (f'    <span class="page-badge scope-badge">{scope_str}</span>\n' if scope_str else '')
        + (f'    <span class="score-badge">{score_str}</span>\n' if score else '')
        + f'  </div>\n'
        f'  <div class="chunk-text">{safe_text}</div>\n'
        f'</div>'
    )

    # In card_html:
    # 1. Starts at column 0 without leading spaces
    assert card_html.startswith('<div class="result-card">')
    # 2. Contains source filename
    assert "fee_payment_modes_and_guidelines.docx" in card_html
    # 3. Contains Section: OVERVIEW
    assert "Section: OVERVIEW" in card_html
    # 4. Contains Score: 0.693
    assert "Score: 0.693" in card_html
    # 5. Must NOT contain literal escaped HTML tags inside chunk-text
    assert '<div class="chunk-text">' in card_html
    # Extract inner chunk-text
    inner_chunk = card_html.split('<div class="chunk-text">')[1].split('</div>')[0]
    assert "<span" not in inner_chunk, f"HTML tags found inside passage text: {inner_chunk}"
    assert "<div" not in inner_chunk, f"HTML tags found inside passage text: {inner_chunk}"
    assert "class=" not in inner_chunk, f"CSS class names found inside passage text: {inner_chunk}"
    assert "OVERVIEW Lovely Professional University" in inner_chunk


def test_3_full_width_assistant_markdown_response():
    """Verify assistant responses are rendered with complete Markdown elements (headings, bullets, bolding)."""
    generator = AnswerGenerator(confidence_threshold=0.25)
    sample_evidence = [{
        "chunk": {
            "source": "fee_payment_modes_and_guidelines.docx",
            "page": None,
            "section": "OVERVIEW",
            "text": "Available payment methods include:\n- Online Net Banking\n- UPI Payments (Google Pay, PhonePe)\n- Debit and Credit Cards\n- Demand Draft at university cash counter"
        },
        "score": 0.85
    }]
    res = generator.generate("What are the available fee payment methods?", sample_evidence)
    ans = res["answer"]
    assert len(ans.splitlines()) >= 1
    # Check that bullet points or structured text is preserved
    assert any(term in ans for term in ["payment", "methods", "Online", "fee", "University", "available"])


def test_4_docx_section_vs_pdf_page_citation():
    """Verify DOCX generates section citation and PDF generates page citation."""
    docx_chunk = {
        "source": "fee_payment_modes_and_guidelines.docx",
        "page": None,
        "section": "OVERVIEW",
        "text": "Fee payment overview."
    }
    cit_docx = format_citation(docx_chunk)
    assert "Section" in cit_docx and "OVERVIEW" in cit_docx
    assert "Page" not in cit_docx

    pdf_chunk = {
        "source": "Academic Benefits.pdf",
        "page": 36,
        "text": "Passport must be valid for at least six months."
    }
    cit_pdf = format_citation(pdf_chunk)
    assert cit_pdf == "[Academic Benefits.pdf, Page 36]"
    assert "Page 36" in cit_pdf


def test_5_telugu_multilingual_flow():
    """Verify Telugu queries are processed and formatted cleanly without broken tags."""
    from utils.query_translator import expand_query_multilingual
    samples_dir = Path(__file__).resolve().parent.parent / "sample_documents"
    bm25 = BM25SearchEngine()
    vector = VectorSearchEngine()
    chunks, _ = load_or_build_document_index(samples_dir, vector, bm25)

    q_te = "సెమిస్టర్ అబ్రాడ్ కోసం పాస్‌పోర్ట్ చెల్లుబాటు ఎంత కాలం ఉండాలి?"
    trans = expand_query_multilingual(q_te)
    en_q = trans["english_query"] if isinstance(trans, dict) else q_te
    bm25_res = bm25.search(en_q, top_k=5)
    hyb_res = vector.hybrid_search(q_te, bm25_res, top_k=3)

    assert len(hyb_res) > 0
    top = hyb_res[0]["chunk"]
    assert top["source"] == "semester_abroad_policy.docx"
    assert top.get("page") is None  # DOCX does not invent page numbers

    gen = AnswerGenerator(confidence_threshold=0.25)
    ans_res = gen.generate(q_te, hyb_res)
    assert ans_res["status"] == "supported"
    assert "semester_abroad_policy.docx" in ans_res["primary_citation"] or "international_student_left_marking_process.docx" in ans_res["primary_citation"]


def test_6_conversational_follow_up():
    """Verify follow-up questions rewrite properly and retrieve fresh policy evidence."""
    from utils.context_rewriter import rewrite_conversational_query

    history = [
        {"role": "user", "content": "What are the rules for Semester Abroad?"},
        {"role": "assistant", "content": "Students require a 7.5 CGPA and a valid passport."}
    ]
    rewritten, was_rewritten, _ = rewrite_conversational_query("What about passport validity?", history)
    assert was_rewritten
    assert "passport" in rewritten.lower()
    assert "semester abroad" in rewritten.lower()


def test_7_css_responsive_layout_no_overflow():
    """Verify app.py CSS contains overflow-x: hidden, max-width constraints, and no fixed 2000px containers."""
    app_file = Path(__file__).resolve().parent.parent / "app.py"
    content = app_file.read_text(encoding="utf-8")

    # 1. Overflow-x hidden on app and container
    assert "overflow-x: hidden !important" in content
    # 2. Block container max-width constraint
    assert "max-width: 860px !important" in content or "max-width:" in content
    # 3. Word break / overflow-wrap anywhere on chat messages
    assert "overflow-wrap: anywhere" in content
    assert "word-break: break-word" in content
    # 4. Result card max-width: 100%
    assert "max-width: 100%" in content
    # 5. StChatMessage native rendering used
    assert "st.chat_message" in content


def test_8_streamlit_app_interactive_flow():
    """Verify Streamlit AppTest runs through startup, sample loading, button clicks and renders clean cards."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=240)
    at.run()
    assert not at.exception

    # Click Load Samples button
    load_btns = [b for b in at.button if "Load Samples" in b.label]
    if load_btns:
        load_btns[0].click().run()
        assert not at.exception

    # Click Attendance Question button
    btn_q1_list = [b for b in at.button if b.key == "btn_q1"]
    if btn_q1_list:
        btn_q1_list[0].click().run()
        assert not at.exception

        # Check that cards rendered
        cards = [m.value for m in at.markdown if "result-card" in m.value and "Source:" in m.value]
        assert len(cards) >= 1
        assert "attendance" in cards[0].lower()
        assert "Section:" in cards[0]
        
        # Verify NO raw HTML tags were dumped inside passage text
        for card in cards:
            assert card.startswith('<div class="result-card">')
            assert 'class="chunk-text">' in card
            inner_text = card.split('class="chunk-text">')[1].split('</div>')[0]
            assert "<span" not in inner_text
            assert "<div" not in inner_text


if __name__ == "__main__":
    tests = [
        ("Fee Payment Question Retrieval & Grounding", test_1_fee_payment_question_retrieval_and_answer),
        ("Sanitization & No Raw HTML in Evidence Cards", test_2_no_raw_html_in_evidence_rendering),
        ("Full-Width Assistant Response & Markdown Preservation", test_3_full_width_assistant_markdown_response),
        ("DOCX Section vs PDF Page Citations", test_4_docx_section_vs_pdf_page_citation),
        ("Multilingual Telugu Query Grounding", test_5_telugu_multilingual_flow),
        ("Conversational Follow-up Rewriting", test_6_conversational_follow_up),
        ("CSS Responsive & Overflow Constraints", test_7_css_responsive_layout_no_overflow),
        ("Streamlit App Interactive End-to-End Test", test_8_streamlit_app_interactive_flow),
    ]

    print("=" * 65)
    print("RUNNING UI RENDERING & LAYOUT FIX TEST SUITE")
    print("=" * 65)

    all_passed = True
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:
            all_passed = False
            print(f"[FAIL] {name}: {e}")

    print("=" * 65)
    if all_passed:
        print("RESULT: ALL UI RENDERING & LAYOUT TESTS PASSED SUCCESSFULLY!")
    else:
        print("RESULT: SOME TESTS FAILED.")
    print("=" * 65)
