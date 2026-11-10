"""
utils/context_rewriter.py
--------------------------
Conversational context tracking and query rewriting for follow-up questions.

Key Requirements:
1. Detect when a user question is an incomplete follow-up (e.g., "What about passport validity?",
   "And for second-year students?", "What is the deadline?", "Explain that in Telugu.").
2. Convert follow-up questions into clear standalone search queries by resolving context
   from previous user turns and discussion topics.
3. Preserve the student's original question separately.
4. Never add facts or eligibility criteria the student did not provide.
5. Do not use previous AI answers as official policy evidence; only use context to identify the topic.
6. Handle English, Hindi, Telugu, and Punjabi.
7. Treat queries that introduce a new topic as standalone (no false rewrites).
   Specifically: "What is the fee refund policy?" following a family business question must
   REMAIN standalone and NEVER be rewritten to include unrelated subjects.
8. Lightweight, fast local rewriting by default; optional Gemini API refinement when available.
"""

import re
import os
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Follow-Up Signal Detectors (English, Hindi, Telugu, Punjabi)
# ---------------------------------------------------------------------------

FOLLOW_UP_CONNECTIVES = [
    # English elliptical / connective starters
    r"^(?:and\s+(?:what|which|how|when|where|who|is|are|can|does|do|will)|what\s+about|how\s+about|and\s+for|what\s+if)\b",
    r"^(?:and\s+also|and\s+then|also\s+what\s+about|also\s+for|what\s+then)\b",
    r"^and\s+(?:the\s+)?(?:deadline|fees?|eligibility|requirements?|criteria|timing|dates?|process|concession|rule|same|cgpa)\b",
    # Hindi elliptical / connective starters
    r"^(?:और\s+इसके\s+अलावा|और\s+क्या|और\s+कब|और\s+कहाँ|और\s+कितना|और\s+किसके\s+लिए|और\s+अंतिम\s+तिथि)\b",
    r"^(?:इसके\s+बारे\s+में|उसी\s+नियम|वही\s+नियम)\b",
    # Telugu elliptical / connective starters
    r"^(?:మరియు\s+అర్హత|మరియు\s+చివరి\s+తేదీ|మరియు\s+ఎంత|మరి\s+ఎప్పుడు|మరి\s+ఎక్కడ|మరి\s+పాస్‌పోర్ట్|గురించి\s+ఏమిటి)\b",
    r"^(?:అదే\s+నిబంధన|అదే\s+రూల్|దీని\s+గురించి|దాని\s+గురించి)\b",
    # Punjabi elliptical / connective starters
    r"^(?:ਅਤੇ\s+ਆਖਰੀ\s+ਮਿਤੀ|ਅਤੇ\s+ਕਦੋਂ|ਅਤੇ\s+ਕਿੱਥੇ|ਅਤੇ\s+ਕਿੰਨਾ|ਅਤੇ\s+ਕਿਸ\s+ਲਈ|ਵੈਧਤਾ\s+ਬਾਰੇ)\b",
    r"^(?:ਇਸ\s+ਬਾਰੇ|ਉਹੀ\s+ਨਿਯਮ)\b",
]

PRONOUN_OR_ELLIPSIS_PATTERNS = [
    # Pronouns referring to prior entity
    r"\b(?:it|its|this|that|these|those|there|same rule|same policy|such students|such cases)\b",
    # Incomplete questions
    r"^(?:what about|how about|and for|for whom|what if|why|how so)\b",
    # Standalone queries without explicit topic noun
    r"^(?:what is the deadline\??|when is the deadline\??|what is the fee\??|how much is the fee\??)$",
    r"^(?:who is eligible\??|what are the criteria\??|what is the minimum cgpa\??)$",
    # Hindi pronouns / indicators
    r"\b(?:यह|वह|इसका|उसका|उसी|वही नियम|उन छात्रों)\b",
    r"(?:कितनी होनी चाहिए|अंतिम तिथि क्या है|वैधता कितनी|नियम क्या है|पात्रता क्या है)",
    # Telugu pronouns / indicators
    r"\b(?:ఇది|అది|దీని|దాని|అదే నిబంధన|అటువంటి విద్యార్థులు)\b",
    r"(?:ఎంత ఉండాలి|చెల్లుబాటు ఎంత|చివరి తేదీ ఏమిటి|నిబంధన ఏమిటి|అర్హత ఏమిటి|వర్తిస్తుందా)",
    # Punjabi pronouns / indicators
    r"\b(?:ਇਹ|ਉਹ|ਇਸਦਾ|ਉਸਦਾ|ਉਹੀ ਨਿਯਮ)\b",
    r"(?:ਕਿੰਨੀ ਹੋਣੀ ਚਾਹੀਦੀ|ਆਖਰੀ ਮਿਤੀ ਕੀ|ਵੈਧਤਾ ਕਿੰਨੀ)",
]

# Common core university topic entities to track across conversation
KNOWN_TOPIC_ENTITIES = [
    # International / Mobility
    (r"(?:\b(?:semester abroad|foreign study|international mobility|partner university|host university)\b|సెమిస్టర్ అబ్రాడ్|విదేశీ|विदेश|ਵਿਦੇਸ਼)", "Semester Abroad"),
    (r"(?:\b(?:passport|passports|visa|airfare)\b|పాస్‌పోర్ట్|పాస్\u200cపోర్ట్|పాస్|पासपोर्ट|ਪਾਸਪੋਰਟ)", "Passport and Travel"),
    (r"(?:\b(?:left marking|frro|international student|visa expiry)\b)", "International Student Compliance"),
    
    # Hostel / Residential
    (r"(?:\b(?:hostel curfew|night curfew|curfew|hostel entry|warden|in-time)\b|హాస్టల్ కర్ఫ్యూ|కర్ఫ్యూ|कर्फ्यू|ਕਰਫਿਊ)", "Hostel Curfew"),
    (r"(?:\b(?:hostel mess|dining|breakfast|lunch|dinner|food timings|vegetarian)\b|మెస్|భోజనం|भोजनालय)", "Hostel Mess Timings"),
    (r"(?:\b(?:night out|leave request|e-leave|overnight stay)\b|సెలవు|రాత్రి సెలవు|अवकाश)", "Hostel Night Leave"),
    (r"(?:\b(?:hostel|hostel accommodation|hostels)\b|హాస్టల్|छात्रावास|ਹੋਸਟਲ)", "Hostel Accommodation"),

    # Academics & Attendance
    (r"(?:\b(?:10% attendance benefit|attendance benefit|duty leave)\b|హాజరు ప్రయోజనం|उपस्थिति लाभ|ਹਾਜ਼ਰੀ ਲਾਭ)", "10% Attendance Benefit"),
    (r"(?:\b(?:attendance|75% attendance|shortage of attendance|condonation)\b|హాజరు|గైర్హాజరు|उपस्थिति|हाज़री|ਹਾਜ਼ਰੀ)", "Attendance Policy"),
    (r"(?:\b(?:late registration|attendance calculation rules|registration change)\b)", "Late Registration Policy"),
    (r"(?:\b(?:7\.5 cgpa|minimum cgpa|cgpa requirement|eligibility criteria)\b|కనీస అర్హత|సిజిపిఎ|అర్హత|पात्रता|सीजीपीए|ਯੋਗਤਾ)", "CGPA Eligibility"),
    (r"(?:\b(?:reappear|improvement exam|backlog|supplementary exam|re-evaluation|answer script|paper inspection)\b|పునఃపరిశీలన|పునర్మూల్యాంకనం|पुनर्मूल्यांकन)", "Reappear and Improvement Examination"),
    (r"(?:\b(?:patent|patents|patent granted|grade upgradation)\b|పేటెంట్|पेटेंट)", "Patent Grade Upgradation"),
    (r"(?:\b(?:nptel|mooc|moocs|proctored moocs|online certification)\b|ఎన్‌పిటిఇఎల్)", "NPTEL and MOOCs"),
    (r"(?:\b(?:hackathon|hackathons|technical competition|competitions)\b|హ్యాకథాన్)", "Hackathons and Competitions"),
    (r"(?:\b(?:odl|distance education|continuous assessment|ca assignment)\b)", "ODL Evaluation Mechanism"),
    (r"(?:\b(?:certificate|certificates|transcript|bonafide|migration certificate|character certificate)\b|సర్టిఫికెట్|प्रमाण पत्र|ਸਰਟੀਫਿਕੇਟ)", "Student Certificates"),

    # Finance & Revenue
    (r"(?:\b(?:family-owned business|family enterprise|family business|revenue generation)\b|కుటుంబ వ్యాపారం|వ్యాపార రాబడి|पारिवारिक व्यवसाय|ਪਰਿਵਾਰਕ ਕਾਰੋਬਾਰ)", "Family-Owned Business Revenue"),
    (r"(?:\b(?:fee refund|refund policy|refund of fee|refund of fees|fee refund policy|refund on withdrawal)\b|రిఫండ్|रिफंड|ਰਿਫੰਡ)", "Fee Refund Policy"),
    (r"(?:\b(?:merit scholarship|tuition fee waiver|scholarship)\b|స్కాలర్‌షిప్|ఫీజు రాయితీ|छात्रवृत्ति|ਵਜ਼ੀਫਾ)", "Merit Scholarship"),
    (r"(?:\b(?:financial aid|merit-cum-means|income limit)\b|ఆర్థిక సహాయం|ఆదాయ పరిమితి|आय सीमा)", "Financial Aid"),
    (r"(?:\b(?:library fine|late fee|book overdue|compounding fee)\b|లైబ్రరీ జరిమానా|లైబ్రరీ ఫైన్|विलंब शुल्क|ਦੇਰੀ ਜੁਰਮਾਨਾ)", "Library Overdue Fine"),
    (r"(?:\b(?:tuition fee|fee payment|fees|payment deadline|semester fee|examination fee)\b|ట్యూషన్ ఫీజు|ఫీజు చెల్లింపు|शिक्षण शुल्क|ਟਿਊਸ਼ਨ ਫੀਸ)", "Tuition Fee Payment"),

    # Careers
    (r"(?:\b(?:placement|placements|campus drive|career services|recruitment|pep fee|spc|placement coordinator)\b|ప్లేస్‌మెంట్|ఉద్యోగ నియామకం|प्लेसमेंट)", "Campus Placement"),
]


def extract_topic_from_text(text: str) -> Optional[str]:
    """Identifies the primary university policy topic from a question or answer."""
    if not text:
        return None
    for pattern, canonical_name in KNOWN_TOPIC_ENTITIES:
        if re.search(pattern, text, re.IGNORECASE):
            return canonical_name
    return None


def is_follow_up_question(query: str, history: List[Dict[str, str]]) -> Tuple[bool, Optional[str]]:
    """
    Determines if the student's question is an incomplete follow-up question.
    Returns (is_follow_up, reason).
    
    Guarantees:
    - Never flags standalone questions (e.g., 'What is the fee refund policy?') as follow-ups.
    - Accurately catches conversational follow-ups ('What about passport validity?',
      'And for second-year students?', 'Does the same rule apply to hostel students?').
    """
    if not history or not query:
        return False, None

    q_clean = query.strip()
    q_lower = q_clean.lower()
    words = q_lower.split()

    # Rule 1: Language switch / explanation follow-up (e.g. "Explain that in Telugu")
    if re.search(r"\b(?:explain that|explain in|tell me in|translate to|in telugu|in hindi|in punjabi)\b", q_lower):
        return True, "explanation_or_language_request"

    # Rule 2: Explicit connective follow-up prefix ("what about...", "and for...", "how about...")
    for pref in FOLLOW_UP_CONNECTIVES:
        if re.search(pref, q_lower):
            return True, "follow_up_connective"

    # Rule 3: Pronouns referring to prior entity without local antecedent
    pronoun_matches = [p for p in PRONOUN_OR_ELLIPSIS_PATTERNS if re.search(p, q_lower)]
    if pronoun_matches:
        # Check if the pronoun is genuinely anaphoric
        if any(w in words for w in ["it", "this", "that", "these", "those", "same"]):
            # If query has its own explicit policy noun phrase and no connective, verify standalone
            topic_in_q = extract_topic_from_text(q_lower)
            if not topic_in_q or "same" in words:
                return True, "pronoun_reference"
        else:
            return True, "elliptical_indicator"

    # Rule 4: Very short queries (<= 3 words) with incomplete question words
    if len(words) <= 3:
        if any(w in words for w in ["deadline", "deadline?", "fee", "fee?", "fees?", "eligibility?", "criteria?"]):
            return True, "short_elliptical_query"

    # Rule 5: Missing subject with question verb/clause (e.g., "Is it mandatory?", "Can they apply?")
    if re.match(r"^(?:is|are|can|does|do|will)\s+(?:it|they|such\s+students|anyone)\s+", q_lower):
        return True, "unbound_question_verb"

    # Rule 6: Aspect follow-up when active topic is broader
    # e.g., asking about passport validity after semester abroad without repeating semester abroad
    active_topic, _ = get_recent_conversation_topic(history)
    if active_topic:
        if active_topic == "Semester Abroad" and re.search(r"(?:passport|visa|పాస్‌పోర్ట్|పాస్\u200cపోర్ట్|పాస్|पासपोर्ट|ਪਾਸਪੋਰਟ)", q_lower):
            if "semester abroad" not in q_lower and "సెమిస్టర్" not in q_lower and "सेमेस्टर" not in q_lower:
                return True, "aspect_follow_up"

    return False, None


def get_recent_conversation_topic(history: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Scans recent conversation turns in reverse to find the active topic and last user query.
    Prioritizes user turns so the student's core inquiry is not masked by secondary details
    in an assistant's response.
    Returns (topic_name, last_user_query).
    """
    last_user_query = None
    user_topic = None
    assistant_topic = None

    for msg in reversed(history):
        role = msg.get("role", "")
        content = msg.get("content", "").strip()

        if role == "user":
            if not last_user_query:
                last_user_query = content
            if not user_topic:
                user_topic = extract_topic_from_text(content)
        elif role == "assistant":
            if not assistant_topic:
                assistant_topic = extract_topic_from_text(content)

    detected_topic = user_topic or assistant_topic
    return detected_topic, last_user_query


def rewrite_query_locally(query: str, history: List[Dict[str, str]]) -> Tuple[str, bool]:
    """
    Lightweight, fast, deterministic local query rewriter.
    Converts incomplete follow-ups into standalone search queries without any external API calls.
    Returns (rewritten_query, was_rewritten).
    """
    is_follow_up, reason = is_follow_up_question(query, history)
    if not is_follow_up or not history:
        return query, False

    active_topic, last_user_q = get_recent_conversation_topic(history)
    if not active_topic and not last_user_q:
        return query, False

    clean_q = query.strip()
    q_lower = clean_q.lower()

    # Case 1: "Explain that in Telugu" / "in Hindi"
    if re.search(r"\b(?:explain that|tell me|in telugu|in hindi|in punjabi|translate)\b", q_lower):
        target_topic = active_topic or last_user_q or ""
        return f"{target_topic} rules and policy details", True

    # Case 2: "What about passport validity?" / "What about X?"
    m_what_about = re.match(r"^(?:what about|how about|and about)\s+(.+?)(?:\?|$)", q_lower)
    if m_what_about:
        aspect = m_what_about.group(1).strip()
        if active_topic:
            return f"What is the {aspect} for {active_topic}?", True
        elif last_user_q:
            return f"What is the {aspect} regarding {last_user_q}?", True

    # Case 3: "And for second-year students?" / "And for X?"
    m_and_for = re.match(r"^(?:and\s+for|for|what\s+about\s+for)\s+(.+?)(?:\?|$)", q_lower)
    if m_and_for:
        group = m_and_for.group(1).strip()
        if active_topic:
            return f"What is the {active_topic} eligibility and rules for {group}?", True
        elif last_user_q:
            return f"{last_user_q} for {group}", True

    # Case 4: "What is the deadline?" / "When is the deadline?"
    if re.search(r"\bdeadline\b", q_lower):
        if active_topic:
            return f"What is the submission deadline for {active_topic}?", True

    # Case 5: "And what CGPA is required?" / "What CGPA?"
    if re.search(r"\bcgpa\b", q_lower):
        if active_topic:
            return f"What is the minimum CGPA requirement for {active_topic}?", True

    # Case 6: "Does the same rule apply to hostel students?"
    m_same_rule = re.search(r"\b(?:same rule|same policy|rules?)\s+apply\s+to\s+(.+?)(?:\?|$)", q_lower)
    if m_same_rule:
        target_group = m_same_rule.group(1).strip()
        if active_topic:
            return f"Does the {active_topic} policy apply to {target_group}?", True

    # Case 7: Pronoun reference "it", "this", "that"
    # e.g., "Is it mandatory?", "How to apply for it?", "Can I get this?"
    if re.search(r"\b(?:it|this|that)\b", q_lower):
        if active_topic:
            resolved = re.sub(r"\b(?:it|this|that)\b", active_topic, clean_q, flags=re.IGNORECASE)
            return resolved, True

    # Case 8: Multilingual follow-ups (Hindi, Telugu, Punjabi)
    # Hindi: "और अंतिम तिथि क्या है?" -> "[active_topic] अंतिम तिथि"
    if any(ch in clean_q for ch in ["और", "क्या", "कब", "कितना", "पासपोर्ट", "वैधता"]):
        if active_topic:
            return f"{clean_q} ({active_topic})", True

    # Telugu: "మరియు అర్హత ఏమిటి?" / "పాస్‌పోర్ట్ చెల్లుబాటు గురించి ఏమిటి?"
    if any(ch in clean_q for ch in ["మరియు", "గురించి", "ఎంత", "ఎప్పుడు", "అర్హత", "పాస్‌పోర్ట్", "చెల్లుబాటు", "పాస్"]):
        if active_topic:
            return f"{clean_q} ({active_topic})", True

    # Punjabi: "ਅਤੇ ਆਖਰੀ ਮਿਤੀ ਕੀ ਹੈ?"
    if any(ch in clean_q for ch in ["ਅਤੇ", "ਕੀ", "ਕਦੋਂ", "ਕਿੰਨਾ", "ਪਾਸਪੋਰਟ", "ਵੈਧਤਾ"]):
        if active_topic:
            return f"{clean_q} ({active_topic})", True

    return query, False


def rewrite_query_with_gemini(
    query: str,
    history: List[Dict[str, str]],
    api_key: str
) -> Optional[str]:
    """
    Optional Gemini API fallback for highly ambiguous conversational follow-ups.
    Enforces strict safety guardrails:
    - Never invents rules or answers.
    - Only disambiguates anaphora / ellipses into a clear standalone retrieval query.
    - Fast timeout (4 seconds) with automatic fallback.
    """
    if not api_key or not history:
        return None

    recent_turns = history[-4:]
    formatted_history = []
    for turn in recent_turns:
        role = turn.get("role", "user")
        text = turn.get("content", "").strip()
        formatted_history.append(f"{role.capitalize()}: {text}")

    prompt = (
        "You are an AI query-rewriting assistant for a university policy search engine.\n"
        "Your only job is to resolve conversational references (such as 'it', 'that', 'what about', 'and for') "
        "and convert the student's follow-up question into a complete, standalone search query.\n\n"
        "RULES:\n"
        "1. Output ONLY the rewritten standalone question. No intro, no quotes, no explanation.\n"
        "2. Never answer the question.\n"
        "3. Never add facts, rules, or requirements not mentioned in the question.\n"
        "4. If the question is already complete or introduces a new topic, output the original question unchanged.\n\n"
        f"Conversation History:\n" + "\n".join(formatted_history) + f"\n\nStudent Follow-Up: {query}\n"
        "Standalone Search Query:"
    )

    models_to_try = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash"]
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 60}
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cand = data.get("candidates", [])
                if cand:
                    parts = cand[0].get("content", {}).get("parts", [])
                    if parts and parts[0].get("text"):
                        rewritten = parts[0]["text"].strip()
                        # Clean any quotation marks or prefixes
                        rewritten = re.sub(r'^["\']|["\']$', '', rewritten).strip()
                        if rewritten and len(rewritten) > 3:
                            return rewritten
        except Exception:
            continue

    return None


def rewrite_conversational_query(
    query: str,
    history: List[Dict[str, str]],
    api_key: Optional[str] = None
) -> Tuple[str, bool, str]:
    """
    Main entry point for Context-Aware Conversational Query Rewriting.

    Args:
        query: The student's latest question.
        history: List of recent conversation turns [{'role': ..., 'content': ...}].
        api_key: Optional Gemini API key.

    Returns:
        (search_query, was_rewritten, rewriting_method)
        - search_query: Standalone query to be fed to BM25 and FAISS.
        - was_rewritten: Boolean flag.
        - rewriting_method: 'local_deterministic' | 'gemini_api' | 'none'
    """
    clean_q = query.strip()
    if not clean_q or not history:
        return clean_q, False, "none"

    is_follow_up, reason = is_follow_up_question(clean_q, history)
    if not is_follow_up:
        return clean_q, False, "none"

    # 1. Deterministic Local Rewriter (Fast, reliable, 0 cost)
    local_rewritten, was_rewritten = rewrite_query_locally(clean_q, history)
    if was_rewritten and local_rewritten != clean_q:
        return local_rewritten, True, "local_deterministic"

    # 2. Gemini API Fallback for complex ambiguous cases if key provided
    key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
    if key:
        gemini_rewritten = rewrite_query_with_gemini(clean_q, history, key)
        if gemini_rewritten and gemini_rewritten.lower() != clean_q.lower():
            return gemini_rewritten, True, "gemini_api"

    # Fallback to local rewrite attempt
    if local_rewritten != clean_q:
        return local_rewritten, True, "local_fallback"

    return clean_q, False, "none"
