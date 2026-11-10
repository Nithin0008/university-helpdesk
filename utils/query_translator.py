"""
utils/query_translator.py
--------------------------
Lightweight Multilingual Query Translation and Expansion for Indic Languages
(Telugu, Hindi, Punjabi) to English for Cross-Lingual RAG Retrieval.

Resource Constraints:
- 0 MB model download
- 0 additional RAM overhead
- Works offline via comprehensive university-domain bilingual dictionary
- Fast online translation via urllib standard library with 2s timeout and fallback
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List


# ---------------------------------------------------------------------------
# Comprehensive Domain-Specific Lexicon for University Regulations
# ---------------------------------------------------------------------------
# Covers Telugu, Hindi, and Punjabi common academic & policy terms
INDIC_DOMAIN_LEXICON = {
    # --- TELUGU ---
    "పాస్‌పోర్ట్": "passport",
    "పాస్ పోర్ట్": "passport",
    "చెల్లుబాటు": "validity valid",
    "చెల్లుబాటులో": "valid",
    "సెమిస్టర్ అబ్రాడ్": "semester abroad",
    "సెమిస్టర్": "semester",
    "అబ్రాడ్": "abroad",
    "విదేశీ": "foreign abroad",
    "ఆరు నెలలు": "six months",
    "నెలలు": "months",
    "కుటుంబ వ్యాపారం": "family-owned business family enterprise",
    "కుటుంబ యాజమాన్యం": "family-owned business",
    "కుటుంబ యాజమాన్యంలోని": "family-owned business",
    "కుటుంబ": "family-owned",
    "వ్యాపారం": "business enterprise",
    "వ్యాపారాలు": "businesses enterprises",
    "ఎంటర్‌ప్రైజ్": "enterprise",
    "సంస్థల": "enterprises organizations",
    "రాబడి": "revenue",
    "రాబడిని": "revenue",
    "ఆదాయం": "revenue",
    "పరిగణించబడుతుందా": "considered",
    "పరిగణిస్తారా": "considered",
    "పరిగణనలోకి": "considered",
    "తీసుకోబడుతుందా": "considered",
    "హాజరు": "attendance",
    "హాజరు ప్రయోజనం": "attendance benefit",
    "ప్రయోజనం": "benefit",
    "ప్రయోజనాలు": "benefits",
    "అర్హత": "eligibility eligible criteria",
    "అర్హులు": "eligible",
    "ప్రమాణాలు": "criteria standards",
    "కనీస": "minimum",
    "కనీసం": "minimum",
    "శాతం": "percent percentage",
    "కర్ఫ్యూ": "curfew timing",
    "రాత్రి": "night curfew",
    "సమయం": "timing hours",
    "హాస్టల్": "hostel",
    "లైబ్రరీ": "library",
    "పుస్తకం": "book",
    "పుస్తకాలు": "books",
    "జరిమానా": "fine overdue penalty",
    "ఫీజు": "fee tuition",
    "స్కాలర్‌షిప్": "scholarship",
    "గ్రేడ్": "grade",
    "పరీక్ష": "examination exam",

    # --- HINDI ---
    "पासपोर्ट": "passport",
    "वैधता": "validity valid",
    "सेमेस्टर अब्रॉड": "semester abroad",
    "सेमेस्टर": "semester",
    "अब्रॉड": "abroad",
    "विदेश": "abroad",
    "छह महीने": "six months",
    "महीने": "months",
    "पारिवारिक व्यवसाय": "family-owned business family enterprise",
    "पारिवारिक उद्यम": "family-owned business family enterprise",
    "पारिवारिक": "family",
    "व्यवसाय": "business",
    "उद्यम": "enterprise",
    "राजस्व": "revenue income",
    "आय": "revenue income",
    "कमाई": "revenue earnings",
    "विचार": "considered",
    "माना": "considered",
    "स्वीकार": "accepted considered",
    "उपस्थिति": "attendance",
    "उपस्थिति लाभ": "attendance benefit",
    "लाभ": "benefit",
    "पात्रता": "eligibility eligible",
    "न्यूनतम": "minimum",
    "कर्फ्यू": "curfew timing",
    "समय": "time timing",
    "पुस्तकालय": "library",
    "किताब": "book",
    "किताबें": "books",
    "जुर्माना": "fine overdue penalty",
    "विलंब शुल्क": "late fee fine",
    "शुल्क": "fee tuition",
    "छात्रवृत्ति": "scholarship",

    # --- PUNJABI ---
    "ਪਾਸਪੋਰਟ": "passport",
    "ਵੈਧਤਾ": "validity valid",
    "ਸਮੈਸਟਰ ਵਿਦੇਸ਼": "semester abroad",
    "ਸਮੈਸਟਰ": "semester",
    "ਵਿਦੇਸ਼": "abroad",
    "ਛੇ ਮਹੀਨੇ": "six months",
    "ਮਹੀਨੇ": "months",
    "ਪਰਿਵਾਰਕ ਕਾਰੋਬਾਰ": "family-owned business family enterprise",
    "ਪਰਿਵਾਰਕ ਉੱਦਮ": "family-owned business family enterprise",
    "ਪਰਿਵਾਰਕ": "family",
    "ਕਾਰੋਬਾਰ": "business",
    "ਉੱਦਮ": "enterprise",
    "ਕਮਾਈ": "revenue earnings",
    "ਆਮਦਨ": "revenue income",
    "ਵਿਚਾਰਿਆ": "considered counted",
    "ਮੰਨਿਆ": "considered",
    "ਹਾਜ਼ਰੀ": "attendance",
    "ਹਾਜ਼ਰੀ ਲਾਭ": "attendance benefit",
    "ਲਾਭ": "benefit",
    "ਯੋਗਤਾ": "eligibility eligible",
    "ਘੱਟੋ-ਘੱਟ": "minimum",
    "ਕਰਫਿਊ": "curfew timing",
    "ਸਮਾਂ": "time timing",
    "ਲਾਇਬ੍ਰੇਰੀ": "library",
    "ਕਿਤਾਬਾਂ": "books",
    "ਜੁਰਮਾਨਾ": "fine overdue penalty",
    "ਫੀਸ": "fee tuition",
    "ਵਜ਼ੀਫਾ": "scholarship"
}


def translate_via_free_api(text: str, source_lang: str = "auto", timeout: float = 2.5) -> Optional[str]:
    """
    Translates text to English using Google's public translation endpoint.
    Zero external dependencies, fast, non-blocking fallback on timeout or error.
    """
    if not text or not text.strip():
        return None

    clean_text = text.strip()
    encoded = urllib.parse.quote(clean_text)
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl=en&dt=t&q={encoded}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) UniversityHelpdesk/3.1"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                translated_parts = [part[0] for part in data[0] if part and len(part) > 0 and part[0]]
                if translated_parts:
                    translated = "".join(translated_parts).strip()
                    if translated and translated.lower() != clean_text.lower():
                        return translated
    except Exception:
        # Fall back gracefully to dictionary translation without disrupting user
        return None

    return None


def translate_via_dictionary(text: str) -> str:
    """
    Offline fallback: Translates and expands Indic queries using the university domain lexicon.
    Preserves numbers (e.g. 7.5, 10%), acronyms (CGPA, DL), and maps key policy concepts.
    """
    expanded_terms = []
    
    # Extract alphanumeric tokens, numbers, and symbols (e.g., 7.5, 10%, CGPA)
    ascii_tokens = re.findall(r'[a-zA-Z0-9.%]+', text)
    for tok in ascii_tokens:
        if len(tok) > 1 or tok.isdigit():
            expanded_terms.append(tok)

    # Match multi-word phrases first, then single words
    remaining = text
    # Sort lexicon keys by length descending to match longest phrases first
    sorted_phrases = sorted(INDIC_DOMAIN_LEXICON.keys(), key=len, reverse=True)

    matched_english = []
    for phrase in sorted_phrases:
        if phrase in remaining:
            eng_equiv = INDIC_DOMAIN_LEXICON[phrase]
            matched_english.append(eng_equiv)
            remaining = remaining.replace(phrase, " ")

    # Combine matched concepts
    all_english = list(dict.fromkeys(matched_english + expanded_terms))
    if all_english:
        return " ".join(all_english)

    return text


def detect_script_lang(text: str) -> str:
    """
    Detects language code based on script unicode code points.
    Returns 'te' (Telugu), 'hi' (Hindi/Devanagari), 'pa' (Punjabi/Gurmukhi), or 'en'.
    """
    for ch in text:
        cp = ord(ch)
        if 0x0C00 <= cp <= 0x0C7F:
            return "te"
        if 0x0900 <= cp <= 0x097F:
            return "hi"
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa"
    return "en"


def expand_query_multilingual(
    query: str,
    lang_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for cross-lingual query translation and expansion.
    Returns the original query and English translated/expanded query.
    Automatically detects Indic scripts (Telugu, Hindi, Punjabi) if lang_code is not provided.
    """
    clean_q = query.strip()
    if not clean_q:
        return {
            "original_query": "",
            "english_query": "",
            "is_translated": False,
            "method": "none"
        }

    detected_script = detect_script_lang(clean_q)
    actual_lang = lang_code if (lang_code and lang_code != "auto") else detected_script

    # If already English and no Indic script detected, no translation needed
    if actual_lang == "en" and detected_script == "en":
        return {
            "original_query": clean_q,
            "english_query": clean_q,
            "is_translated": False,
            "method": "direct"
        }

    source_lang = actual_lang if actual_lang != "en" else detected_script

    # 1. Attempt free fast online translation
    api_translation = translate_via_free_api(clean_q, source_lang=source_lang)
    dict_translation = translate_via_dictionary(clean_q)

    if api_translation:
        # Domain lexicon corrections for common translation confusions (e.g. policy revenue vs family annual income)
        if any(w in clean_q for w in ["రాబడి", "రాబడిని", "రాబడితో", "రాబడిపై", "రాబడిలో", "राजस्व", "ਆਮਦਨ"]):
            api_translation = re.sub(r'\bincome\b', 'revenue', api_translation, flags=re.IGNORECASE)
        if any(w in clean_q for w in ["కుటుంబ వ్యాపారం", "కుటుంబ", "पारिवारिक व्यवसाय", "ਪਰਿਵਾਰਕ ਕਾਰੋਬਾਰ"]):
            api_translation = re.sub(r'\bfamily business\b', 'family-owned business', api_translation, flags=re.IGNORECASE)

        combined_translation = api_translation
        if dict_translation and dict_translation != clean_q:
            api_lower = api_translation.lower()
            additional = [w for w in dict_translation.split() if w.lower() not in api_lower]
            if additional:
                combined_translation = f"{api_translation} {' '.join(additional)}"

        # Augment with critical numbers/acronyms if dropped
        for special in ["7.5", "10%", "CGPA", "60%", "65%"]:
            if special in clean_q and special.lower() not in combined_translation.lower():
                combined_translation += f" {special}"

        return {
            "original_query": clean_q,
            "english_query": combined_translation,
            "is_translated": True,
            "method": "online_api"
        }

    # 2. Offline fallback: domain lexicon expansion
    dict_translation = translate_via_dictionary(clean_q)
    return {
        "original_query": clean_q,
        "english_query": dict_translation,
        "is_translated": (dict_translation != clean_q),
        "method": "local_lexicon"
    }
