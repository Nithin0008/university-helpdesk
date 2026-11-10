"""
utils/category_manager.py
--------------------------
Category-based document management, automatic classification, and query routing.

Supported Categories:
1. Academics
2. Finance
3. Hostel
4. Careers
5. International Affairs
6. General / Other

Key Features:
- Document classification from filename, headings, and extracted text.
- Question classification for intelligent category-restricted retrieval.
- Multi-category assignment for cross-cutting policies.
- Fallback to 'All Categories' when classification confidence is low or uncertain.
- Multilingual keyword awareness for English, Hindi, Telugu, and Punjabi.
"""

import re
from typing import List, Dict, Tuple, Optional, Any

CATEGORIES = [
    "Academics",
    "Finance",
    "Hostel",
    "Careers",
    "International Affairs",
    "General / Other"
]

ALL_CATEGORIES = "All Categories"

# Category UI display icons / badges
CATEGORY_ICONS = {
    "Academics": "📚",
    "Finance": "💳",
    "Hostel": "🏠",
    "Careers": "💼",
    "International Affairs": "🌐",
    "General / Other": "📋",
    "All Categories": "🔍"
}

# ---------------------------------------------------------------------------
# Classification Dictionaries & Patterns
# ---------------------------------------------------------------------------

CATEGORY_FILENAME_KEYWORDS = {
    "Academics": [
        "academic", "attendance", "exam", "examination", "cgpa", "gpa", "grade",
        "condonation", "improvement", "syllabus", "odl", "nptel", "mooc", "course",
        "curriculum", "lecture", "student_handbook", "grading", "evaluation", "detention",
        "academic_regulations", "study", "reappear", "late_registration", "education_revolution",
        "edu_revolution"
    ],
    "Finance": [
        "fee", "fees", "finance", "tuition", "scholarship", "refund", "payment",
        "waiver", "fine", "dues", "challan", "penalty", "financial", "revenue",
        "bursar", "account", "accounts", "merit_cum_means", "lpunest", "scholarship_structure",
        "fee_payment"
    ],
    "Hostel": [
        "hostel", "mess", "dining", "curfew", "warden", "room", "residential",
        "night_out", "accommodation", "dorm", "housing", "in_time", "leave_request",
        "gate_pass", "hostel_handbook", "student_residential"
    ],
    "Careers": [
        "career", "placement", "placements", "internship", "job", "recruitment",
        "recruiter", "package", "ctc", "campus_drive", "resume", "interview",
        "training_and_placement", "tpo", "career_services", "placement_coordinator",
        "spc"
    ],
    "International Affairs": [
        "international", "abroad", "semester_abroad", "foreign", "exchange",
        "global", "passport", "visa", "partner_university", "oia", "mobility",
        "left_marking", "year_abroad"
    ],
    "General / Other": [
        "general", "conduct", "handbook", "anti_ragging", "helpline", "discipline",
        "id_card", "transport", "bus", "medical", "facilities", "administration",
        "campus", "circular", "notification", "certificate", "student_certificate",
        "library", "process"
    ]
}

CATEGORY_CONTENT_PATTERNS = {
    "Academics": [
        # English
        r"\b(?:attendance|75%|10% benefit|cgpa|grade point|examination|re-evaluation|condonation|detention)\b",
        r"\b(?:academic|internal marks|credits|nptel|moocs|patent grade|curriculum|hall ticket|backlog)\b",
        r"\b(?:semester|course registration|lecture|practical|mid-term|end-term|proctor)\b",
        r"\b(?:odl|distance learning|reappear|improvement exam|late registration|evaluation mechanism)\b",
        # Hindi
        r"(?:उपस्थिति|हाज़री|सीजीपीए|परीक्षा|ग्रेड|अंक|पाठ्यक्रम|अकादमिक|सेमेस्टर)",
        # Telugu
        r"(?:హాజరు|పరీక్ష|సిజిపిఎ|గ్రేడ్|మార్కులు|అకడమిక్|సెమిస్టర్|రీ-వాల్యుయేషన్|సిలబస్)",
        # Punjabi
        r"(?:ਹਾਜ਼ਰੀ|ਪ੍ਰੀਖਿਆ|ਸੀਜੀਪੀਏ|ਗ੍ਰੇਡ|ਅੰਕ|ਅਕਾਦਮਿਕ|ਸਮੈਸਟਰ)",
    ],
    "Finance": [
        # English
        r"\b(?:tuition fee|fees?|payment|refund|scholarship|fee waiver|financial aid|library fine)\b",
        r"\b(?:late fee|penalty|challan|family-owned business|revenue generation|installment|concession)\b",
        r"\b(?:lpunest|scholarship band|payment gateway|demand draft|bursar)\b",
        # Hindi
        r"(?:शुल्क|फीस|छात्रवृत्ति|रिफंड|जुर्माना|भुगतान|वित्तीय सहायता)",
        # Telugu
        r"(?:ఫీజు|స్కాలర్‌షిప్|రీఫండ్|చెల్లింపు|జరిమానా|ఆర్థిక సహాయం)",
        # Punjabi
        r"(?:ਫੀਸ|ਵਜ਼ੀਫ਼ਾ|ਰਿਫੰਡ|ਭੁਗਤਾਨ|ਜੁਰਮਾਨਾ)",
    ],
    "Hostel": [
        # English
        r"\b(?:hostel|curfew|9:30 pm|warden|mess|dining|room allocation|night out|turnstile)\b",
        r"\b(?:residential|dormitory|gate pass|leave approval|hostel fee|biometric)\b",
        r"\b(?:mess facility|hostel withdrawal|mess timings?|hostel rules)\b",
        # Hindi
        r"(?:हॉस्टल|छात्रावास|कर्फ्यू|वार्डन|मेस|भोजन|कमरा|रात की छुट्टी)",
        # Telugu
        r"(?:హాస్టల్|కర్ఫ్యూ|వార్డెన్|మెస్|భోజనం|గది|రాత్రి సెలవు)",
        # Punjabi
        r"(?:ਹੌਸਟਲ|ਕਰਫਿਊ|ਵਾਰਡਨ|ਮੈੱਸ|ਭੋਜਨ|ਕਮਰਾ)",
    ],
    "Careers": [
        # English
        r"\b(?:placement|placements|internship|internships|career|careers|job offer|campus drive)\b",
        r"\b(?:recruiter|package|ctc|interview|offer letter|resume|training and placement|tpo)\b",
        r"\b(?:pre-placement talk|shortlist|hiring|stipend|placement coordinator|career services|dcs)\b",
        # Hindi
        r"(?:प्लेसमेंट|इंटर्नशिप|करियर|नौकरी|भर्ती|पैकेज|साक्षात्कार|कंपनी)",
        # Telugu
        r"(?:ప్లేస్‌మెంట్|ఇంటర్న్‌షిప్|కెరీర్|ఉద్యోగం|నియామకం|ప్యాకేజీ|ఇంటర్వ్యూ)",
        # Punjabi
        r"(?:ਪਲੇਸਮੈਂਟ|ਇੰਟਰਨਸ਼ਿਪ|ਕੈਰੀਅਰ|ਨੌਕਰੀ|ਭਰਤੀ|ਇੰਟਰਵਿਊ)",
    ],
    "International Affairs": [
        # English
        r"\b(?:semester abroad|study abroad|foreign study|international mobility|partner university)\b",
        r"\b(?:passport|visa|international affairs|global exchange|foreign student|airfare)\b",
        r"\b(?:host institution|overseas|consulate|embassy|left marking|frro)\b",
        # Hindi
        r"(?:विदेश अध्ययन|सेमेस्टर अब्रॉड|पासपोर्ट|वीज़ा|अंतर्राष्ट्रीय|विदेशी विश्वविद्यालय)",
        # Telugu
        r"(?:సెమిస్టర్ అబ్రాడ్|విదేశీ అధ్యయనం|పాస్‌పోర్ట్|వీసా|అంతర్జాతీయ|విదేశీ యూనివర్సిటీ)",
        # Punjabi
        r"(?:ਵਿਦੇਸ਼ ਅਧਿਐਨ|ਸਮੈਸਟਰ ਵਿਦੇਸ਼|ਪਾਸਪੋਰਟ|ਵੀਜ਼ਾ|ਅੰਤਰਰਾਸ਼ਟਰੀ)",
    ],
    "General / Other": [
        # English
        r"\b(?:anti-ragging|helpline|bonafide|id card|bus|transport|medical center|health|discipline)\b",
        r"\b(?:administration|chancellor|registrar|code of conduct|grievance|facilities)\b",
        r"\b(?:certificate|bonafide certificate|profile verification|express mode|library policy|borrowing limit)\b",
        # Hindi
        r"(?:एंटी-रैगिंग|हेल्पलाइन|पहचान पत्र|प्रशासन|अनुशासन|सुविधाएं|प्रमाणपत्र)",
        # Telugu
        r"(?:యాంటీ-ర్యాగింగ్|హెల్ప్‌లైన్|గుర్తింపు కార్డు|రవాణా|పరిపాలన|సౌకర్యాలు|సర్టిఫికేట్)",
        # Punjabi
        r"(?:ਐਂਟੀ-ਰੈਗਿੰਗ|ਹੈਲਪਲਾਈਨ|ਸ਼ਨਾਖਤੀ ਕਾਰਡ|ਪ੍ਰਸ਼ਾਸਨ)",
    ]
}

# Mapping of known conversation topics to their core category
TOPIC_TO_CATEGORY = {
    "Semester Abroad": "International Affairs",
    "Passport and Travel": "International Affairs",
    "Hostel Curfew": "Hostel",
    "Hostel Mess Timings": "Hostel",
    "Hostel Night Leave": "Hostel",
    "Hostel Accommodation": "Hostel",
    "10% Attendance Benefit": "Academics",
    "Attendance Policy": "Academics",
    "CGPA Eligibility": "Academics",
    "Examination Paper Re-evaluation": "Academics",
    "Patent Grade Upgradation": "Academics",
    "NPTEL and MOOCs": "Academics",
    "Hackathons and Competitions": "Academics",
    "Family-Owned Business Revenue": "Finance",
    "Merit Scholarship": "Finance",
    "Financial Aid": "Finance",
    "Library Overdue Fine": "Finance",
    "Tuition Fee Payment": "Finance",
    "Campus Placement": "Careers",
    "ODL Evaluation": "Academics",
    "Open and Distance Learning": "Academics",
    "Student Certificates": "General / Other",
    "Career Services Enrolment": "Careers",
    "Student Placement Coordinator": "Careers",
    "Library Policy": "General / Other",
    "Hostel Residential and Mess": "Hostel",
    "LPUNEST Scholarship": "Finance",
    "Fee Payment Modes": "Finance",
    "International Student Left Marking": "International Affairs",
    "Reappear and Improvement": "Academics",
    "Late Registration Attendance": "Academics",
}


def classify_document(filename: str, text: str = "") -> List[str]:
    """
    Automatically classifies a document into one or more categories based on:
    1. Directory and folder path hints (e.g. docs_raw/academic/ -> Academics).
    2. Filename terms (heavily weighted).
    3. Headings and content keywords (frequency weighted).
    
    Returns a sorted list of matched categories. Primary category is at index 0.
    """
    scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    fn_normalized = filename.replace("\\", "/").lower()
    path_parts = fn_normalized.split("/")
    fn_base = path_parts[-1]
    text_sample = text[:15000].lower() if text else ""

    # 1. Folder path hints (weight = 8.0)
    if len(path_parts) > 1:
        folders = path_parts[:-1]
        for f in folders:
            if f == "academic":
                scores["Academics"] += 8.0
            elif f == "finance":
                scores["Finance"] += 8.0
            elif f == "career":
                scores["Careers"] += 8.0
            elif f == "hostel":
                scores["Hostel"] += 8.0
            elif f == "international":
                scores["International Affairs"] += 8.0
            elif f == "administration":
                scores["General / Other"] += 8.0
            elif f == "facilities":
                scores["General / Other"] += 4.0

    # 2. Filename scoring (weight = 5.0)
    for cat, kw_list in CATEGORY_FILENAME_KEYWORDS.items():
        for kw in kw_list:
            if kw in fn_base:
                scores[cat] += 5.0

    # 3. Content pattern scoring
    if text_sample:
        for cat, patterns in CATEGORY_CONTENT_PATTERNS.items():
            for p in patterns:
                matches = re.findall(p, text_sample, re.IGNORECASE)
                if matches:
                    scores[cat] += min(10.0, len(matches) * 0.8)

    # Filter categories with non-zero score and sort
    ranked = sorted(
        [(cat, score) for cat, score in scores.items() if score > 0],
        key=lambda x: x[1],
        reverse=True
    )

    if not ranked:
        return ["General / Other"]

    top_cat, top_score = ranked[0]
    results = [top_cat]

    # Support multiple categories if secondary category has substantial score
    for cat, score in ranked[1:]:
        if score >= 3.0 and score >= (top_score * 0.55):
            results.append(cat)

    return results


def classify_question(
    query: str,
    context_topic: Optional[str] = None
) -> Tuple[Optional[str], float]:
    """
    Identifies the likely category of a student's question.
    
    Args:
        query: The student's question (or rewritten query).
        context_topic: Active conversation topic if follow-up.
        
    Returns:
        (category, confidence)
        If confidence < 0.60 or ambiguous, returns (None, confidence)
        to trigger safe fallback to All Categories.
    """
    if not query:
        return None, 0.0

    q_lower = query.lower()

    # Rule 1: Context topic mapping for follow-ups
    if context_topic and context_topic in TOPIC_TO_CATEGORY:
        mapped_cat = TOPIC_TO_CATEGORY[context_topic]
        # Check if question seems to stay on topic
        return mapped_cat, 0.90

    # Rule 2: High-precision domain intent patterns
    scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}

    # Direct keyword matches
    for cat, patterns in CATEGORY_CONTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q_lower, re.IGNORECASE):
                scores[cat] += 1.0

    # Extra exact-phrase boosts
    if re.search(r"\b(?:cgpa|attendance|exam|condonation|grade|paper|course|credit)\b", q_lower):
        scores["Academics"] += 2.0
    if re.search(r"\b(?:curfew|hostel|mess|dining|warden|night out|in-time)\b", q_lower):
        scores["Hostel"] += 2.0
    if re.search(r"\b(?:fee|fees|refund|scholarship|waiver|fine|challan|payment|financial)\b", q_lower):
        scores["Finance"] += 2.0
    if re.search(r"\b(?:placement|internship|job|career|recruitment|ctc|package)\b", q_lower):
        scores["Careers"] += 2.0
    if re.search(r"\b(?:semester abroad|passport|visa|foreign|abroad|partner university)\b", q_lower):
        scores["International Affairs"] += 2.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score == 0:
        return None, 0.0

    # Calculate confidence margin
    if top_score >= 2.0:
        if second_score == 0:
            confidence = 0.95
        else:
            confidence = min(0.95, 0.65 + 0.15 * (top_score - second_score))
    elif top_score >= 1.0:
        if second_score == 0:
            confidence = 0.80
        else:
            confidence = 0.55  # Ambiguous / uncertain
    else:
        confidence = 0.30

    if confidence >= 0.65:
        return top_cat, round(confidence, 2)
    else:
        # Uncertain classification -> return None to safely fall back to searching all categories
        return None, round(confidence, 2)

