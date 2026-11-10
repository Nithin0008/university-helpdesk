"""
utils/document_processor.py
---------------------------
Handles extracting text from PDF and TXT documents, dividing the text into
manageable chunks, and preserving metadata (filename, page numbers).

Why chunking is needed in RAG:
In Retrieval-Augmented Generation (RAG), large documents are broken into
smaller chunks (e.g., 300-500 characters). This allows the search system to
find the exact sentence or paragraph that answers the student's question,
rather than returning an entire 50-page handbook.
"""

import io
import zipfile
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from pypdf import PdfReader


def extract_text_from_txt(content: bytes | str, filename: str = "document.txt") -> List[Dict[str, Any]]:
    """
    Extracts text from a plain text (.txt) document.
    
    Args:
        content: The text content as string or bytes.
        filename: Name of the uploaded file.
        
    Returns:
        List of dictionaries with 'text', 'page', and 'source'.
    """
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
    else:
        text = str(content)

    clean_text = text.strip()
    if not clean_text:
        return []

    # Plain text files have 1 page by default
    return [{
        "text": clean_text,
        "page": 1,
        "source": filename
    }]


def extract_text_from_pdf(content: bytes | io.BytesIO, filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF document using pypdf.
    Preserves exact logical and printed page numbers for each extracted page
    by inspecting document pagination sequences (e.g. accounting for front-matter
    covers and tables of contents).
    
    Args:
        content: The PDF bytes or stream.
        filename: Name of the uploaded PDF file.
        
    Returns:
        List of dictionaries with 'text', 'page' (logical/printed page number),
        'phys_page' (1-indexed physical page), and 'source'.
    """
    if isinstance(content, bytes):
        stream = io.BytesIO(content)
    else:
        stream = content

    reader = PdfReader(stream)
    pages_data = []

    # First pass: detect printed page numbers and compute document pagination offset
    detected_offsets = []
    page_raw_records = []

    for index, page in enumerate(reader.pages):
        phys_page = index + 1
        page_text = page.extract_text() or ""
        clean_text = page_text.strip()
        lines = [l.strip() for l in clean_text.split("\n") if l.strip()]

        printed_num = None
        if lines:
            # Check first line or last line for standalone page number digits
            if lines[0].isdigit() and 1 <= int(lines[0]) <= 500:
                printed_num = int(lines[0])
            elif lines[-1].isdigit() and 1 <= int(lines[-1]) <= 500:
                printed_num = int(lines[-1])

        if printed_num is not None:
            detected_offsets.append(phys_page - printed_num)

        page_raw_records.append({
            "phys_page": phys_page,
            "printed_num": printed_num,
            "clean_text": clean_text
        })

    # Determine dominant pagination offset if document uses front-matter
    common_offset = 0
    if detected_offsets:
        from collections import Counter
        offset_counts = Counter(detected_offsets)
        most_common_offset, count = offset_counts.most_common(1)[0]
        if count >= 3:
            common_offset = most_common_offset

    # Second pass: assign verified logical page numbers
    for record in page_raw_records:
        phys_page = record["phys_page"]
        clean_text = record["clean_text"]
        if not clean_text:
            continue

        if record["printed_num"] is not None:
            logical_page = record["printed_num"]
        elif common_offset > 0 and phys_page > common_offset:
            logical_page = phys_page - common_offset
        else:
            logical_page = phys_page

        pages_data.append({
            "text": clean_text,
            "page": logical_page,
            "phys_page": phys_page,
            "source": filename
        })

    return pages_data


def extract_programme_scope(filename: str, text: str = "") -> str:
    """
    Extracts the programme applicability of a policy document:
    - 'ODL' for Open and Distance Learning programmes.
    - 'Regular' for regular / on-campus degree programmes.
    - 'General' for university-wide policies applicable across all modes.
    """
    fn_lower = filename.lower()
    text_sample = text[:2500].lower() if text else ""

    # 1. Distinct ODL indicators (ensure it's the document's subject, not an exclusion)
    if "odl" in fn_lower or "distance learning" in fn_lower or "open and distance learning" in text_sample:
        if "odl" in fn_lower or "for odl" in text_sample or "(odl)" in text_sample or "evaluation framework applicable to students enrolled in open and distance learning" in text_sample:
            return "ODL"

    # 2. Regular on-campus indicators
    if any(k in fn_lower or k in text_sample for k in [
        "btech", "residential", "hostel", "mess", "regular", "on-campus",
        "full-time", "reappear", "attendance", "placement", "abroad", "library"
    ]):
        return "Regular"

    return "General"


def extract_document_sections(text: str) -> List[Dict[str, Any]]:
    """
    Parses section and subsection markers with their character offsets in the document.
    Extracts authentic sections from DOCX/PDF content:
      - Numbered sections: e.g. 'SECTION 1 – 10% ATTENDANCE BENEFIT', 'SECTION 1: ATTENDANCE POLICY'
      - Numbered subsections: e.g. '1.1 Minimum Attendance Requirement:'
      - Major structural and uppercase headings: e.g. 'RULES OF CONDUCT IN RESIDENTIAL FACILITY',
        'MESS PLAN', 'ONLINE PAYMENT OPTIONS', 'PROCEDURE — ...', 'KEY RULES AND POLICY DETAILS'
    """
    import re
    lines = text.split("\n")
    sections = []
    char_offset = 0

    section_re = re.compile(
        r'^\s*(?:SECTION\s+(\d+|[IVXLCDM]+)\s*[-:–—]\s*([^\n]+)|SECTION\s+(\d+|[IVXLCDM]+)\s*:\s*([^\n]+)|(\d+\.\d+)\s+([^:\n]+)[:\n]?)',
        re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        line_len = len(line) + 1  # account for \n

        if not stripped:
            char_offset += line_len
            continue

        # Ignore table rows, bullets, URLs, and document title markers
        if stripped.startswith(("|", "•", "-", "*", "http://", "https://")) or stripped.startswith("TITLE "):
            char_offset += line_len
            continue

        # Check numbered section
        m = section_re.match(stripped)
        if m:
            if m.group(5):  # 1.1 subsection
                num = m.group(5).strip()
                title = m.group(6).strip().rstrip(":")
                sections.append({
                    "num": num,
                    "title": title,
                    "start": char_offset,
                    "end": char_offset + len(stripped),
                    "is_major": False
                })
            elif m.group(1):
                num = m.group(1).strip()
                title = m.group(2).strip()
                sections.append({
                    "num": num,
                    "title": title,
                    "start": char_offset,
                    "end": char_offset + len(stripped),
                    "is_major": True
                })
            elif m.group(3):
                num = m.group(3).strip()
                title = m.group(4).strip()
                sections.append({
                    "num": num,
                    "title": title,
                    "start": char_offset,
                    "end": char_offset + len(stripped),
                    "is_major": True
                })
            char_offset += line_len
            continue

        # Check structural or uppercase heading
        # Uppercase line, length 3..80 chars, letters present, not ending with question mark
        letters = [c for c in stripped if c.isalpha()]
        if letters and len(stripped) <= 80 and not stripped.endswith("?"):
            is_upper = all(c.isupper() for c in letters)
            is_proc = stripped.startswith(("PROCEDURE —", "PROCEDURE -", "SUMMARY TABLE —", "SUMMARY TABLE -"))
            if is_upper or is_proc:
                sections.append({
                    "num": "",
                    "title": stripped,
                    "start": char_offset,
                    "end": char_offset + len(stripped),
                    "is_major": True
                })
                char_offset += line_len
                continue

        char_offset += line_len

    return sections


def extract_text_from_docx(content: bytes | io.BytesIO, filename: str = "document.docx") -> List[Dict[str, Any]]:
    """
    Extracts text from a Microsoft Word (.docx) document using standard library
    zipfile and XML parsing. Does not require external python-docx.
    
    Extracts:
    - Paragraphs and headings
    - Numbered and bulleted lists (with bullet prefix)
    - Tables (formatted cleanly as Markdown tables with headers and cell alignment)
    - Sets page=None and phys_page=None (does not invent artificial page numbers).
    
    Args:
        content: The docx bytes or stream.
        filename: Name of the uploaded file.
        
    Returns:
        List of dictionaries with 'text', 'page': None, and 'source'.
    """
    stream = io.BytesIO(content) if isinstance(content, bytes) else content
    blocks = []
    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    try:
        with zipfile.ZipFile(stream) as z:
            xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            body = tree.find(f"{w_ns}body")
            if body is not None:
                for child in body:
                    tag = child.tag.replace(w_ns, "")
                    if tag == "p":
                        is_list = child.find(f".//{w_ns}numPr") is not None
                        texts = [node.text for node in child.iter(f"{w_ns}t") if node.text]
                        p_text = "".join(texts).strip()
                        if p_text:
                            if is_list and not p_text.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                                p_text = f"• {p_text}"
                            blocks.append(p_text)
                    elif tag == "tbl":
                        rows = child.findall(f"{w_ns}tr")
                        tbl_lines = []
                        for r_idx, r in enumerate(rows):
                            cells = []
                            for c in r.findall(f"{w_ns}tc"):
                                c_texts = [n.text for n in c.iter(f"{w_ns}t") if n.text]
                                cell_val = " ".join("".join(c_texts).split())
                                cells.append(cell_val)
                            if any(cells):
                                tbl_lines.append("| " + " | ".join(cells) + " |")
                                if r_idx == 0 and len(cells) > 1:
                                    tbl_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                        if tbl_lines:
                            blocks.append("\n".join(tbl_lines))
    except Exception:
        return []

    clean_text = "\n\n".join(blocks).strip()
    if not clean_text:
        return []

    # Fluid pagination: DO NOT invent arbitrary page numbers for DOCX files
    return [{
        "text": clean_text,
        "page": None,
        "phys_page": None,
        "source": filename
    }]


def chunk_text(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = 650,
    chunk_overlap: int = 130,
    category: Optional[str] = None,
    categories: Optional[List[str]] = None,
    programme_scope: Optional[str] = None,
    doc_hash: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Splits document pages into smaller chunks while preserving source file,
    logical/printed page number, physical page number, verified section metadata,
    category classifications, and programme applicability.
    
    Args:
        pages_data: List of page dictionaries from extract functions.
        chunk_size: Target character count per chunk.
        chunk_overlap: Overlap character count between consecutive chunks.
        category: Primary category name.
        categories: Full list of assigned categories.
        programme_scope: Applicability mode ('ODL', 'Regular', 'General').
        doc_hash: SHA-256 content hash of the source document.
        
    Returns:
        List of chunk dictionaries.
    """
    chunks = []
    chunk_index = 0
    assigned_cats = categories or ([category] if category else ["General / Other"])
    primary_cat = category or (assigned_cats[0] if assigned_cats else "General / Other")
    scope = programme_scope or "General"
    content_hash = doc_hash or ""

    for page_entry in pages_data:
        text = page_entry["text"]
        source = page_entry["source"]
        page_num = page_entry.get("page")
        phys_page_num = page_entry.get("phys_page", page_num)

        clean_text = text.strip()
        if not clean_text:
            continue

        doc_sections = extract_document_sections(clean_text)

        def get_section_at(pos: int):
            active = None
            for s in doc_sections:
                if s["start"] <= pos:
                    active = s
                else:
                    break
            return active

        # Construct distinct chunk_id without inventing page number
        def build_chunk_id(p_num: Optional[int], idx: int) -> str:
            if p_num is not None:
                return f"{source}_p{p_num}_c{idx}"
            return f"{source}_c{idx}"

        # If text is shorter than chunk size, keep it as a single chunk
        if len(clean_text) <= chunk_size:
            active_sec = get_section_at(0)
            chunks.append({
                "chunk_id": build_chunk_id(page_num, chunk_index),
                "text": clean_text,
                "source": source,
                "page": page_num,
                "phys_page": phys_page_num,
                "char_count": len(clean_text),
                "section": active_sec["num"] if (active_sec and not active_sec["is_major"]) else (active_sec["num"] if active_sec else ""),
                "section_title": active_sec["title"] if active_sec else "",
                "sections_in_chunk": [
                    {"num": s["num"], "title": s["title"], "is_major": s["is_major"], "rel_offset": s["start"]}
                    for s in doc_sections
                ],
                "category": primary_cat,
                "categories": assigned_cats,
                "programme_scope": scope,
                "doc_hash": content_hash
            })
            chunk_index += 1
            continue

        # Split using sliding window with natural boundary
        start = 0
        text_len = len(clean_text)
        step_size = max(1, chunk_size - chunk_overlap)

        while start < text_len:
            end = min(start + chunk_size, text_len)

            # Break at natural word/line boundary
            if end < text_len:
                last_space = clean_text.rfind(" ", start, end)
                last_newline = clean_text.rfind("\n", start, end)
                boundary = max(last_space, last_newline)
                if boundary > start + (chunk_size // 2):
                    end = boundary

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                active_sec = get_section_at(start)
                inner = [s for s in doc_sections if start <= s["start"] < end]

                primary_sec = ""
                primary_title = ""
                if active_sec and not active_sec["is_major"]:
                    primary_sec = active_sec["num"]
                    primary_title = active_sec["title"]
                elif inner:
                    subsecs = [s for s in inner if not s["is_major"]]
                    chosen = subsecs[0] if subsecs else inner[0]
                    primary_sec = chosen["num"]
                    primary_title = chosen["title"]
                elif active_sec:
                    primary_sec = active_sec["num"]
                    primary_title = active_sec["title"]

                chunks.append({
                    "chunk_id": build_chunk_id(page_num, chunk_index),
                    "text": chunk_content,
                    "source": source,
                    "page": page_num,
                    "phys_page": phys_page_num,
                    "char_count": len(chunk_content),
                    "section": primary_sec,
                    "section_title": primary_title,
                    "sections_in_chunk": [
                        {"num": s["num"], "title": s["title"], "is_major": s["is_major"], "rel_offset": s["start"] - start}
                        for s in inner
                    ],
                    "category": primary_cat,
                    "categories": assigned_cats,
                    "programme_scope": scope,
                    "doc_hash": content_hash
                })
                chunk_index += 1

            if end >= text_len:
                break

            actual_step = max(1, end - start - chunk_overlap) if (end - start > chunk_overlap) else step_size
            start += actual_step

    return chunks


def process_document(
    file_bytes_or_str: bytes | str,
    filename: str,
    chunk_size: int = 650,
    chunk_overlap: int = 130,
    categories: Optional[List[str]] = None,
    category: Optional[str] = None,
    programme_scope: Optional[str] = None,
    doc_hash: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function: determines file type (.pdf, .docx, or .txt),
    extracts text with page numbers or markdown tables, automatically assigns
    category metadata and programme applicability, and divides into chunks.
    """
    import hashlib
    lower_name = filename.lower()
    
    if isinstance(file_bytes_or_str, bytes):
        content_hash = doc_hash or hashlib.sha256(file_bytes_or_str).hexdigest()
    else:
        content_hash = doc_hash or hashlib.sha256(str(file_bytes_or_str).encode("utf-8")).hexdigest()

    if lower_name.endswith(".pdf"):
        pages_data = extract_text_from_pdf(file_bytes_or_str, filename)
    elif lower_name.endswith(".docx"):
        pages_data = extract_text_from_docx(file_bytes_or_str, filename)
    elif lower_name.endswith(".txt"):
        pages_data = extract_text_from_txt(file_bytes_or_str, filename)
    else:
        pages_data = extract_text_from_txt(file_bytes_or_str, filename)

    if not pages_data:
        return []

    all_text = " ".join([p.get("text", "") for p in pages_data])

    # Determine programme scope
    scope = programme_scope or extract_programme_scope(filename, all_text)

    # Determine category metadata
    if categories:
        assigned_cats = list(categories)
    elif category:
        assigned_cats = [category]
    else:
        from utils.category_manager import classify_document
        assigned_cats = classify_document(filename, all_text)

    primary_cat = assigned_cats[0] if assigned_cats else "General / Other"

    return chunk_text(
        pages_data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        category=primary_cat,
        categories=assigned_cats,
        programme_scope=scope,
        doc_hash=content_hash
    )


# ---------------------------------------------------------------------------
# Document Discovery, Content Hashing, and Persistent Indexing
# ---------------------------------------------------------------------------

def compute_content_hash(content: bytes | str) -> str:
    """Computes SHA-256 hash of document content for deduplication and cache validation."""
    import hashlib
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


EXCLUDED_OLD_SAMPLES = {
    "academic benefits.pdf",
    "sample_hostel_policy.pdf",
    "sample_library_and_fee_rules.txt",
    "sample_university_policy.txt"
}


def scan_all_documents(samples_dir: str | Path = "sample_documents/docs_raw") -> List[Path]:
    """
    Recursively scans the directory and subdirectories to discover all active PDF, DOCX, and TXT policy files.
    Strictly uses docs_raw as the active knowledge source and excludes all old demo/sample files and archive folders.
    
    Returns a sorted list of Path objects.
    """
    root = Path(samples_dir)
    if (root / "docs_raw").exists() and (root / "docs_raw").is_dir():
        root = root / "docs_raw"
    elif not root.exists():
        alt = Path("sample_documents/docs_raw")
        if alt.exists():
            root = alt
        else:
            return []

    valid_exts = {".pdf", ".docx", ".txt"}
    files = [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in valid_exts
        and ".index_cache" not in str(p).lower()
        and "archive" not in str(p).lower()
        and p.name.lower() not in EXCLUDED_OLD_SAMPLES
    ]
    return sorted(files, key=lambda p: str(p).lower())


def load_or_build_document_index(
    samples_dir: str | Path = "sample_documents/docs_raw",
    vector_engine: Any = None,
    search_engine: Any = None,
    cache_dir: Optional[str | Path] = None,
    force_reindex: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Main persistent ingestion engine:
    1. Recursively discovers all active documents in docs_raw and subdirectories.
    2. Calculates SHA-256 content hashes to avoid duplicate ingestion.
    3. If cache is valid and matches all document hashes, loads chunks and precomputed
       embeddings in milliseconds without re-encoding.
    4. If new or modified documents are discovered, processes only those files,
       updates the FAISS embeddings, and persists the updated cache to disk.
    
    Returns:
        (indexed_chunks, doc_categories)
    """
    import json
    import pickle
    import numpy as np

    root = Path(samples_dir)
    if (root / "docs_raw").exists() and (root / "docs_raw").is_dir():
        root = root / "docs_raw"
    elif not root.exists():
        alt = Path("sample_documents/docs_raw")
        if alt.exists():
            root = alt
        else:
            return [], {}

    c_dir = Path(cache_dir) if cache_dir else root / ".index_cache"
    c_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = c_dir / "cache_manifest.json"
    chunks_path = c_dir / "cached_chunks.pkl"
    embeddings_path = c_dir / "cached_embeddings.npy"

    discovered_files = scan_all_documents(root)
    file_hashes: Dict[str, Dict[str, Any]] = {}

    for f_path in discovered_files:
        raw_bytes = f_path.read_bytes()
        h = compute_content_hash(raw_bytes)
        rel_key = str(f_path.relative_to(root)).replace("\\", "/")
        file_hashes[rel_key] = {
            "path": str(f_path),
            "filename": f_path.name,
            "hash": h,
            "size": len(raw_bytes),
            "ext": f_path.suffix.lower()
        }

    # Check if existing cache matches current file state
    cache_is_valid = False
    if not force_reindex and manifest_path.exists() and chunks_path.exists() and embeddings_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                saved_manifest = json.load(mf)
            
            # Check if all current files are in manifest with exact hashes
            saved_docs = saved_manifest.get("documents", {})
            # Ensure no excluded old sample files exist in the saved manifest
            has_excluded = any(
                info.get("filename", "").lower() in EXCLUDED_OLD_SAMPLES
                for info in saved_docs.values()
            )
            if not has_excluded and len(saved_docs) == len(file_hashes):
                all_match = True
                for rel_k, info in file_hashes.items():
                    if rel_k not in saved_docs or saved_docs[rel_k].get("hash") != info["hash"]:
                        all_match = False
                        break
                if all_match:
                    cache_is_valid = True
        except Exception:
            cache_is_valid = False

    if cache_is_valid:
        try:
            with open(chunks_path, "rb") as cf:
                cached_chunks = pickle.load(cf)
            cached_embeddings = np.load(embeddings_path)

            doc_categories: Dict[str, List[str]] = {}
            for c in cached_chunks:
                src = c.get("source", "")
                if src and src not in doc_categories:
                    doc_categories[src] = c.get("categories", [c.get("category", "General / Other")])

            if search_engine is not None:
                search_engine.index_chunks(cached_chunks)
            if vector_engine is not None:
                vector_engine.load_precomputed_embeddings(cached_chunks, cached_embeddings)

            return cached_chunks, doc_categories
        except Exception:
            cache_is_valid = False

    # Build or incrementally update index
    all_chunks: List[Dict[str, Any]] = []
    doc_categories = {}
    manifest_docs: Dict[str, Any] = {}

    for rel_k, info in file_hashes.items():
        f_path = Path(info["path"])
        raw_bytes = f_path.read_bytes()
        doc_chunks = process_document(
            raw_bytes,
            filename=info["filename"],
            doc_hash=info["hash"]
        )
        if doc_chunks:
            primary_c = doc_chunks[0].get("category", "General / Other")
            assigned = doc_chunks[0].get("categories", [primary_c])
            doc_categories[info["filename"]] = assigned
            all_chunks.extend(doc_chunks)

        manifest_docs[rel_k] = {
            "filename": info["filename"],
            "hash": info["hash"],
            "size": info["size"],
            "chunk_count": len(doc_chunks),
            "categories": doc_categories.get(info["filename"], ["General / Other"]),
            "programme_scope": doc_chunks[0].get("programme_scope", "General") if doc_chunks else "General"
        }

    # Index with search engines
    if search_engine is not None:
        search_engine.index_chunks(all_chunks)

    if vector_engine is not None:
        vector_engine.index_chunks(all_chunks)
        embeddings_to_save = vector_engine.embeddings
    else:
        embeddings_to_save = np.empty((0, 384), dtype=np.float32)

    # Persist cache to disk
    try:
        with open(chunks_path, "wb") as cf:
            pickle.dump(all_chunks, cf)
        if embeddings_to_save is not None and embeddings_to_save.size > 0:
            np.save(embeddings_path, embeddings_to_save)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump({
                "version": "1.0",
                "total_documents": len(manifest_docs),
                "total_chunks": len(all_chunks),
                "documents": manifest_docs
            }, mf, indent=2)
    except Exception:
        pass

    return all_chunks, doc_categories
