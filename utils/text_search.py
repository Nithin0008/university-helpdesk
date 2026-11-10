"""
utils/text_search.py
--------------------
Implements a lightweight, fast, pure-Python search engine using BM25
(Best Matching 25) scoring.

What is BM25?
BM25 is the gold-standard algorithm used by search engines (like Lucene and
Elasticsearch). It calculates how relevant a document passage is to a search
query based on:
1. Term Frequency (TF): How often query words appear in a chunk.
2. Inverse Document Frequency (IDF): Rare words across the document carry more
   weight than common words.
3. Document Length Normalization: Prevents longer chunks from having an unfair advantage.

This requires 0 MB download, uses minimal RAM, and supports Unicode
characters (English, Hindi, Telugu, Punjabi).
"""

import math
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter


import unicodedata

def tokenize(text: str) -> List[str]:
    """
    Splits text into lowercase word tokens.
    Preserves exact numeric expressions (e.g. 7.5, 9.0), percentages (e.g. 10%, 75%),
    time formats (e.g. 9:30), and hyphenated policy terms (e.g. family-owned) while
    cleaning general punctuation across all languages without breaking Indic script
    combining marks (matras/viramas in Hindi, Telugu, Punjabi).
    """
    if not text:
        return []

    chars = []
    n = len(text)
    for i, c in enumerate(text):
        if unicodedata.category(c).startswith("P"):
            # Context-sensitive punctuation preservation:
            # 1. Decimal points between digits (e.g. 7.5, 9.0, 1.1)
            if c == '.' and 0 < i < n - 1 and text[i-1].isdigit() and text[i+1].isdigit():
                chars.append(c)
            # 2. Percentages immediately following digits (e.g. 10%, 75%, 50%)
            elif c == '%' and i > 0 and text[i-1].isdigit():
                chars.append(c)
            # 3. Colons between digits (e.g. 9:30, 7:30)
            elif c == ':' and 0 < i < n - 1 and text[i-1].isdigit() and text[i+1].isdigit():
                chars.append(c)
            # 4. Hyphens between alphanumeric characters (e.g. family-owned, pre-final)
            elif c == '-' and 0 < i < n - 1 and (text[i-1].isalnum() and text[i+1].isalnum()):
                chars.append(c)
            else:
                chars.append(" ")
        else:
            chars.append(c)

    cleaned = "".join(chars).lower()
    raw_tokens = cleaned.split()

    tokens = []
    for t in raw_tokens:
        tokens.append(t)
        # Emit sub-tokens for compound tokens so partial/generic queries also match
        if "." in t:
            for p in t.split("."):
                if p and p not in tokens:
                    tokens.append(p)
        if "%" in t:
            num = t.replace("%", "")
            if num and num not in tokens:
                tokens.append(num)
        if "-" in t:
            for p in t.split("-"):
                if p and p not in tokens:
                    tokens.append(p)

    return tokens


ENGLISH_STOP_WORDS = {
    "what", "is", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "s", "which", "who", "whom", "this", "that",
    "it", "its", "are", "were", "was", "be", "been"
}


class BM25SearchEngine:
    """
    Lightweight in-memory BM25 search index for document chunks.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: Controls term frequency saturation (standard default: 1.5)
            b: Controls document length normalization (standard default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_frequencies: Dict[str, int] = Counter()
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Indexes a list of document chunks into memory.
        """
        self.chunks = chunks
        self.num_docs = len(chunks)
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_frequencies = Counter()

        if self.num_docs == 0:
            self.avg_doc_length = 0.0
            return

        total_length = 0
        for chunk in chunks:
            tokens = tokenize(chunk.get("text", ""))
            self.doc_tokens.append(tokens)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length
            
            # Count distinct tokens per document for IDF
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] += 1

        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0.0

    def _compute_idf(self, term: str) -> float:
        """
        Calculates Inverse Document Frequency for a term.
        Terms that appear in fewer documents get higher importance.
        """
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        # Standard Robertson-Spärck Jones BM25 IDF formula
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        translated_query: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the indexed chunks for passages relevant to the query.
        Supports cross-lingual retrieval via an optional translated_query
        and category-restricted filtering with safe fallback.
        
        Args:
            query: The student's question string.
            top_k: Number of top results to return.
            translated_query: Optional translated/expanded English query.
            category: Optional category filter (e.g. 'Academics', 'Hostel').
            
        Returns:
            List of matching dictionaries sorted by relevance score:
            - chunk: Original chunk data (text, source, page, chunk_id, category)
            - score: BM25 relevance score
            - matched_terms: List of query words found in the passage
        """
        if not query or self.num_docs == 0:
            return []

        if translated_query is None:
            try:
                from utils.query_translator import expand_query_multilingual
                exp = expand_query_multilingual(query)
                if exp.get("is_translated"):
                    translated_query = exp.get("english_query")
            except Exception:
                pass

        all_tokens = tokenize(query)
        if translated_query and translated_query.strip() != query.strip():
            all_tokens.extend(tokenize(translated_query))

        if not all_tokens:
            return []

        # Filter out common stop words so 'the', 'is', 'what' don't produce false matches
        query_tokens = [t for t in all_tokens if t not in ENGLISH_STOP_WORDS]
        if not query_tokens:
            query_tokens = all_tokens

        # Deduplicate query tokens while preserving order
        query_tokens = list(dict.fromkeys(query_tokens))

        use_category = category if (category and category != "All Categories") else None

        def _execute_search(cat_filter: Optional[str]) -> List[Tuple[int, float, List[str]]]:
            res_scores = []
            for doc_idx, doc_words in enumerate(self.doc_tokens):
                if not doc_words:
                    continue

                chunk = self.chunks[doc_idx]
                if cat_filter:
                    chunk_cat = chunk.get("category")
                    chunk_cats = chunk.get("categories", [])
                    if chunk_cat != cat_filter and cat_filter not in chunk_cats:
                        continue

                doc_len = self.doc_lengths[doc_idx]
                word_counts = Counter(doc_words)
                doc_score = 0.0
                matched_terms = []

                for q_term in query_tokens:
                    if q_term in word_counts:
                        matched_terms.append(q_term)
                        tf = word_counts[q_term]
                        idf = self._compute_idf(q_term)
                        
                        # BM25 term weighting formula
                        numerator = tf * (self.k1 + 1.0)
                        denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_length or 1.0)))
                        term_score = idf * (numerator / denominator)
                        doc_score += term_score

                # Only return chunks where at least one meaningful term matched
                if doc_score > 0 and matched_terms:
                    res_scores.append((doc_idx, doc_score, list(set(matched_terms))))
            return res_scores

        scores = _execute_search(use_category)
        # If filtering by category yielded zero results, gracefully fall back to all categories
        if not scores and use_category:
            scores = _execute_search(None)

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build top-k results
        results = []
        for doc_idx, score, matched in scores[:top_k]:
            chunk_copy = dict(self.chunks[doc_idx])
            results.append({
                "chunk": chunk_copy,
                "score": round(score, 3),
                "matched_terms": matched
            })

        return results
