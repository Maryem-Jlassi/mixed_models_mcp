import sys
import os
import json
import logging
import re
import tempfile
from typing import Dict, Any, List, Tuple
import mcp.types
from fastmcp import FastMCP

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer

# --- Logging ---
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Config ---
SIMILARITY_THRESHOLD_FR = 0.75
SIMILARITY_THRESHOLD_EN = 0.85
TFIDF_MAX_FEATURES = 2000
CHAR_NGRAM_RANGE = (2, 4)

# --- Helper functions ---
def normalize_word(w: str) -> str:
    return re.sub(r'[^\w\-]', '', w).strip()

def clean_ocr_output(text: str) -> str:
    bullet_chars = r"[@•·\*\u2022\u2023\u25E6\u2043\u2219]"
    text = re.sub(bullet_chars, " ", text)
    text = re.sub(r"\s[.,;:!?']\s", " ", text)
    text = re.sub(r"\s[.,;:!?']$", " ", text)
    text = re.sub(r"^[.,;:!?']\s", " ", text)
    text = re.sub(r"[^\w\s.,;:!?'\-\(\)]", " ", text)
    text = re.sub(r"([.,;:!?'\-]){2,}", r"\1", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()

def extract_ocr_text_blocks(pdf_path: str) -> List[str]:
    texts = []
    try:
        images = convert_from_path(pdf_path)
        for img in images:
            page_text = pytesseract.image_to_string(img)
            blocks = re.split(r'\n{1,}|\r{1,}', page_text)
            blocks = [b.strip() for b in blocks if b.strip()]
            texts.extend(blocks)
    except Exception as e:
        logging.error("OCR error: %s", e)
    return texts

def extract_pdfplumber_words(pdf_path: str) -> List[str]:
    words = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    tokens = re.findall(r"\b[\w\-']+\b", page_text)
                    words.extend([normalize_word(t) for t in tokens if normalize_word(t)])
    except Exception as e:
        logging.error("pdfplumber error: %s", e)
    seen = set()
    unique = []
    for w in words:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            unique.append(w)
    return unique

# --- Index classes ---
class WordFaissIndex:
    def __init__(self, words: List[str]):
        self.words = words
        if not words:
            self.index = None
            self.vectorizer = None
            self.dim = 0
            return
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=CHAR_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES
        )
        X = self.vectorizer.fit_transform(words).toarray().astype("float32")
        faiss.normalize_L2(X)
        self.dim = X.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(X)

    def query(self, token: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if self.index is None or self.vectorizer is None:
            return []
        vec = self.vectorizer.transform([token]).toarray().astype("float32")
        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.words) and score > 0:
                results.append((self.words[idx], float(score)))
        return results

class BlockEmbeddingIndex:
    def __init__(self, blocks: List[str]):
        self.blocks = blocks
        if not blocks:
            self.index = None
            self.model = None
            self.embeddings = None
            return
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(blocks, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(self.embeddings)
        self.dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def query(self, block: str) -> Tuple[int, float]:
        if self.index is None or self.model is None:
            return -1, 0.0
        emb = self.model.encode([block], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, 1)
        return int(I[0][0]), float(D[0][0])

# --- Main Corrector ---
class CVTextOCRCorrector:
    def __init__(self):
        self.word_index = None
        self.block_index = None
        self.similarity_threshold = None
        self.lang = None

    def detect_language(self, ocr_blocks: List[str]) -> str:
        try:
            return detect(" ".join(ocr_blocks))
        except Exception:
            return "unknown"

    def build_word_index(self, pdf_path: str):
        self.word_index = WordFaissIndex(extract_pdfplumber_words(pdf_path))

    def build_block_index(self, pdf_path: str):
        blocks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        page_blocks = re.split(r'\n\s*\n', text)
                        blocks.extend([b.strip() for b in page_blocks if b.strip()])
        except Exception as e:
            logging.error(f"pdfplumber extract blocks error: {e}")
        self.block_index = BlockEmbeddingIndex(blocks)

    def correct_ocr_word_level(self, ocr_blocks: List[str]) -> Tuple[str, Dict[str, Any]]:
        corrected_blocks = []
        stats = {"method": "word-level tfidf", "blocks_total": len(ocr_blocks), "blocks_corrected": 0, "replacements": []}

        if not self.word_index or not self.word_index.index:
            return "\n\n".join(ocr_blocks), stats

        for idx, block in enumerate(ocr_blocks):
            tokens = re.findall(r"\b[\w\-']+\b|\S", block)
            replaced_block = False
            corrected_tokens = []
            for token in tokens:
                norm = normalize_word(token)
                if not norm:
                    corrected_tokens.append(token)
                    continue
                candidates = self.word_index.query(norm)
                if candidates:
                    best_word, score = candidates[0]
                    if score >= self.similarity_threshold:
                        replaced_block = True
                        if token.istitle():
                            chosen = best_word.capitalize()
                        elif token.isupper():
                            chosen = best_word.upper()
                        else:
                            chosen = best_word
                        corrected_tokens.append(chosen)
                        stats["replacements"].append({"block_idx": idx, "original": token, "replaced_by": chosen, "score": score})
                        continue
                corrected_tokens.append(token)

            corrected_block = re.sub(r'\s+([.,;:!?])', r'\1', " ".join(corrected_tokens))
            corrected_blocks.append(corrected_block)
            if replaced_block:
                stats["blocks_corrected"] += 1

        return "\n\n".join(corrected_blocks), stats

    def correct_ocr_block_level(self, ocr_blocks: List[str]) -> Tuple[str, Dict[str, Any]]:
        stats = {"method": "block-level embedding", "blocks_total": len(ocr_blocks), "blocks_corrected": 0, "replacements": []}
        if not self.block_index or not self.block_index.index:
            return "\n\n".join(ocr_blocks), stats

        corrected_blocks = []
        for idx, ocr_block in enumerate(ocr_blocks):
            best_i, best_score = self.block_index.query(ocr_block)
            if best_score >= self.similarity_threshold:
                corrected_blocks.append(self.block_index.blocks[best_i])
                stats["blocks_corrected"] += 1
                stats["replacements"].append({"block_idx": idx, "score": best_score})
            else:
                corrected_blocks.append(ocr_block)

        return "\n\n".join(corrected_blocks), stats

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        ocr_blocks = extract_ocr_text_blocks(pdf_path)
        if not ocr_blocks:
            return {"error": "No OCR text extracted", "corrected_text": "", "stats": {}}

        self.lang = self.detect_language(ocr_blocks)
        if self.lang == 'fr':
            self.similarity_threshold = SIMILARITY_THRESHOLD_FR
            self.build_block_index(pdf_path)
            corrected_text, stats = self.correct_ocr_block_level(ocr_blocks)
        elif self.lang == 'en':
            self.similarity_threshold = SIMILARITY_THRESHOLD_EN
            self.build_word_index(pdf_path)
            corrected_text, stats = self.correct_ocr_word_level(ocr_blocks)
        else:
            corrected_text = clean_ocr_output("\n\n".join(ocr_blocks))
            stats = {"method": "none", "blocks_total": len(ocr_blocks), "blocks_corrected": 0, "replacements": []}

        corrected_text = clean_ocr_output(corrected_text)
        return {"corrected_text": corrected_text, "stats": stats, "language": self.lang}

# --- FastMCP setup ---
mcp = FastMCP()

@mcp.tool(annotations={"title": "process_cv_pdf"})
async def process_cv_pdf(file_path: str) -> dict:
    """
    Process a CV PDF file: OCR, language detection, and correction.
    Returns corrected text and stats.
    """
    corrector = CVTextOCRCorrector()
    result = corrector.process_pdf(file_path)

    # Restructure the output to be compatible with the orchestrator
    # The orchestrator expects a 'text' field.
    # We'll also pass along the detailed stats for logging/debugging.
    return {
        "text": result.get("corrected_text", ""),
        "details": {
            "stats": result.get("stats", {}),
            "language": result.get("language", "unknown")
        }
    }

if __name__ == "__main__":
    import traceback
    print("Starting CV OCR Server with FastMCP...", file=sys.stderr)
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Server stopped due to: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)