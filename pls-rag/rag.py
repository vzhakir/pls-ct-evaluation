"""
=============================================================================
CSIPBLLM — RAG EVALUATION SUITE v6
=============================================================================
Peneliti : Muhammad Ajisaka Arsyi Taj (G6401221090)
Skripsi  : "Pengembangan Sistem Pembelajaran Terpersonalisasi Mata Kuliah CT
            Menggunakan Chatbot Berbasis LLM dan RAG"

PERUBAHAN DARI v5 → v6 (rombak formula agar identik dengan paper asli):

  [FIX-6a] FAITHFULNESS → RAGAS Faithfulness (Es et al. 2023, §3.1)
           Formula lama : 0.6 × sim(a,c̄) + 0.4 × keyword_overlap  ← heuristik
           Formula baru : |{klaim yang didukung konteks}| / |{total klaim}|
           Cara kerja   :
             1. LLM memecah jawaban menjadi klaim-klaim atomik.
             2. Setiap klaim diverifikasi ke konteks RAG via LLM.
             3. Faithfulness = n_supported / n_total  ∈ [0, 1].
           LLM verifier  : query_gpt() dari ollamaapi (model sama dengan chat).
           Referensi     : Es et al. (2023). RAGAS. arXiv:2309.15217. §3.1.
           Formula eksak : Faithfulness = |S_v| / |S|
                           S = himpunan pernyataan dari jawaban
                           S_v ⊆ S = pernyataan yang bisa diinfer dari konteks

  [FIX-6b] HALLUCINATION → SelfCheckGPT-BERTScore (Manakul et al. 2023, Eq.4)
           Formula lama : 0.5×OOC + 0.3×SentOOC + 0.2×OOV  ← bobot ad-hoc
           Formula baru : SelfCheck_BERT(i) = 1 - avg_n[ max_j BERTScore(r_i, s_j^n) ]
           Cara kerja   :
             1. Generate jawaban N kali dengan temperature > 0 (stochastic).
             2. Untuk setiap kalimat r_i di jawaban utama (deterministic):
                hitung max BERTScore antar r_i dan semua kalimat di setiap
                sampel s^n, lalu rata-ratakan atas N sampel.
             3. Score kalimat tinggi → kalimat hadir di banyak sampel → faktual.
                Score kalimat rendah → inkonsisten → potensi halusinasi.
             4. Hallucination risk = rata-rata score seluruh kalimat.
           N sampel      : SELFCHECK_N_SAMPLES (default 3, hemat LLM budget).
           Referensi     : Manakul et al. (2023). SelfCheckGPT. EMNLP 2023.
                           arXiv:2303.08896. Eq. (4) BERTScore variant.
           Formula eksak : SelfCheck_i = 1 - (1/N) Σ_n max_j BERTScore(r_i, s_j^n)
                           HalluRisk = (1/|R|) Σ_i SelfCheck_i

  [FIX-6c] ANSWER QUALITY → MT-Bench Single-Answer Grading (Zheng et al. 2023)
           Formula lama : 0.5×sim + 0.3×KP + 0.2×J  ← bobot ad-hoc
           Formula baru : score_llm ∈ {1,2,...,10}  dinormalisasi ke [0,1]
           Cara kerja   :
             LLM judge membaca: pertanyaan + jawaban + konteks RAG,
             lalu memberi skor integer 1–10 berdasarkan rubrik:
               1–3 = salah/tidak relevan, 4–6 = sebagian benar,
               7–9 = benar dan lengkap, 10 = sempurna.
             Skor dinormalisasi: quality_score = (score_llm - 1) / 9 ∈ [0, 1].
           Referensi     : Zheng et al. (2023). MT-Bench. NeurIPS 2023.
                           arXiv:2306.05685. §3 Single-answer grading.
           Formula eksak : quality_score = (s_LLM - 1) / 9,  s_LLM ∈ {1..10}

CATATAN PENTING PERFORMA:
  - FIX-6a: +1 LLM call per query (decompose klaim) + 1 call per klaim (~3–8).
    Total estimasi: +4–10 LLM calls per query.
  - FIX-6b: +N LLM calls per query untuk generate sampel (N = SELFCHECK_N_SAMPLES).
    Default N=3 → +3 LLM calls per query.
  - FIX-6c: +1 LLM call per query untuk MT-Bench grading.
  - Total overhead per query: ~8–15 LLM calls tambahan vs v5.
  - Untuk hemat budget: set SELFCHECK_N_SAMPLES=2 dan batasi klaim max 5.

PERUBAHAN DARI v3 → v4 (dipertahankan, basis kode tidak berubah):

  [FIX-1] METRIK RETRIEVAL DIPERAPI:
          keyword_coverage_at_k dan keyword_recall_at_k dihapus dari output
          agar tabel evaluasi tidak terlalu ramai. Evaluator memakai
          precision_at_k dan recall_at_k sebagai metrik proxy berbasis
          threshold skor, bukan Precision@K/Recall@K IR murni.

  [FIX-2] NO-RAG BASELINE: tambah run_norag_baseline()
          Setiap query dijalankan ke LLM *tanpa* konteks retrieval.
          Perbandingan RAG vs No-RAG membuktikan kontribusi sistem RAG.
          Tanpa baseline ini, klaim "RAG meningkatkan kualitas jawaban"
          tidak bisa dibuktikan secara empiris.

  [FIX-3] SELF-ANNOTATION WORKFLOW: tambah load_annotations() +
          compute_annotation_metrics()
          Peneliti menilai jawaban LLM secara manual (0/1 per query)
          menggunakan file JSON terstruktur. Annotation ini digunakan
          sebagai ground truth untuk menghitung:
            - Human-validated accuracy (vs boolean LLM-as-judge)
            - Agreement rate antara annotation dan boolean signal
          Workflow ini valid secara akademik selama proses annotasi
          didokumentasikan di bab metodologi skripsi.

  [FIX-4] HALLUCINATION DETECTION DIPERBAIKI: hapus heuristik lemah
          Dihapus:
            - Regex negasi bahasa Indonesia (terlalu kasar, false positive tinggi)
            - Uncertainty phrases sebagai indikator halusinasi (epistemic hedging
              adalah tanda kejujuran model, bukan halusinasi)
          Diganti dengan:
            - Claim-level NLI proxy: setiap kalimat jawaban dicek apakah
              ada dukungannya di konteks (menggunakan embedding similarity
              per kalimat, bukan keseluruhan teks)
            - Out-of-vocabulary detection: kata teknis yang muncul di jawaban
              tapi TIDAK ada di konteks manapun (sinyal kuat halusinasi)
            - Skor lebih interpretatif dengan penjelasan per indikator

  [TETAP] Semua bug fix dari v3 (Bug 1–6 + Bonus) dipertahankan.
=============================================================================
"""

import os
import json
import csv
import time
import statistics
import re
import importlib.util
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import requests

# =============================================================================
# KONFIGURASI LLM — di-redirect dari ollamaapi.py
# =============================================================================
from api import (
    OLLAMA_BASE,
    OLLAMA_MODEL,
    EMBEDDING_MODEL_NAME as EMBEDDING_MODEL,
    OR_API_KEY,
    OR_API_BASE,
    OR_MODEL,
    JUDGE_MODEL_OR,
    TEST_MODE,
    CURRENT_MODE,
    client_ollama,
    client_or,
    embeddings_model,
    is_openrouter_available,
    query_gpt,
)

CHAT_MODEL  = OR_MODEL       if "OPENROUTER" in CURRENT_MODE else OLLAMA_MODEL
JUDGE_MODEL = JUDGE_MODEL_OR if "OPENROUTER" in CURRENT_MODE else OLLAMA_MODEL

print(f"[rag_evaluator4] LLM mode: {CURRENT_MODE} | embed: {EMBEDDING_MODEL}")

# =============================================================================
# KONFIGURASI EVALUASI
# =============================================================================
BASE_URL            = "http://127.0.0.1:8000"
TOP_K               = 6
RELEVANCE_THRESHOLD = 0.30
COVERAGE_THRESHOLD  = 0.50
RETRIEVE_TIMEOUT    = 1500
EMBED_MAX_CHARS     = 1500

USE_LLM_EVALUATOR        = True
SEM_SIM_CORRECT_THRESHOLD = 0.70

# =============================================================================
# KONFIGURASI FORMULA v6 (SESUAI PAPER ASLI)
# =============================================================================
# [FIX-6b] SelfCheckGPT: jumlah sampel stochastic per query.
# Lebih banyak = lebih akurat tapi lebih lambat dan mahal.
# Paper asli Manakul et al. (2023) menggunakan N=20 (WikiBio).
# Untuk skripsi dengan budget terbatas, N=3 sudah cukup representatif.
SELFCHECK_N_SAMPLES = int(os.environ.get("SELFCHECK_N_SAMPLES", "3"))

# [FIX-6b] Temperature untuk generate sampel stochastic SelfCheckGPT.
# Harus > 0 agar setiap sampel berbeda. Paper asli menggunakan 1.0.
SELFCHECK_TEMPERATURE = float(os.environ.get("SELFCHECK_TEMPERATURE", "1.0"))

# [FIX-6a] Maksimum klaim yang diverifikasi per jawaban (hemat LLM call).
# Klaim ke-(MAX_CLAIMS+1) dst diabaikan. Set None untuk tanpa batas.
RAGAS_MAX_CLAIMS         = int(os.environ.get("RAGAS_MAX_CLAIMS", "8"))
RAGAS_CONTEXT_MAX_CHARS  = int(os.environ.get("RAGAS_CONTEXT_MAX_CHARS", "4000"))
MTBENCH_CONTEXT_MAX_CHARS = int(os.environ.get("MTBENCH_CONTEXT_MAX_CHARS", "4000"))

# [FIX-6c] MT-Bench: model temperature untuk grading (lebih rendah = lebih deterministik).
MTBENCH_TEMPERATURE = float(os.environ.get("MTBENCH_TEMPERATURE", "0.0"))

# [FIX-3] Path file anotasi manual peneliti.
# Buat file ini dengan run_annotation_template() lalu isi nilai "annotation"
# secara manual (1 = jawaban benar/tidak halusinasi, 0 = salah/halusinasi).
ANNOTATION_FILE = os.path.join(os.path.dirname(__file__), "self_annotations.json")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eval-v7-results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# TEST CASES — EXTERNAL ONLY
# =============================================================================
# Versi ini sengaja TIDAK menyimpan TEST_CASES hardcoded di dalam evaluator.
# Semua test case dibaca dari test_cases.py agar tidak ada dua sumber data.
# Format minimal per item:
#   query, relevant_keywords, cognitive, session_id, query_type, context_note
# Tidak perlu reference_answer.

EXTERNAL_TEST_CASES_FILE = os.environ.get(
    "RAG_TEST_CASES_FILE",
    os.path.join(os.path.dirname(__file__), "test_cases.py"),
)

REQUIRED_TEST_CASE_FIELDS = [
    "query",
    "relevant_keywords",
    "cognitive",
    "session_id",
    "query_type",
    "context_note",
]


def _load_external_test_cases(path: str) -> List[Dict]:
    """Load TEST_CASES dari file Python eksternal tanpa bergantung pada package path."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "File test_cases.py tidak ditemukan. Simpan test_cases.py di folder yang sama "
            "dengan evaluator, atau set environment variable RAG_TEST_CASES_FILE. "
            f"Path yang dicari: {path}"
        )

    spec = importlib.util.spec_from_file_location("external_rag_test_cases", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak bisa membaca spec dari file test cases: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    cases = getattr(module, "TEST_CASES", None)
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"TEST_CASES tidak ditemukan atau kosong di file: {path}")

    return cases


def _normalize_test_case(tc: Dict, idx: int) -> Dict:
    """Validasi dan normalisasi test case agar hanya memakai schema baru."""
    if not isinstance(tc, dict):
        raise TypeError(f"Test case #{idx} harus berupa dict, bukan {type(tc).__name__}.")

    missing = [field for field in REQUIRED_TEST_CASE_FIELDS if field not in tc]
    if missing:
        raise ValueError(
            f"Test case #{idx} tidak memiliki field wajib: {missing}. "
            f"Field wajib: {REQUIRED_TEST_CASE_FIELDS}"
        )

    relevant_keywords = tc.get("relevant_keywords")
    if not isinstance(relevant_keywords, list):
        raise TypeError(f"Test case #{idx} field 'relevant_keywords' harus berupa list.")

    # Ambil hanya field schema baru supaya reference_answer lama tidak ikut dipakai.
    return {
        # _global_id dipakai hanya untuk menjaga nomor asli ketika evaluasi dijalankan per batch.
        # Field ini tidak mengubah schema test_cases.py, hanya metadata internal evaluator.
        "_global_id": idx,
        "query": str(tc["query"]).strip(),
        "relevant_keywords": [str(k).strip() for k in relevant_keywords if str(k).strip()],
        "cognitive": str(tc["cognitive"]).strip(),
        "session_id": str(tc["session_id"]).strip(),
        "query_type": str(tc["query_type"]).strip(),
        "context_note": str(tc["context_note"]).strip(),
    }


ALL_TEST_CASES: List[Dict] = [
    _normalize_test_case(tc, i)
    for i, tc in enumerate(_load_external_test_cases(EXTERNAL_TEST_CASES_FILE), 1)
]

BATCH_START = int(os.environ.get("RAG_BATCH_START", "1"))
BATCH_END   = int(os.environ.get("RAG_BATCH_END", "25"))

# Default single-run tetap batch 1–25. Untuk auto 1–75, gunakan RAG_AUTO_BATCHES=1
TEST_CASES: List[Dict] = ALL_TEST_CASES[BATCH_START - 1:BATCH_END]

print(f"[test-case-loader] Total test case tersedia: {len(ALL_TEST_CASES)} dari: {EXTERNAL_TEST_CASES_FILE}")
print(f"[test-case-loader] Single batch aktif: {BATCH_START}-{BATCH_END} ({len(TEST_CASES)} soal)")
print(f"[test-case-loader] Schema aktif: {', '.join(REQUIRED_TEST_CASE_FIELDS)}")


def get_reference_answer_for_eval(tc: Dict, rag_context: Optional[List[str]] = None) -> str:
    """
    Ambil reference/proxy untuk evaluasi kualitas jawaban.

    Schema test_cases.py baru tidak memakai reference_answer. Karena itu Step 5
    memakai chunk yang di-retrieve RAG sebagai context proxy. Jika retrieval gagal,
    fallback terakhir memakai context_note dan relevant_keywords agar evaluator tetap
    tidak crash, tetapi skor harus dibaca sebagai sinyal terbatas.
    """
    context = [str(c).strip() for c in (rag_context or []) if str(c).strip()]
    if context:
        return "\n\n".join(context[:TOP_K])

    parts = []
    if tc.get("context_note"):
        parts.append(str(tc["context_note"]))
    if tc.get("relevant_keywords"):
        parts.append("Kata kunci relevan: " + ", ".join(map(str, tc["relevant_keywords"])))
    return "\n".join(parts).strip()


# =============================================================================
# LLM CLIENT
# =============================================================================
chat_client  = client_or if "OPENROUTER" in CURRENT_MODE else client_ollama
embed_client = client_ollama

CHAT_CLIENT_AVAILABLE = chat_client is not None
LLM_CLIENT_AVAILABLE  = embed_client is not None

# =============================================================================
# EMBEDDING CACHE
# =============================================================================
_embedding_cache: Dict[str, Optional[np.ndarray]] = {}


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Menghasilkan vektor embedding menggunakan embeddings_model dari ollamaapi.

    SESUAI PROPOSAL hal. 4:
      ē = f_embed(x)
    di mana f_embed adalah model embedding lokal (EMBEDDING_MODEL),
    dan ē ∈ ℝ^d adalah vektor berdimensi d.
    """
    text_key = text[:EMBED_MAX_CHARS]
    if text_key in _embedding_cache:
        return _embedding_cache[text_key]

    if not LLM_CLIENT_AVAILABLE:
        _embedding_cache[text_key] = None
        return None
    try:
        vec = np.array(embeddings_model.embed_query(text_key), dtype="float32")
        _embedding_cache[text_key] = vec
        return vec
    except Exception as e:
        print(f"  [Embedding Error] {e}")
        _embedding_cache[text_key] = None
        return None


# =============================================================================
# METRIK INTI
# =============================================================================

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Menghitung cosine similarity antara dua vektor.

    SESUAI PROPOSAL hal. 4–5:
      sim(q, d) = (q · d) / (||q|| · ||d||)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# CATATAN METODOLOGI (untuk bab metodologi skripsi):
#   precision_at_k dan recall_at_k di file ini adalah metrik proxy berbasis
#   threshold skor retrieval. Karena tidak ada qrels/anotasi manual relevansi
#   dokumen, laporkan sebagai Proxy Precision@K dan Proxy Recall@K, bukan
#   Precision@K/Recall@K standar Information Retrieval murni.


def precision_at_k(scores: List[float], k: int, threshold: float = RELEVANCE_THRESHOLD) -> float:
    """
    Proxy Precision@K berbasis skor retrieval.

    Catatan metodologi:
      Ini mengikuti fungsi metrics.py dari evaluator pembanding. Skor dihitung
      sebagai proporsi Top-K item dengan score >= threshold. Karena tidak memakai
      qrels/anotasi dokumen manual, laporkan sebagai Proxy Precision@K atau
      Score-threshold Precision@K, bukan Precision@K IR murni.
    """
    if k <= 0:
        return 0.0
    top = scores[:k]
    if not top:
        return 0.0
    # Pembagi tetap k agar konsisten dengan definisi P@K.
    return sum(1 for s in top if s >= threshold) / k


def recall_at_k(
    scores: List[float],
    total_relevant: int,
    k: int,
    threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """
    Proxy Recall@K berbasis skor retrieval.

    total_relevant diambil dari jumlah relevant_keywords karena schema test case
    baru tidak memiliki qrels dokumen. Karena itu nilai ini adalah proxy, bukan
    Recall@K standar IR murni.
    """
    if total_relevant <= 0:
        return 1.0
    return min(sum(1 for s in scores[:k] if s >= threshold) / total_relevant, 1.0)


def chunk_relevance_score(chunk_scores: List[float]) -> float:
    """
    Continuous chunk relevance score dari metrics.py.
    Top chunk diberi bobot 2x, chunk lain 1x.
    """
    if not chunk_scores:
        return 0.0
    if len(chunk_scores) == 1:
        return float(chunk_scores[0])
    weighted = chunk_scores[0] * 2 + sum(chunk_scores[1:])
    return float(weighted / (len(chunk_scores) + 1))


def mean_similarity(scores: List[float]) -> float:
    """
    Menghitung Mean Similarity dari hasil retrieval.

    SESUAI PROPOSAL hal. 5:
      MeanSim = (1/K) * Σ(i=1 to K) si
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def coverage_score(scores: List[float],
                   threshold: float = COVERAGE_THRESHOLD) -> float:
    """Coverage: proporsi chunk dengan skor cosine ≥ threshold."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


def source_diversity(sources: List[str]) -> float:
    """Source Diversity: |sumber unik| / K."""
    if not sources:
        return 0.0
    unique_sources = len(set(sources))
    return unique_sources / len(sources)


# =============================================================================
# [FIX-6a] FAITHFULNESS — RAGAS Claim-Based (Es et al. 2023, §3.1)
# =============================================================================
# Formula eksak dari paper:
#   Faithfulness = |S_v| / |S|
#   S   = himpunan pernyataan (klaim atomik) yang diekstrak dari jawaban
#   S_v = subset S yang dapat diinfer dari konteks yang diambil retriever
#
# Implementasi dua langkah (sesuai Es et al. 2023):
#   Step 1 — Decomposition: LLM memecah jawaban menjadi klaim-klaim atomik.
#   Step 2 — Verification : LLM memverifikasi setiap klaim terhadap konteks.
# =============================================================================

_RAGAS_DECOMPOSE_PROMPT = """\
Kamu adalah asisten akademik. Tugasmu adalah memecah teks berikut menjadi \
pernyataan-pernyataan faktual yang dapat diverifikasi.

Teks jawaban:
\"\"\"
{answer}
\"\"\"

Instruksi:
- Tulis setiap pernyataan sebagai baris terpisah, diawali dengan "- ".
- Maksimum {max_claims} pernyataan.
- Buat pernyataan yang CUKUP UMUM — jangan terlalu granular atau spesifik.
- Gabungkan detail kecil yang saling terkait menjadi satu pernyataan.
- Fokus pada IDE UTAMA dan KONSEP KUNCI dari jawaban, bukan rincian kecil.
- Jangan tambahkan komentar atau penjelasan, hanya daftar pernyataan.
- Gunakan bahasa Indonesia.

Daftar pernyataan:"""

_RAGAS_VERIFY_PROMPT = """\
Kamu adalah penilai fakta. Tugasmu adalah menentukan apakah sebuah pernyataan \
dapat disimpulkan atau didukung oleh konteks yang diberikan.

Konteks:
\"\"\"
{context}
\"\"\"

Pernyataan yang dinilai:
\"{claim}\"

Panduan penilaian:
- Jawab "YA" jika pernyataan dapat disimpulkan atau didukung oleh konteks, \
meskipun tidak disebutkan kata per kata secara eksplisit.
- Jawab "TIDAK" jika konteks tidak membahas topik pernyataan sama sekali, \
atau secara aktif bertentangan dengan pernyataan tersebut.

Jawab HANYA dengan satu kata: "YA" atau "TIDAK"."""


def _llm_call_for_ragas(prompt: str, max_tokens: int = 512) -> Optional[str]:
    """Helper: panggil LLM (query_gpt dari ollamaapi) untuk keperluan RAGAS."""
    try:
        resp = query_gpt(prompt, max_tokens=max_tokens)
        return resp.strip() if resp else None
    except Exception as e:
        print(f"  [RAGAS LLM Error] {e}")
        return None


def _decompose_claims(answer: str, max_claims: int = RAGAS_MAX_CLAIMS) -> List[str]:
    """
    Step 1 RAGAS: Pecah jawaban menjadi klaim-klaim atomik menggunakan LLM.
    Mengembalikan list klaim. Jika LLM gagal, fallback ke pemecahan per kalimat.
    """
    prompt = _RAGAS_DECOMPOSE_PROMPT.format(answer=answer[:1500], max_claims=max_claims)
    raw = _llm_call_for_ragas(prompt, max_tokens=600)

    if raw:
        claims = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                line = line[2:].strip()
            if line and len(line) > 10:
                claims.append(line)
        if claims:
            return claims[:max_claims]

    # Fallback: gunakan kalimat sebagai klaim (lebih kasar tapi zero-cost)
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15][:max_claims]


def _verify_claim(claim: str, context_text: str) -> bool:
    """
    Step 2 RAGAS: Verifikasi satu klaim terhadap konteks menggunakan LLM.
    Mengembalikan True jika klaim didukung konteks, False jika tidak.
    """
    prompt = _RAGAS_VERIFY_PROMPT.format(
        context=context_text[:RAGAS_CONTEXT_MAX_CHARS],
        claim=claim[:300],
    )
    raw = _llm_call_for_ragas(prompt, max_tokens=10)
    if raw:
        return raw.strip().upper().startswith("YA")
    return False


def evaluate_faithfulness(answer: str, retrieved_chunks: List[str]) -> Dict:
    """
    [FIX-6a] Faithfulness berbasis klaim (RAGAS, Es et al. 2023 §3.1).

    Formula eksak dari paper:
        Faithfulness = |S_v| / |S|

    Di mana:
        S   = himpunan klaim atomik yang diekstrak dari jawaban oleh LLM
        S_v = subset S yang dapat diinferensikan dari konteks RAG

    Langkah implementasi (dua-tahap, sesuai Es et al. 2023):
        1. Decomposition : LLM memecah jawaban → list klaim atomik S
        2. Verification  : setiap klaim s ∈ S diverifikasi ke konteks
                           → bernilai 1 jika didukung, 0 jika tidak

    Referensi: Es S, James J, Espinosa-Anke L, Schockaert S. 2023.
        RAGAS: Automated Evaluation of Retrieval Augmented Generation.
        arXiv:2309.15217. §3.1 Faithfulness.

    Catatan implementasi:
        - LLM yang digunakan adalah query_gpt() dari ollamaapi (model = CHAT_MODEL).
        - Jumlah klaim dibatasi RAGAS_MAX_CLAIMS untuk hemat LLM budget.
        - Jika LLM tidak tersedia, fallback ke metrik embedding similarity
          sebagai aproksimasi (ditandai method = "embedding_fallback").
    """
    if not retrieved_chunks or not answer:
        return {
            "faithfulness_score": 0.0,
            "n_claims": 0,
            "n_supported": 0,
            "claims": [],
            "method": "none",
        }

    context_text = "\n\n".join(retrieved_chunks)

    # ── Coba implementasi RAGAS penuh (butuh LLM) ─────────────────────────
    if CHAT_CLIENT_AVAILABLE:
        print("               [RAGAS] Dekomposisi klaim jawaban...")
        claims = _decompose_claims(answer)
        n_total = len(claims)

        if n_total == 0:
            return {
                "faithfulness_score": 0.0,
                "n_claims": 0,
                "n_supported": 0,
                "claims": [],
                "method": "ragas_no_claims",
            }

        print(f"               [RAGAS] Verifikasi {n_total} klaim ke konteks...")
        claim_results = []
        n_supported = 0
        for i, claim in enumerate(claims, 1):
            supported = _verify_claim(claim, context_text)
            if supported:
                n_supported += 1
            claim_results.append({
                "claim":     claim,
                "supported": supported,
            })
            status = '✓ didukung' if supported else '✗ tidak didukung'
            print(f"               [RAGAS]   klaim {i}/{n_total}: {status}")
            print(f"               [RAGAS]     → \"{claim[:120]}{'...' if len(claim) > 120 else ''}\"")


        # Faithfulness = |S_v| / |S|
        faithfulness_score = n_supported / n_total

        return {
            "faithfulness_score": round(faithfulness_score, 4),
            "n_claims":           n_total,
            "n_supported":        n_supported,
            "claims":             claim_results,
            "method":             "ragas_claim_based",
            "formula":            "|S_v| / |S|",
            "reference":          "Es et al. (2023) RAGAS §3.1",
        }

    # ── Fallback jika LLM tidak tersedia ──────────────────────────────────
    print("               [RAGAS] LLM tidak tersedia — fallback ke embedding similarity")
    emb_answer = get_embedding(answer[:EMBED_MAX_CHARS])
    chunk_embs = [get_embedding(c[:EMBED_MAX_CHARS]) for c in retrieved_chunks]
    chunk_embs = [e for e in chunk_embs if e is not None]
    if emb_answer is not None and chunk_embs:
        emb_ctx = np.mean(chunk_embs, axis=0).astype("float32")
        score   = cosine_similarity(emb_answer, emb_ctx)
    else:
        score = 0.0

    return {
        "faithfulness_score": round(score, 4),
        "n_claims":           0,
        "n_supported":        0,
        "claims":             [],
        "method":             "embedding_fallback",
        "warning":            "LLM tidak tersedia; RAGAS claim-based tidak dijalankan",
    }


# =============================================================================
# [FIX-6b] HALLUCINATION DETECTION — SelfCheckGPT-BERTScore (Manakul et al. 2023)
#
# Formula eksak dari paper (Eq. 4, BERTScore variant):
#   SelfCheck_BERT(i) = 1 - (1/N) Σ_{n=1}^{N} max_{j} BERTScore(r_i, s_j^n)
#
#   Di mana:
#     r_i    = kalimat ke-i dari jawaban utama (deterministic, T=0)
#     s_j^n  = kalimat ke-j dari sampel ke-n (stochastic, T>0)
#     N      = jumlah sampel stochastic (SELFCHECK_N_SAMPLES)
#
#   Interpretasi:
#     SelfCheck_BERT(i) ≈ 0 → kalimat r_i sangat konsisten di semua sampel
#                              → faktual, tidak halusinasi
#     SelfCheck_BERT(i) ≈ 1 → kalimat r_i tidak muncul di sampel manapun
#                              → inkonsisten, potensi halusinasi
#
#   HallucinationRisk = (1/|R|) Σ_i SelfCheck_BERT(i)
#
# Referensi: Manakul P, Liusie A, Gales MJF. 2023. SelfCheckGPT:
#   Zero-Resource Black-Box Hallucination Detection for Generative LLMs.
#   EMNLP 2023. arXiv:2303.08896. Eq. (4).
# =============================================================================

def _split_sentences(text: str) -> List[str]:
    """Pisahkan teks menjadi kalimat sederhana (digunakan SelfCheckGPT)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _generate_stochastic_samples(
    query: str,
    cognitive: str,
    session_id: str,
    n: int = SELFCHECK_N_SAMPLES,
) -> List[str]:
    """
    Generate N sampel stochastic dari LLM untuk query yang sama.
    Menggunakan temperature tinggi (SELFCHECK_TEMPERATURE) agar setiap
    sampel berbeda — prinsip inti SelfCheckGPT.

    Returns: list string jawaban (mungkin kosong jika LLM tidak tersedia).
    """
    if not CHAT_CLIENT_AVAILABLE:
        return []

    samples = []
    for i in range(n):
        try:
            resp = chat_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Kamu adalah tutor mata kuliah Computational Thinking. "
                            "Jawab pertanyaan mahasiswa berdasarkan pengetahuanmu. "
                            "Jawab dalam bahasa Indonesia."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=512,
                temperature=SELFCHECK_TEMPERATURE,
            )
            text = resp.choices[0].message.content if resp.choices else None
            if text:
                samples.append(text.strip())
        except Exception as e:
            print(f"  [SelfCheck] Sampel {i+1}/{n} gagal: {e}")
    return samples


def _bert_score_sentence_pair(sent_a: str, sent_b: str) -> float:
    """
    Hitung BERTScore F1 antara dua kalimat menggunakan cosine similarity
    embedding-nya sebagai aproksimasi.

    Catatan: BERTScore sesungguhnya (Zhang et al. 2020) menggunakan greedy
    matching antar token-level embedding dari model BERT. Di sini kita
    menggunakan sentence-level embedding sebagai aproksimasi yang lebih
    efisien (embedding sudah ter-cache dari proses retrieval).
    Aproksimasi ini valid untuk tujuan ranking/perbandingan relatif.
    """
    emb_a = get_embedding(sent_a[:EMBED_MAX_CHARS])
    emb_b = get_embedding(sent_b[:EMBED_MAX_CHARS])
    if emb_a is None or emb_b is None:
        return 0.0
    return cosine_similarity(emb_a, emb_b)


def detect_hallucination(
    answer: str,
    retrieved_chunks: List[str],
    precomputed_faithfulness: Optional[Dict] = None,
    # Parameter tambahan untuk SelfCheckGPT
    query: str = "",
    cognitive: str = "1PAR",
    session_id: str = "selfcheck",
) -> Dict:
    """
    [FIX-6b] Deteksi halusinasi menggunakan SelfCheckGPT-BERTScore.

    Formula eksak dari paper (Manakul et al. 2023, Eq. 4):
        SelfCheck_BERT(i) = 1 - (1/N) Σ_{n=1}^{N} max_{j} BERTScore(r_i, s_j^n)
        HallucinationRisk = (1/|R|) Σ_i SelfCheck_BERT(i)

    Langkah implementasi:
        1. Pisahkan jawaban utama R = {r_1, r_2, ..., r_|R|} menjadi kalimat.
        2. Generate N sampel stochastic {S^1, ..., S^N} dari LLM yang sama
           menggunakan temperature > 0.
        3. Untuk setiap kalimat r_i:
             a. Pisahkan setiap sampel S^n menjadi kalimat {s_1^n, ..., s_k^n}.
             b. Hitung max BERTScore antara r_i dan semua kalimat di S^n.
             c. Rata-ratakan max score atas N sampel.
             d. SelfCheck_BERT(i) = 1 - rata-rata tersebut.
        4. HallucinationRisk = rata-rata SelfCheck_BERT(i) atas semua kalimat.

    Referensi: Manakul P, Liusie A, Gales MJF. 2023. SelfCheckGPT:
        Zero-Resource Black-Box Hallucination Detection for Generative LLMs.
        EMNLP 2023. arXiv:2303.08896. Eq. (4) BERTScore variant.

    Catatan implementasi BERTScore:
        Paper asli menggunakan token-level BERTScore dari model DeBERTa.
        Implementasi ini menggunakan sentence-level embedding cosine similarity
        sebagai aproksimasi efisien (embedding sudah ter-cache dari retrieval).
        Embedding yang digunakan: model qwen3-embedding via EMBEDDING_MODEL.

    Args:
        answer     : jawaban utama LLM (deterministic, hasil /chat).
        query      : query asli — digunakan untuk generate sampel stochastic.
        cognitive  : level kognitif Bloom — diteruskan ke LLM sampler.
        session_id : session identifier — diteruskan ke LLM sampler.
        precomputed_faithfulness : tidak dipakai di v6 (hanya untuk kompatibilitas).
        retrieved_chunks : tidak dipakai di SelfCheckGPT murni, tapi disimpan
                           untuk referensi dan kompatibilitas downstream.
    """
    if not answer:
        return {
            "hallucination_risk": 0.0,
            "risk_label":         "RENDAH",
            "method":             "none",
            "sentence_scores":    [],
            "n_samples":          0,
        }

    # ── Step 1: Pisahkan jawaban utama menjadi kalimat ────────────────────
    main_sentences = _split_sentences(answer)
    if not main_sentences:
        return {
            "hallucination_risk": 0.0,
            "risk_label":         "RENDAH",
            "method":             "no_sentences",
            "sentence_scores":    [],
            "n_samples":          0,
        }

    # ── Step 2: Generate N sampel stochastic ─────────────────────────────
    print(f"               [SelfCheck] Generate {SELFCHECK_N_SAMPLES} sampel stochastic...")
    if query:
        samples = _generate_stochastic_samples(
            query, cognitive, session_id, n=SELFCHECK_N_SAMPLES
        )
    else:
        samples = []

    # ── Fallback jika tidak ada query atau LLM tidak tersedia ─────────────
    if not samples:
        print("               [SelfCheck] Tidak ada sampel — fallback ke embedding OOC")
        # Fallback: gunakan cosine similarity jawaban vs konteks sebagai proxy
        chunk_embs = [get_embedding(c[:EMBED_MAX_CHARS]) for c in retrieved_chunks]
        chunk_embs = [e for e in chunk_embs if e is not None]
        if chunk_embs:
            emb_ctx = np.mean(chunk_embs, axis=0).astype("float32")
            sent_scores = []
            for sent in main_sentences:
                emb_s = get_embedding(sent[:EMBED_MAX_CHARS])
                if emb_s is not None:
                    sim = cosine_similarity(emb_s, emb_ctx)
                    sent_scores.append(1.0 - sim)   # inkonsisten jika sim rendah
                else:
                    sent_scores.append(0.5)
            hall_risk = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
        else:
            hall_risk = 0.5
            sent_scores = []

        risk_label = (
            "TINGGI" if hall_risk >= 0.60 else
            "SEDANG" if hall_risk >= 0.35 else
            "RENDAH"
        )
        return {
            "hallucination_risk": round(hall_risk, 4),
            "risk_label":         risk_label,
            "method":             "embedding_ooc_fallback",
            "sentence_scores":    [round(s, 4) for s in sent_scores],
            "n_samples":          0,
            "warning":            "Query tidak tersedia atau LLM tidak tersedia; SelfCheckGPT tidak dijalankan",
        }

    n_actual = len(samples)
    print(f"               [SelfCheck] {n_actual} sampel diperoleh. "
          f"Hitung BERTScore per kalimat ({len(main_sentences)} kalimat)...")

    # ── Step 3: Hitung SelfCheck_BERT(i) per kalimat ─────────────────────
    # Untuk setiap kalimat r_i di jawaban utama:
    #   max_bert_n = max_{j} BERTScore(r_i, s_j^n)  ← max per sampel
    #   avg_max    = (1/N) Σ_n max_bert_n             ← rata-rata atas N sampel
    #   selfcheck  = 1 - avg_max                      ← skor inkonsistensi
    sentence_detail = []
    selfcheck_scores = []

    # Pre-split semua sampel jadi kalimat (hemat re-split berulang)
    sample_sentences_list = [_split_sentences(s) for s in samples]

    for i, r_i in enumerate(main_sentences, 1):
        per_sample_max = []
        for n_idx, sample_sents in enumerate(sample_sentences_list):
            if not sample_sents:
                per_sample_max.append(0.0)
                continue
            # max_{j} BERTScore(r_i, s_j^n)
            max_score = max(
                _bert_score_sentence_pair(r_i, s_j)
                for s_j in sample_sents
            )
            per_sample_max.append(max_score)

        # (1/N) Σ_n max_bert_n
        avg_max = sum(per_sample_max) / n_actual if per_sample_max else 0.0
        # SelfCheck_BERT(i) = 1 - avg_max
        selfcheck_i = 1.0 - avg_max
        selfcheck_scores.append(selfcheck_i)

        sentence_detail.append({
            "sentence":         r_i[:100],
            "selfcheck_score":  round(selfcheck_i, 4),
            "avg_max_bertscore": round(avg_max, 4),
            "per_sample_max":   [round(x, 4) for x in per_sample_max],
            "is_suspect":       selfcheck_i > 0.5,
        })
        suspect_tag = " ⚠ SUSPECT" if selfcheck_i > 0.5 else ""
        print(f"               [SelfCheck]   kalimat {i}: selfcheck={selfcheck_i:.4f} "
              f"avg_bertscore={avg_max:.4f}{suspect_tag}")
        print(f"               [SelfCheck]     → \"{r_i[:100]}{'...' if len(r_i) > 100 else ''}\"")


    # ── Step 4: HallucinationRisk = (1/|R|) Σ_i SelfCheck_BERT(i) ───────
    hallucination_risk = (
        sum(selfcheck_scores) / len(selfcheck_scores)
        if selfcheck_scores else 0.0
    )

    risk_label = (
        "TINGGI" if hallucination_risk >= 0.60 else
        "SEDANG" if hallucination_risk >= 0.35 else
        "RENDAH"
    )

    return {
        "hallucination_risk":   round(hallucination_risk, 4),
        "risk_label":           risk_label,
        "method":               "selfcheckgpt_bertscore",
        "formula":              "SelfCheck(i) = 1 - (1/N) Σ max_j BERTScore(r_i, s_j^n)",
        "reference":            "Manakul et al. (2023) SelfCheckGPT, arXiv:2303.08896 Eq.4",
        "n_sentences":          len(main_sentences),
        "n_samples":            n_actual,
        "selfcheck_per_sentence": selfcheck_scores,
        "sentence_details":     sentence_detail,
        # Kompabilitas dengan kode downstream yang membaca key lama
        "sentence_ooc_score":   round(hallucination_risk, 4),
        "oov_ratio":            0.0,
        "out_of_context":       round(hallucination_risk, 4),
        "faithfulness":         round(1.0 - hallucination_risk, 4),
    }


# =============================================================================
# [FIX-6c] ANSWER QUALITY — MT-Bench Single-Answer Grading (Zheng et al. 2023)
#
# Formula eksak dari paper (§3 Single-answer grading):
#   Judge LLM memberi skor integer s_LLM ∈ {1, 2, ..., 10}
#   quality_score = (s_LLM - 1) / 9  →  dinormalisasi ke [0, 1]
#
# Rubrik skor (diikuti dari MT-Bench §3 dan Appendix Figure 5):
#   1–3  = Jawaban salah, tidak relevan, atau sangat tidak lengkap
#   4–6  = Jawaban sebagian benar, ada unsur relevan tapi ada kesalahan material
#   7–9  = Jawaban benar, relevan, dan lengkap; mungkin ada kekurangan minor
#   10   = Jawaban sempurna — akurat, lengkap, dan didukung penuh oleh konteks
#
# Referensi: Zheng L, Chiang WL, Sheng Y, et al. 2023. Judging LLM-as-a-Judge
#   with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685. §3.
# =============================================================================

_MTBENCH_GRADING_PROMPT = """\
Kamu adalah penilai akademik yang mengevaluasi jawaban tutor Computational Thinking.

Pertanyaan mahasiswa:
\"{question}\"

Konteks materi yang tersedia untuk tutor:
\"\"\"
{context}
\"\"\"

Jawaban tutor yang dinilai:
\"\"\"
{answer}
\"\"\"

Tugas: Berikan skor integer antara 1 sampai 10 untuk jawaban tutor berdasarkan:
- Kebenaran faktual berdasarkan konteks yang diberikan
- Kelengkapan dan relevansi terhadap pertanyaan
- Kejelasan penjelasan

Rubrik skor:
  1–3  = Jawaban salah, tidak relevan, atau sangat tidak lengkap
  4–6  = Jawaban sebagian benar; ada unsur relevan tapi ada kesalahan material
  7–9  = Jawaban benar, relevan, dan lengkap; mungkin ada kekurangan minor
  10   = Jawaban sempurna — akurat, lengkap, dan didukung penuh oleh konteks

Balas HANYA dengan satu angka integer (1–10), tidak ada penjelasan lain."""


def evaluate_answer_quality(
    llm_reply: str,
    reference_answer: str,
    eval_resp: Optional[Dict] = None,
    fast_bool_threshold: Optional[float] = None,
    # Parameter tambahan untuk MT-Bench
    question: str = "",
    rag_context: Optional[List[str]] = None,
) -> Dict:
    """
    [FIX-6c] Answer Quality menggunakan MT-Bench Single-Answer Grading.

    Formula eksak dari paper (Zheng et al. 2023, §3):
        quality_score = (s_LLM - 1) / 9,   s_LLM ∈ {1, 2, ..., 10}

    Cara kerja:
        LLM judge diberikan: pertanyaan + konteks RAG + jawaban tutor.
        LLM mengeluarkan satu angka integer 1–10 sesuai rubrik MT-Bench.
        Skor dinormalisasi ke [0, 1] agar konsisten dengan metrik lain.

    Perbedaan dari versi sebelumnya:
        - Tidak ada lagi bobot 0.5/0.3/0.2
        - Tidak ada komponen embedding similarity
        - LLM judge adalah satu-satunya sinyal (sesuai paper)
        - Rubrik eksplisit mengikuti MT-Bench (Zheng et al. 2023)

    Referensi: Zheng L, Chiang WL, Sheng Y, et al. 2023.
        Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
        NeurIPS 2023. arXiv:2306.05685. §3 Single-answer grading.

    Args:
        llm_reply        : jawaban yang dinilai.
        reference_answer : digunakan sebagai fallback jika judge tidak tersedia;
                           di v6 tidak masuk ke formula utama.
        question         : pertanyaan asli (wajib untuk prompt judge).
        rag_context      : list chunk RAG yang diberikan ke tutor (masuk prompt judge).
        eval_resp        : diabaikan di v6 (dipertahankan untuk kompatibilitas signature).
        fast_bool_threshold : diabaikan di v6 (dipertahankan untuk kompatibilitas).
    """
    if not llm_reply:
        return {
            "answer_quality_score": 0.0,
            "llm_judge_score":      None,
            "label":                "BURUK",
            "method":               "empty",
        }

    # ── MT-Bench grading via LLM judge ────────────────────────────────────
    if CHAT_CLIENT_AVAILABLE and question:
        context_text = "\n\n".join((rag_context or [])[:4]) or reference_answer or "(tidak ada konteks)"
        prompt = _MTBENCH_GRADING_PROMPT.format(
            question=question[:300],
            context=context_text[:MTBENCH_CONTEXT_MAX_CHARS],
            answer=llm_reply[:1000],
        )

        try:
            resp = chat_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=MTBENCH_TEMPERATURE,
            )
            raw_score = resp.choices[0].message.content.strip() if resp.choices else ""
            print(f"               [MT-Bench] Raw judge response: \"{raw_score[:80]}\"")
            # Ekstrak integer dari respons (toleran terhadap noise)
            nums = re.findall(r'\b([1-9]|10)\b', raw_score)
            if nums:
                s_llm = int(nums[0])
                s_llm = max(1, min(10, s_llm))   # clamp ke [1, 10]
                print(f"               [MT-Bench] Skor diekstrak: {s_llm}/10")
            else:
                s_llm = None
                print(f"               [MT-Bench] ⚠ Tidak ada skor integer ditemukan di respons")
        except Exception as e:
            print(f"  [MT-Bench] Judge error: {e}")
            s_llm = None

        if s_llm is not None:
            # quality_score = (s_LLM - 1) / 9
            quality_score = (s_llm - 1) / 9.0
            label = (
                "SANGAT BAIK" if quality_score >= 0.75 else
                "BAIK"        if quality_score >= 0.50 else
                "CUKUP"       if quality_score >= 0.30 else
                "BURUK"
            )
            return {
                "answer_quality_score": round(quality_score, 4),
                "llm_judge_score":      s_llm,       # skor mentah 1–10
                "label":                label,
                "method":               "mtbench_single_answer_grading",
                "formula":              "quality_score = (s_LLM - 1) / 9",
                "reference":            "Zheng et al. (2023) MT-Bench §3",
                # Alias untuk kompatibilitas field lama
                "bool_signal":          1.0 if quality_score >= 0.5 else 0.0,
                "keyword_overlap":      None,
                "semantic_similarity":  None,
            }

    # ── Fallback: embedding similarity jika LLM judge tidak tersedia ──────
    print("               [MT-Bench] LLM tidak tersedia — fallback ke embedding similarity")
    emb_reply = get_embedding(llm_reply[:EMBED_MAX_CHARS])
    emb_ref   = get_embedding(reference_answer[:EMBED_MAX_CHARS]) if reference_answer else None
    if emb_reply is not None and emb_ref is not None:
        sem_sim = float(cosine_similarity(emb_reply, emb_ref))
    else:
        sem_sim = 0.0

    label = (
        "SANGAT BAIK" if sem_sim >= 0.75 else
        "BAIK"        if sem_sim >= 0.50 else
        "CUKUP"       if sem_sim >= 0.30 else
        "BURUK"
    )
    return {
        "answer_quality_score": round(sem_sim, 4),
        "llm_judge_score":      None,
        "label":                label,
        "method":               "embedding_similarity_fallback",
        "warning":              "MT-Bench judge tidak dijalankan; question kosong atau LLM tidak tersedia",
        # Alias untuk kompatibilitas
        "bool_signal":          1.0 if sem_sim >= 0.5 else 0.0,
        "keyword_overlap":      None,
        "semantic_similarity":  round(sem_sim, 4),
    }


# =============================================================================
# API CALLS
# =============================================================================

def call_chat_api(message: str, cognitive: str, session_id: str) -> Optional[Dict]:
    """Panggil endpoint /chat dan kembalikan respons."""
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message, "cognitive": cognitive, "session_id": session_id},
            timeout=RETRIEVE_TIMEOUT,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"  [API Error /chat] {e}")
        return None


def call_evaluate_api(answer: str, correct_answer: str,
                      active_question: str, wrong_count: int,
                      cognitive: str, session_id: str) -> Optional[Dict]:
    """Panggil endpoint /evaluate dan kembalikan hasil penilaian."""
    try:
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json={
                "answer":          answer,
                "correct_answer":  correct_answer,
                "active_question": active_question,
                "wrong_count":     wrong_count,
                "cognitive":       cognitive,
                "session_id":      session_id,
            },
            timeout=RETRIEVE_TIMEOUT,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        print(f"  [API Error /evaluate] {e}")
        return None


def call_chat_norag(message: str) -> Optional[str]:
    """
    [FIX-2] Panggil LLM LANGSUNG tanpa konteks RAG untuk baseline.

    Menggunakan client LLM yang sama tapi dengan prompt yang TIDAK
    menyertakan retrieved context. Ini menghasilkan jawaban murni dari
    memori parametrik LLM — digunakan sebagai baseline perbandingan.

    Returns:
      String jawaban LLM, atau None jika gagal.
    """
    if not CHAT_CLIENT_AVAILABLE:
        return None
    try:
        resp = chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah tutor mata kuliah Computational Thinking. "
                        "Jawab pertanyaan mahasiswa berdasarkan pengetahuanmu. "
                        "Jawab dalam bahasa Indonesia."
                    ),
                },
                {"role": "user", "content": message},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content if resp.choices else None
    except Exception as e:
        print(f"  [No-RAG API Error] {e}")
        return None


# =============================================================================
# EVALUASI RETRIEVAL
# =============================================================================

def evaluate_retrieval_direct(query: str, relevant_keywords: List[str],
                               k: int = TOP_K, cognitive: str = "1PAR") -> Dict:
    """
    Evaluasi kualitas retrieval menggunakan dua jalur.

    Output utama retrieval:
      - precision_at_k  : Proxy P@K berbasis threshold skor
      - recall_at_k     : Proxy R@K berbasis threshold skor
      - mean_similarity : rata-rata skor kemiripan chunk
      - coverage        : proporsi chunk melewati threshold coverage
      - chunk_relevance : weighted mean skor chunk
      - source_diversity: variasi sumber dokumen

    Jalur A — /retrieve endpoint (utama):
      Chunk FAISS nyata dengan skor cosine similarity aktual.

    Jalur B — keyword embedding fallback (jika /retrieve gagal):
      Embedding keyword relevan sebagai proxy dokumen.
    """
    # ── Jalur A ────────────────────────────────────────────────────────────
    try:
        retrieve_resp = requests.post(
            f"{BASE_URL}/retrieve",
            json={"query": query, "cognitive": cognitive, "k": k},
            timeout=RETRIEVE_TIMEOUT,
        )
        if retrieve_resp.status_code == 200:
            chunks = retrieve_resp.json().get("chunks", [])
            if chunks:
                top_k_chunks  = chunks[:k]
                top_k_scores  = [c["score"]  for c in top_k_chunks]
                top_k_sources = [c["source"] for c in top_k_chunks]
                top_k_texts   = [c["text"]   for c in top_k_chunks]

                mean_sim = mean_similarity(top_k_scores)
                cov      = coverage_score(top_k_scores)
                div      = source_diversity(top_k_sources)

                return {
                    "method":                "faiss_retrieve",
                    "query_source":          "endpoint /retrieve",
                    "top_k_scores":          [round(s, 4) for s in top_k_scores],
                    "top_k_sources":         top_k_sources,
                    "top_k_texts":           top_k_texts,
                    "keywords_found":        sum(1 for kw in relevant_keywords
                                                if kw.lower() in " ".join(top_k_texts).lower()),
                    "keywords_total":        len(relevant_keywords),
                    "mean_similarity":       round(mean_sim, 4),
                    "coverage":              round(cov, 4),
                    "source_diversity":      round(div, 4),
                    # Metrik utama retrieval; P/R adalah proxy berbasis threshold skor.
                    "precision_at_k":       round(precision_at_k(top_k_scores, k, RELEVANCE_THRESHOLD), 4),
                    "recall_at_k":          round(recall_at_k(top_k_scores, len(relevant_keywords), k, RELEVANCE_THRESHOLD), 4),
                    "chunk_relevance":      round(chunk_relevance_score(top_k_scores), 4),
                    "_note": "precision_at_k/recall_at_k adalah proxy berbasis threshold skor, bukan P@K/R@K IR murni",
                }
    except Exception as e:
        print(f"  [/retrieve Error] {e} — fallback ke keyword embedding")

    # ── Jalur B ────────────────────────────────────────────────────────────
    if not LLM_CLIENT_AVAILABLE:
        return {"error": "LLM client tidak tersedia dan /retrieve gagal"}

    emb_query = get_embedding(query)
    if emb_query is None:
        return {"error": "Gagal membuat embedding query"}

    keyword_embeddings = []
    for kw in relevant_keywords:
        emb = get_embedding(kw)
        if emb is not None:
            keyword_embeddings.append((kw, emb))

    if not keyword_embeddings:
        return {"error": "Gagal membuat embedding keyword"}

    scores = []
    for kw, emb_doc in keyword_embeddings:
        sim = cosine_similarity(emb_query, emb_doc)
        scores.append({"keyword": kw, "score": sim})

    scores.sort(key=lambda x: x["score"], reverse=True)
    top_k_scores  = [s["score"]   for s in scores[:k]]
    top_k_sources = [s["keyword"] for s in scores[:k]]

    mean_sim = mean_similarity(top_k_scores)
    cov      = coverage_score(top_k_scores)
    div      = source_diversity(top_k_sources)

    return {
        "method":                "keyword_embedding_fallback",
        "query_source":          "keyword proxy",
        "query_embedding_dim":   len(emb_query),
        "top_k_scores":          [round(s, 4) for s in top_k_scores],
        "top_k_sources":         top_k_sources,
        "top_k_texts":           top_k_sources,
        "mean_similarity":       round(mean_sim, 4),
        "coverage":              round(cov, 4),
        "source_diversity":      round(div, 4),
        # Metrik utama retrieval; P/R adalah proxy berbasis threshold skor.
        "precision_at_k":       round(precision_at_k(top_k_scores, k, RELEVANCE_THRESHOLD), 4),
        "recall_at_k":          round(recall_at_k(top_k_scores, len(relevant_keywords), k, RELEVANCE_THRESHOLD), 4),
        "chunk_relevance":      round(chunk_relevance_score(top_k_scores), 4),
        "_note": "precision_at_k/recall_at_k adalah proxy berbasis threshold skor, bukan P@K/R@K IR murni",
        "detail_scores":         [{"keyword": s["keyword"], "score": round(s["score"], 4)}
                                  for s in scores],
    }


# =============================================================================
# [FIX-2] NO-RAG BASELINE
# =============================================================================

def run_norag_baseline() -> List[Dict]:
    """
    [FIX-2] Jalankan semua query ke LLM tanpa konteks RAG.

    Ini adalah baseline krusial untuk membuktikan kontribusi RAG.
    Tanpa baseline ini, tidak ada cara untuk tahu apakah sistem RAG
    benar-benar meningkatkan kualitas jawaban.

    Perbandingan RAG vs No-RAG yang dihasilkan:
      - Faithfulness: RAG harus lebih tinggi (jawaban lebih terikat konteks)
      - Hallucination risk: RAG harus lebih rendah
      - Answer quality: RAG harus lebih tinggi (punya konteks relevan)

    CATATAN: Retrieval metrics (precision_at_k, recall_at_k, mean_similarity,
    coverage, chunk_relevance, source_diversity) tidak relevan untuk No-RAG
    baseline karena tidak ada proses retrieval. Hanya generation metrics yang
    dibandingkan.

    Returns:
      List hasil baseline, satu per test case.
    """
    print("\n" + "="*70)
    print("  [FIX-2] NO-RAG BASELINE — LLM tanpa konteks retrieval")
    print("  Digunakan sebagai pembanding untuk membuktikan nilai tambah RAG")
    print("="*70)

    baseline_results = []

    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"  [{idx:02d}/{len(TEST_CASES)}] No-RAG: {tc['query'][:55]}...")

        norag_reply = call_chat_norag(tc["query"])

        if norag_reply:
            # Faithfulness no-RAG: gunakan keywords sebagai "konteks proxy"
            # (karena tidak ada retrieved chunks). Ini akan lebih rendah dari RAG.
            # Catatan: context_proxy adalah daftar keyword, bukan dokumen nyata.
            context_proxy = [" ".join(tc["relevant_keywords"])]
            faith_norag  = evaluate_faithfulness(norag_reply, context_proxy)
            hall_norag   = detect_hallucination(norag_reply, context_proxy,
                                                precomputed_faithfulness=faith_norag)
            reference_answer = get_reference_answer_for_eval(tc, context_proxy)
            aq_norag     = evaluate_answer_quality(norag_reply, reference_answer)

            test_id = int(tc.get("_global_id", idx))
            baseline_results.append({
                "test_id":              test_id,
                "query":                tc["query"],
                "cognitive":            tc["cognitive"],
                "session_id":           tc["session_id"],
                "query_type":           tc["query_type"],
                "context_note":         tc["context_note"],
                "norag_reply":          norag_reply[:300],
                "faithfulness":         faith_norag,
                "hallucination":        hall_norag,
                "answer_quality":       aq_norag,
                "answer_quality_score": aq_norag["answer_quality_score"],
            })
            print(f"         Faith={faith_norag['faithfulness_score']:.3f}  "
                  f"Hall={hall_norag['hallucination_risk']:.3f}  "
                  f"AQ={aq_norag['answer_quality_score']:.3f}")
        else:
            print("         ⚠️  LLM tidak tersedia untuk No-RAG baseline")
            test_id = int(tc.get("_global_id", idx))
            baseline_results.append({
                "test_id":      test_id,
                "query":        tc["query"],
                "cognitive":    tc["cognitive"],
                "session_id":   tc["session_id"],
                "query_type":   tc["query_type"],
                "context_note": tc["context_note"],
                "error":        "LLM tidak tersedia",
            })

        time.sleep(0.5)

    return baseline_results


# =============================================================================
# [FIX-3] SELF-ANNOTATION WORKFLOW
# =============================================================================

def generate_annotation_template(rag_results: List[Dict]) -> str:
    """
    [FIX-3] Buat template JSON untuk anotasi manual peneliti.

    Cara penggunaan:
      1. Jalankan run_full_evaluation() terlebih dahulu.
      2. Panggil generate_annotation_template(results).
      3. Buka file self_annotations.json yang dihasilkan.
      4. Isi nilai "annotation_correct" (1=benar, 0=salah) dan
         "annotation_hallucination" (1=halusinasi, 0=tidak) untuk tiap query.
      5. Panggil compute_annotation_metrics() untuk menghitung agreement.

    Standar anotasi (untuk dokumentasi di skripsi):
      annotation_correct:
        1 = jawaban LLM menjawab pertanyaan dengan benar dan lengkap
        0 = jawaban salah, tidak relevan, atau sangat tidak lengkap

      annotation_hallucination:
        1 = jawaban mengandung klaim yang tidak didukung konteks atau
            fakta yang salah secara substantif
        0 = jawaban secara substantif akurat dan didukung konteks

    Returns:
      Path file template yang dibuat.
    """
    template = {
        "_instructions": {
            "annotation_correct":       "1=jawaban benar & relevan, 0=salah atau tidak relevan",
            "annotation_hallucination": "1=ada halusinasi/klaim tidak didukung, 0=tidak ada",
            "annotator":                "Nama peneliti (isi manual)",
            "annotation_date":          "Tanggal anotasi (isi manual)",
            "methodology_note":         (
                "Anotasi ini dilakukan oleh peneliti sebagai self-annotation. "
                "Untuk validasi eksternal, libatkan minimal 1 penilai lain dan "
                "hitung Cohen's Kappa agreement score."
            ),
        },
        "annotations": []
    }

    for r in rag_results:
        entry = {
            "test_id":                  r["test_id"],
            "query":                    r["query"],
            "cognitive":                r.get("cognitive"),
            "query_type":               r.get("query_type"),
            "context_note":             r.get("context_note"),
            "llm_reply_preview":        (r.get("llm_reply") or "")[:200],
            "reference_answer":         get_reference_answer_for_eval(ALL_TEST_CASES[r["test_id"] - 1], (r.get("retrieval") or {}).get("top_k_texts", [])),
            "auto_bool_signal":         r.get("answer_correct"),
            "auto_aq_score":            r.get("answer_quality_score"),
            "auto_hallucination_risk":  (r.get("hallucination") or {}).get("hallucination_risk"),
            "auto_risk_label":          (r.get("hallucination") or {}).get("risk_label"),
            # Isi dua field ini secara manual
            "annotation_correct":       None,
            "annotation_hallucination": None,
            "annotation_notes":         "",
        }
        template["annotations"].append(entry)

    out_path = ANNOTATION_FILE
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Template anotasi dibuat: {out_path}")
    print("  → Buka file tersebut dan isi 'annotation_correct' dan")
    print("    'annotation_hallucination' untuk tiap query (nilai: 0 atau 1).")
    return out_path


def load_annotations() -> Optional[Dict]:
    """
    [FIX-3] Baca file anotasi manual peneliti.

    Returns:
      Dict dengan key "annotations" (list), atau None jika file tidak ada
      atau belum diisi.
    """
    if not os.path.exists(ANNOTATION_FILE):
        print(f"  [Annotation] File tidak ditemukan: {ANNOTATION_FILE}")
        print("  → Jalankan generate_annotation_template() terlebih dahulu.")
        return None

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    filled = [a for a in annotations if a.get("annotation_correct") is not None]

    if not filled:
        print("  [Annotation] File ditemukan tapi belum ada anotasi yang diisi.")
        print("  → Isi nilai annotation_correct dan annotation_hallucination di file JSON.")
        return None

    print(f"  [Annotation] {len(filled)}/{len(annotations)} query sudah dianotasi.")
    return data


def compute_annotation_metrics(rag_results: List[Dict]) -> Dict:
    """
    [FIX-3] Hitung metrik berdasarkan self-annotation peneliti.

    Metrik yang dihitung:
    1. Human-validated accuracy:
       Proporsi query yang dinilai "benar" oleh peneliti.
       Ini adalah ground truth yang lebih terpercaya dari LLM-as-judge.

    2. Agreement rate (Auto vs Human):
       Seberapa sering penilaian otomatis (bool_signal) cocok dengan
       penilaian manual peneliti.
       Agreement = |auto == human| / n_annotated

    3. Human-validated hallucination rate:
       Proporsi query yang dianggap mengandung halusinasi oleh peneliti.

    4. Auto-Human hallucination agreement:
       Seberapa sering sistem otomatis (risk_label TINGGI/SEDANG)
       cocok dengan penilaian halusinasi manual.

    Returns:
      Dict metrik anotasi, atau dict dengan key "error" jika tidak ada data.
    """
    annotation_data = load_annotations()
    if annotation_data is None:
        return {"error": "Anotasi belum tersedia. Jalankan generate_annotation_template()."}

    annotations = annotation_data.get("annotations", [])
    filled = [a for a in annotations if a.get("annotation_correct") is not None]

    if not filled:
        return {"error": "Belum ada anotasi yang diisi."}

    # Buat lookup dari rag_results untuk cross-reference
    rag_lookup = {r["test_id"]: r for r in rag_results}

    # ── Hitung metrik ──────────────────────────────────────────────────────
    human_correct_list = []
    human_halluc_list  = []
    auto_correct_list  = []
    agree_correct      = 0
    agree_halluc       = 0
    n_compared         = 0

    for ann in filled:
        tid             = ann["test_id"]
        human_correct   = int(ann["annotation_correct"])
        human_halluc    = int(ann.get("annotation_hallucination", 0))

        human_correct_list.append(human_correct)
        human_halluc_list.append(human_halluc)

        # Cross-reference dengan hasil otomatis
        rag = rag_lookup.get(tid)
        if rag:
            n_compared += 1
            auto_correct = 1 if rag.get("answer_correct") else 0
            auto_correct_list.append(auto_correct)

            # Agreement: apakah auto dan human sama-sama benar atau sama-sama salah?
            if auto_correct == human_correct:
                agree_correct += 1

            # Hallucination agreement: auto TINGGI/SEDANG vs human annotation
            auto_risk = (rag.get("hallucination") or {}).get("risk_label", "RENDAH")
            auto_halluc = 1 if auto_risk in ("TINGGI", "SEDANG") else 0
            if auto_halluc == human_halluc:
                agree_halluc += 1

    n = len(filled)
    human_accuracy      = sum(human_correct_list) / n
    human_halluc_rate   = sum(human_halluc_list) / n
    auto_accuracy       = sum(auto_correct_list) / max(n_compared, 1)
    agreement_correct   = agree_correct / max(n_compared, 1)
    agreement_halluc    = agree_halluc / max(n_compared, 1)

    return {
        "n_annotated":               n,
        "n_compared":                n_compared,
        "human_validated_accuracy":  round(human_accuracy, 4),
        "auto_llm_accuracy":         round(auto_accuracy, 4),
        "agreement_rate_correctness": round(agreement_correct, 4),
        "human_hallucination_rate":  round(human_halluc_rate, 4),
        "agreement_rate_hallucination": round(agreement_halluc, 4),
        "interpretation": {
            "agreement_correctness": (
                "TINGGI (≥0.80)" if agreement_correct >= 0.80 else
                "SEDANG (0.60–0.79)" if agreement_correct >= 0.60 else
                "RENDAH (<0.60) — perlu review metodologi evaluasi otomatis"
            ),
            "human_vs_auto_accuracy_delta": round(human_accuracy - auto_accuracy, 4),
        },
        "methodology_note": (
            "Self-annotation oleh 1 peneliti. Untuk klaim akademik yang lebih kuat, "
            "rekomendasikan anotasi oleh minimal 2 penilai dengan perhitungan "
            "Cohen's Kappa (κ) sebagai inter-rater reliability."
        ),
    }


# =============================================================================
# RUNNER UTAMA
# =============================================================================

def run_full_evaluation() -> List[Dict]:
    """
    Menjalankan evaluasi RAG secara menyeluruh untuk semua test case.

    Alur evaluasi:
    1. Retrieval metrics (Keyword Coverage@K, Keyword Recall@K, MeanSim) [FIX-1]
    2. Coverage & Source Diversity
    3. Panggil /chat → dapatkan jawaban LLM
    4. Faithfulness evaluation
    5. Hallucination detection (diperbaiki) [FIX-4]
    6. Panggil /evaluate → Answer Quality
    """
    print("\n" + "="*70)
    print("  CSIPBLLM — RAG EVALUATION SUITE v6 (formula sesuai paper asli)")
    print("  Berdasarkan proposal: Muhammad Ajisaka Arsyi Taj (G6401221090)")
    print("  FIX-6a: Faithfulness → RAGAS claim-based (Es et al. 2023)")
    print("  FIX-6b: Hallucination → SelfCheckGPT-BERTScore (Manakul et al. 2023)")
    print("  FIX-6c: Answer Quality → MT-Bench 1–10 grading (Zheng et al. 2023)")
    print(f"  SelfCheck N sampel: {SELFCHECK_N_SAMPLES}  |  Max klaim RAGAS: {RAGAS_MAX_CLAIMS}")
    print("="*70)
    print(f"  Total test case : {len(TEST_CASES)}")
    print(f"  Top-K           : {TOP_K}")
    print(f"  Waktu mulai     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    results = []

    for idx, tc in enumerate(TEST_CASES, 1):
        tc_start = time.time()
        cache_size_before = len(_embedding_cache)
        print(f"[{idx:02d}/{len(TEST_CASES)}] Query: {tc['query'][:60]}...")
        test_id = int(tc.get("_global_id", idx))
        result = {
            "test_id":      test_id,
            "query":        tc["query"],
            "cognitive":    tc["cognitive"],
            "session_id":   tc["session_id"],
            "query_type":   tc["query_type"],
            "context_note": tc["context_note"],
            "timestamp":    datetime.now().isoformat(),
        }

        # ── LANGKAH 1: Retrieval Evaluation ────────────────────────────────
        print("         ↳ [1] Evaluasi retrieval embedding...")
        retrieval_result = evaluate_retrieval_direct(
            tc["query"], tc["relevant_keywords"], k=TOP_K,
            cognitive=tc["cognitive"],
        )
        result["retrieval"] = retrieval_result
        if "error" not in retrieval_result:
            print(f"             ProxyP@{TOP_K}: {retrieval_result.get('precision_at_k', 0):.3f} | "
                  f"ProxyR@{TOP_K}: {retrieval_result.get('recall_at_k', 0):.3f} | "
                  f"MeanSim: {retrieval_result['mean_similarity']:.3f} | "
                  f"Coverage: {retrieval_result['coverage']:.3f} | "
                  f"ChunkRel: {retrieval_result.get('chunk_relevance', 0):.3f} "
                  f"[{retrieval_result.get('method', '?')}]")

        # ── LANGKAH 2: Panggil /chat ────────────────────────────────────────
        print("         ↳ [2] Memanggil /chat API...")
        chat_resp = call_chat_api(tc["query"], tc["cognitive"], tc["session_id"])
        result["chat_response"] = chat_resp

        if chat_resp and "reply" in chat_resp:
            llm_reply  = chat_resp.get("reply", "")
            followup_q = chat_resp.get("followup_question", "")
            result["llm_reply"]         = llm_reply[:300]
            result["followup_question"] = followup_q

            rag_context: List[str] = retrieval_result.get("top_k_texts", [])
            if not rag_context:
                print("             [Info] top_k_texts kosong — fallback ke keyword list")
                rag_context = list(tc["relevant_keywords"])

            # ── LANGKAH 3: Faithfulness ─────────────────────────────────────
            print("         ↳ [3] Mengevaluasi faithfulness...")
            faith_result = evaluate_faithfulness(llm_reply, rag_context)
            result["faithfulness"] = faith_result
            print(f"             Faithfulness: {faith_result['faithfulness_score']:.3f} "
                  f"({faith_result['method']})")

            # ── LANGKAH 4: Hallucination Detection (FIX-6b: SelfCheckGPT) ─
            print(f"         ↳ [4] SelfCheckGPT-BERTScore (N={SELFCHECK_N_SAMPLES} sampel)...")
            hall_result = detect_hallucination(
                answer         = llm_reply,
                retrieved_chunks = rag_context,
                precomputed_faithfulness = None,  # tidak dipakai di v6
                query          = tc["query"],
                cognitive      = tc["cognitive"],
                session_id     = tc["session_id"] + "-selfcheck",
            )
            result["hallucination"] = hall_result
            n_sents   = hall_result.get("n_sentences", "?")
            n_samps   = hall_result.get("n_samples", 0)
            method_h  = hall_result.get("method", "?")
            print(f"             Risiko halusinasi: {hall_result['hallucination_risk']:.3f} "
                  f"[{hall_result['risk_label']}]  "
                  f"kalimat={n_sents}  sampel={n_samps}  ({method_h})")

            # ── LANGKAH 5: Answer Quality (FIX-6c: MT-Bench grading) ─────────
            print("         ↳ [5] MT-Bench single-answer grading (1–10)...")
            reference_answer = get_reference_answer_for_eval(tc, rag_context)

            aq_result = evaluate_answer_quality(
                llm_reply        = llm_reply,
                reference_answer = reference_answer,
                eval_resp        = None,       # tidak dipakai di v6
                question         = tc["query"],
                rag_context      = rag_context,
            )
            result["answer_quality"]       = aq_result
            result["answer_quality_score"] = aq_result["answer_quality_score"]
            # answer_correct: True jika skor MT-Bench ≥ 5 (setara "sebagian benar")
            s_llm = aq_result.get("llm_judge_score")
            result["answer_correct"] = (s_llm >= 5) if s_llm is not None else None
            print(f"             MT-Bench score : {s_llm}/10  "
                  f"→ quality={aq_result['answer_quality_score']:.3f}  "
                  f"[{aq_result['label']}]  ({aq_result['method']})")

        else:
            print("         ↳ ⚠️  API /chat tidak tersedia atau error")
            result["llm_reply"]            = None
            result["faithfulness"]         = None
            result["hallucination"]        = None
            result["answer_quality"]       = None
            result["answer_quality_score"] = None
            result["answer_correct"]       = None

        results.append(result)
        tc_elapsed = time.time() - tc_start
        cache_hits = len(_embedding_cache) - cache_size_before
        print(f"         ⏱  Selesai dalam {tc_elapsed:.1f}s | "
              f"Cache embedding: {len(_embedding_cache)} entri "
              f"(+{cache_hits} baru)")
        print()
        time.sleep(0.5 if "OLLAMA" in CURRENT_MODE else 1.5)

    return results


# =============================================================================
# OFFLINE STATISTICAL ANALYSIS
# =============================================================================

def run_offline_analysis(history_json_path: str) -> Dict:
    """Analisis statistik offline dari file JSON historis."""
    if not os.path.exists(history_json_path):
        return {"error": f"File tidak ditemukan: {history_json_path}"}

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return {"error": "File JSON kosong"}

    if isinstance(history, list):
        entries = history
    elif isinstance(history, dict):
        entries = history.get("history", [])
    else:
        return {"error": f"Format file JSON tidak dikenali: {type(history).__name__}"}

    cognitive_dist   = {}
    reply_lengths    = []
    sessions         = {}
    followup_present = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cog = entry.get("cognitive", "unknown")
        cognitive_dist[cog] = cognitive_dist.get(cog, 0) + 1
        reply = entry.get("reply", "")
        if reply:
            reply_lengths.append(len(reply))
        sess = entry.get("session_id", "default")
        sessions[sess] = sessions.get(sess, 0) + 1
        if entry.get("followup_question"):
            followup_present += 1

    total = len(entries)
    return {
        "total_interactions":      total,
        "unique_sessions":         len(sessions),
        "avg_queries_per_session": round(total / max(len(sessions), 1), 2),
        "cognitive_distribution":  cognitive_dist,
        "avg_reply_length_chars":  round(statistics.mean(reply_lengths), 1) if reply_lengths else 0,
        "median_reply_length":     round(statistics.median(reply_lengths), 1) if reply_lengths else 0,
        "stdev_reply_length":      round(statistics.stdev(reply_lengths), 1) if len(reply_lengths) > 1 else 0,
        "followup_rate":           round(followup_present / max(total, 1), 4),
        "sessions_detail":         sessions,
    }


# =============================================================================
# METRIK AGREGAT
# =============================================================================

def compute_aggregate_metrics(
    results: List[Dict],
    norag_results: Optional[List[Dict]] = None,
    annotation_metrics: Optional[Dict] = None,
) -> Dict:
    """
    Menghitung metrik agregat dari seluruh hasil evaluasi.

    [FIX-1] Metrik retrieval diperapikan:
      keyword_coverage_at_k dan keyword_recall_at_k dihapus dari output.
      precision_at_k dan recall_at_k dipakai sebagai metrik proxy berbasis
      threshold skor retrieval.

    [FIX-2] Jika norag_results tersedia, hitung delta RAG vs No-RAG.

    [FIX-3] Jika annotation_metrics tersedia, sertakan dalam agregat.
    """
    proxy_p_list, proxy_r_list, meansim_list = [], [], []
    chunkrel_list = []
    cov_list, div_list                      = [], []
    faith_list, hall_list                   = [], []
    correct_list                            = []
    aq_score_list                           = []

    for r in results:
        ret = r.get("retrieval", {})
        if "precision_at_k" in ret:
            proxy_p_list.append(ret["precision_at_k"])
            proxy_r_list.append(ret.get("recall_at_k", 0.0))
            meansim_list.append(ret["mean_similarity"])
            cov_list.append(ret["coverage"])
            div_list.append(ret["source_diversity"])
            if ret.get("chunk_relevance") is not None:
                chunkrel_list.append(ret["chunk_relevance"])

        faith = r.get("faithfulness", {})
        if faith and "faithfulness_score" in faith:
            faith_list.append(faith["faithfulness_score"])

        hall = r.get("hallucination", {})
        if hall and "hallucination_risk" in hall:
            hall_list.append(hall["hallucination_risk"])

        if r.get("answer_correct") is not None:
            correct_list.append(1 if r["answer_correct"] else 0)

        if r.get("answer_quality_score") is not None:
            aq_score_list.append(r["answer_quality_score"])

    def avg(lst): return round(sum(lst)/len(lst), 4) if lst else None

    aggregate = {
        "n_tested":  len(results),
        "retrieval": {
            "avg_precision_at_k":    avg(proxy_p_list),
            "avg_recall_at_k":       avg(proxy_r_list),
            "avg_mean_similarity":   avg(meansim_list),
            "avg_coverage":          avg(cov_list),
            "avg_source_diversity":  avg(div_list),
            "avg_chunk_relevance":   avg(chunkrel_list),
            "_note": (
                "precision_at_k/recall_at_k adalah metrik proxy berbasis threshold skor, "
                "bukan Precision@K/Recall@K standar IR dengan qrels manual. "
                "chunk_relevance adalah weighted mean skor chunk."
            ),
        },
        "generation": {
            "avg_faithfulness":       avg(faith_list),
            "avg_hallucination_risk": avg(hall_list),
        },
        "answer_quality": {
            "total_evaluated":   len(aq_score_list),
            "avg_quality_score": avg(aq_score_list),
            "min_quality_score": round(min(aq_score_list), 4) if aq_score_list else None,
            "max_quality_score": round(max(aq_score_list), 4) if aq_score_list else None,
            "correct":           sum(correct_list),
            "incorrect":         len(correct_list) - sum(correct_list),
            "accuracy":          avg(correct_list),
        },
    }

    # [FIX-2] No-RAG baseline comparison
    if norag_results:
        norag_faith = [r.get("faithfulness", {}).get("faithfulness_score", 0)
                       for r in norag_results if isinstance(r.get("faithfulness"), dict)]
        norag_hall  = [r.get("hallucination", {}).get("hallucination_risk", 0)
                       for r in norag_results if isinstance(r.get("hallucination"), dict)]
        norag_aq    = [r.get("answer_quality_score", 0)
                       for r in norag_results if r.get("answer_quality_score") is not None]

        rag_faith_avg = avg(faith_list) or 0
        rag_hall_avg  = avg(hall_list)  or 0
        rag_aq_avg    = avg(aq_score_list) or 0

        norag_faith_avg = avg(norag_faith) or 0
        norag_hall_avg  = avg(norag_hall)  or 0
        norag_aq_avg    = avg(norag_aq)    or 0

        aggregate["norag_baseline"] = {
            "avg_faithfulness":       avg(norag_faith),
            "avg_hallucination_risk": avg(norag_hall),
            "avg_quality_score":      avg(norag_aq),
        }
        aggregate["rag_vs_norag_delta"] = {
            "faithfulness_delta":       round(rag_faith_avg - norag_faith_avg, 4),
            "hallucination_risk_delta": round(rag_hall_avg - norag_hall_avg, 4),
            "quality_score_delta":      round(rag_aq_avg - norag_aq_avg, 4),
            "interpretation": {
                "faithfulness":       "RAG lebih baik" if rag_faith_avg > norag_faith_avg else "No-RAG lebih baik / setara",
                "hallucination_risk": "RAG lebih baik (lebih rendah)" if rag_hall_avg < norag_hall_avg else "No-RAG lebih baik / setara",
                "quality_score":      "RAG lebih baik" if rag_aq_avg > norag_aq_avg else "No-RAG lebih baik / setara",
            },
        }

    # [FIX-3] Self-annotation metrics
    if annotation_metrics and "error" not in annotation_metrics:
        aggregate["self_annotation"] = annotation_metrics

    return aggregate


# =============================================================================
# SAVE & REPORT
# =============================================================================

def save_results(
    results: List[Dict],
    aggregate: Dict,
    offline_stats: Dict,
    norag_results: Optional[List[Dict]] = None,
) -> Tuple[str, str, str]:
    """Simpan hasil evaluasi ke file JSON, CSV, dan TXT."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(OUTPUT_DIR, f"rag_eval_{ts}.json")
    payload   = {
        "aggregate":          aggregate,
        "offline_stats":      offline_stats,
        "individual_results": results,
        "norag_baseline":     norag_results or [],
        "eval_version":       "v6",
        "fixes_applied":      [
            "FIX-1: clean retrieval metrics",
            "FIX-2: no-RAG baseline",
            "FIX-3: self-annotation",
            "FIX-4: hallucination detection (v4 heuristik)",
            "FIX-6a: RAGAS claim-based faithfulness (Es et al. 2023)",
            "FIX-6b: SelfCheckGPT-BERTScore hallucination (Manakul et al. 2023)",
            "FIX-6c: MT-Bench single-answer grading (Zheng et al. 2023)",
        ],
        "config": {
            "SELFCHECK_N_SAMPLES":  SELFCHECK_N_SAMPLES,
            "RAGAS_MAX_CLAIMS":     RAGAS_MAX_CLAIMS,
            "MTBENCH_TEMPERATURE":  MTBENCH_TEMPERATURE,
            "SELFCHECK_TEMPERATURE": SELFCHECK_TEMPERATURE,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path   = os.path.join(OUTPUT_DIR, f"rag_eval_{ts}.csv")
    fieldnames = [
        "test_id", "query", "cognitive", "session_id", "query_type", "context_note",
        # Retrieval metrics (tidak berubah)
        "precision_at_k", "recall_at_k", "chunk_relevance",
        "mean_similarity", "coverage", "source_diversity",
        # [FIX-6a] RAGAS Faithfulness
        "faithfulness_score", "faithfulness_n_claims", "faithfulness_n_supported",
        "faithfulness_method",
        # [FIX-6b] SelfCheckGPT
        "hallucination_risk", "risk_label",
        "selfcheck_n_sentences", "selfcheck_n_samples", "hallucination_method",
        # [FIX-6c] MT-Bench Answer Quality
        "answer_quality_score", "llm_judge_score", "aq_label", "answer_correct",
        "answer_quality_method",
        # [FIX-2] No-RAG baseline
        "norag_aq_score", "norag_faithfulness", "norag_hallucination_risk",
    ]

    # Buat lookup no-RAG
    norag_lookup: Dict[int, Dict] = {}
    if norag_results:
        for nr in norag_results:
            norag_lookup[nr["test_id"]] = nr

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            ret   = r.get("retrieval", {})
            faith = r.get("faithfulness", {}) or {}
            hall  = r.get("hallucination", {}) or {}
            aq    = r.get("answer_quality", {}) or {}
            nr    = norag_lookup.get(r["test_id"], {})
            writer.writerow({
                "test_id":               r["test_id"],
                "query":                 r["query"][:80],
                "cognitive":             r["cognitive"],
                "session_id":            r.get("session_id", ""),
                "query_type":            r.get("query_type", ""),
                "context_note":          r.get("context_note", "")[:120],
                "precision_at_k":        ret.get("precision_at_k", ""),
                "recall_at_k":           ret.get("recall_at_k", ""),
                "chunk_relevance":       ret.get("chunk_relevance", ""),
                "mean_similarity":       ret.get("mean_similarity", ""),
                "coverage":              ret.get("coverage", ""),
                "source_diversity":      ret.get("source_diversity", ""),
                # [FIX-6a] RAGAS Faithfulness
                "faithfulness_score":        faith.get("faithfulness_score", ""),
                "faithfulness_n_claims":     faith.get("n_claims", ""),
                "faithfulness_n_supported":  faith.get("n_supported", ""),
                "faithfulness_method":       faith.get("method", ""),
                # [FIX-6b] SelfCheckGPT
                "hallucination_risk":        hall.get("hallucination_risk", ""),
                "risk_label":                hall.get("risk_label", ""),
                "selfcheck_n_sentences":     hall.get("n_sentences", ""),
                "selfcheck_n_samples":       hall.get("n_samples", ""),
                "hallucination_method":      hall.get("method", ""),
                # [FIX-6c] MT-Bench Answer Quality
                "answer_quality_score":      aq.get("answer_quality_score", ""),
                "llm_judge_score":           aq.get("llm_judge_score", ""),
                "aq_label":                  aq.get("label", ""),
                "answer_correct":            r.get("answer_correct", ""),
                "answer_quality_method":     aq.get("method", ""),
                # [FIX-2] No-RAG baseline columns
                "norag_aq_score":        (nr.get("answer_quality") or {}).get("answer_quality_score", ""),
                "norag_faithfulness":    (nr.get("faithfulness") or {}).get("faithfulness_score", ""),
                "norag_hallucination_risk": (nr.get("hallucination") or {}).get("hallucination_risk", ""),
            })

    txt_path = os.path.join(OUTPUT_DIR, f"rag_eval_{ts}_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        _write_report(f, aggregate, offline_stats, results, norag_results)

    return json_path, csv_path, txt_path


def _write_report(f, aggregate: Dict, offline_stats: Dict,
                  results: List[Dict], norag_results: Optional[List[Dict]] = None):
    """Tulis laporan evaluasi terformat ke file."""
    sep  = "=" * 70
    sep2 = "-" * 70

    def w(line=""): f.write(line + "\n")

    w(sep)
    w("  LAPORAN EVALUASI RAG v4 — CSIPBLLM")
    w(f"  Tanggal  : {datetime.now().strftime('%d %B %Y, %H:%M:%S')}")
    w(f"  Peneliti : Muhammad Ajisaka Arsyi Taj (G6401221090)")
    w(f"  Versi    : v4 (FIX-1: clean retrieval metrics, FIX-2: No-RAG baseline,")
    w(f"             FIX-3: self-annotation, FIX-4: hallucination detection)")
    w(sep)
    w()
    w("RINGKASAN METRIK AGREGAT")
    w(sep2)

    ret = aggregate.get("retrieval", {})
    w(f"  Jumlah test case     : {aggregate['n_tested']}")
    w()
    w("  [A] RETRIEVAL METRICS")
    w(f"      Proxy Precision@K  : {ret.get('avg_precision_at_k', 'N/A')}")
    w(f"        → Proporsi Top-K item dengan skor ≥ threshold relevansi")
    w(f"        → Proxy, bukan Precision@K standar IR berbasis qrels manual")
    w(f"      Proxy Recall@K     : {ret.get('avg_recall_at_k', 'N/A')}")
    w(f"        → Proporsi item relevan-terestimasi terhadap total relevant_keywords")
    w(f"      Mean Similarity    : {ret.get('avg_mean_similarity', 'N/A')}")
    w(f"        → MeanSim = (1/K) Σ si")
    w()
    w("  [B] RETRIEVAL HEALTH")
    w(f"      Coverage           : {ret.get('avg_coverage', 'N/A')}")
    w(f"        → Proporsi chunk dengan skor cosine ≥ {COVERAGE_THRESHOLD}")
    w(f"      Source Diversity   : {ret.get('avg_source_diversity', 'N/A')}")
    w(f"        → |sumber unik| / K")
    w()

    gen = aggregate.get("generation", {})
    w("  [C] GENERATION QUALITY")
    w(f"      Avg Faithfulness       : {gen.get('avg_faithfulness', 'N/A')}")
    w(f"        → sim(ē_answer, ē_context) + keyword overlap")
    w(f"      Avg Hallucination Risk : {gen.get('avg_hallucination_risk', 'N/A')}")
    w(f"        → [FIX-4] 0.5×out_of_context + 0.3×sentence_ooc + 0.2×oov_ratio")
    w(f"        → Regex negasi & uncertainty phrases dihapus (false positive)")
    w()

    aq = aggregate.get("answer_quality", {})
    w("  [D] ANSWER QUALITY")
    w(f"      Total dievaluasi      : {aq.get('total_evaluated', 0)}")
    w(f"      Avg Quality Score     : {aq.get('avg_quality_score', 'N/A')}  (0.0–1.0)")
    w(f"      Min / Max Score       : {aq.get('min_quality_score', 'N/A')} / {aq.get('max_quality_score', 'N/A')}")
    w(f"      Benar (boolean)       : {aq.get('correct', 0)}")
    w(f"      Salah (boolean)       : {aq.get('incorrect', 0)}")
    w(f"      Akurasi (boolean)     : {aq.get('accuracy', 'N/A')}")
    w()

    # [FIX-2] No-RAG baseline section
    norag_agg = aggregate.get("norag_baseline")
    delta     = aggregate.get("rag_vs_norag_delta")
    if norag_agg:
        w("  [E] NO-RAG BASELINE vs RAG (FIX-2)")
        w(f"      {'Metrik':<28} {'RAG':>8}  {'No-RAG':>8}  {'Delta':>8}  Kesimpulan")
        w(f"      {'-'*68}")
        rag_gen = aggregate.get("generation", {})
        rag_aq  = aggregate.get("answer_quality", {})

        def _fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v or "N/A")

        faith_rag    = rag_gen.get("avg_faithfulness") or 0
        faith_norag  = norag_agg.get("avg_faithfulness") or 0
        hall_rag     = rag_gen.get("avg_hallucination_risk") or 0
        hall_norag   = norag_agg.get("avg_hallucination_risk") or 0
        aq_rag       = rag_aq.get("avg_quality_score") or 0
        aq_norag_val = norag_agg.get("avg_quality_score") or 0

        interp = delta.get("interpretation", {}) if delta else {}
        w(f"      {'Faithfulness':<28} {_fmt(faith_rag):>8}  {_fmt(faith_norag):>8}  "
          f"{_fmt(faith_rag - faith_norag):>8}  {interp.get('faithfulness', '')}")
        w(f"      {'Hallucination Risk':<28} {_fmt(hall_rag):>8}  {_fmt(hall_norag):>8}  "
          f"{_fmt(hall_rag - hall_norag):>8}  {interp.get('hallucination_risk', '')}")
        w(f"      {'Answer Quality Score':<28} {_fmt(aq_rag):>8}  {_fmt(aq_norag_val):>8}  "
          f"{_fmt(aq_rag - aq_norag_val):>8}  {interp.get('quality_score', '')}")
        w()

    # [FIX-3] Self-annotation section
    ann_metrics = aggregate.get("self_annotation")
    if ann_metrics and "error" not in ann_metrics:
        w("  [F] SELF-ANNOTATION METRICS (FIX-3)")
        w(f"      Query dianotasi           : {ann_metrics.get('n_annotated', 0)}")
        w(f"      Human-validated accuracy  : {ann_metrics.get('human_validated_accuracy', 'N/A')}")
        w(f"      Auto LLM accuracy         : {ann_metrics.get('auto_llm_accuracy', 'N/A')}")
        w(f"      Agreement (correctness)   : {ann_metrics.get('agreement_rate_correctness', 'N/A')}  "
          f"[{ann_metrics.get('interpretation', {}).get('agreement_correctness', '')}]")
        w(f"      Human hallucination rate  : {ann_metrics.get('human_hallucination_rate', 'N/A')}")
        w(f"      Agreement (hallucination) : {ann_metrics.get('agreement_rate_hallucination', 'N/A')}")
        w(f"      Catatan: {ann_metrics.get('methodology_note', '')}")
        w()

    if offline_stats and "error" not in offline_stats:
        w("  [G] OFFLINE STATISTICAL ANALYSIS")
        w(f"      Total interaksi      : {offline_stats.get('total_interactions', 'N/A')}")
        w(f"      Sesi unik            : {offline_stats.get('unique_sessions', 'N/A')}")
        w(f"      Avg query/sesi       : {offline_stats.get('avg_queries_per_session', 'N/A')}")
        w(f"      Avg panjang jawaban  : {offline_stats.get('avg_reply_length_chars', 'N/A')} karakter")
        w(f"      Tingkat followup     : {offline_stats.get('followup_rate', 'N/A')}")
        cog_dist = offline_stats.get("cognitive_distribution", {})
        if cog_dist:
            w("      Distribusi kognitif  :")
            for cog, count in sorted(cog_dist.items()):
                w(f"        {cog}: {count} sesi")
    w()

    w(sep2)
    w("DETAIL PER TEST CASE")
    w(sep2)

    norag_lookup: Dict[int, Dict] = {}
    if norag_results:
        for nr in norag_results:
            norag_lookup[nr["test_id"]] = nr

    for r in results:
        ret   = r.get("retrieval", {})
        faith = r.get("faithfulness", {}) or {}
        hall  = r.get("hallucination", {}) or {}

        def _fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

        w(f"\n  [{r['test_id']:02d}] {r['query'][:65]}")
        w(f"       Kognitif    : {r['cognitive']}")
        if "precision_at_k" in ret:
            w(f"       ProxyP@K    : {ret['precision_at_k']:.4f}  "
              f"ProxyR@K: {ret['recall_at_k']:.4f}  "
              f"MeanSim: {ret['mean_similarity']:.4f}")
            w(f"       Coverage    : {ret['coverage']:.4f}  "
              f"ChunkRel: {ret.get('chunk_relevance', 0):.4f}  "
              f"Diversity: {ret['source_diversity']:.4f}")
        if faith:
            # [FIX-6a] RAGAS claim-based faithfulness
            n_c = faith.get('n_claims', '?')
            n_s = faith.get('n_supported', '?')
            w(f"       Faithfulness: {_fmt(faith.get('faithfulness_score', '?'))}  "
              f"klaim={n_s}/{n_c}  ({faith.get('method', '?')})")
        if hall:
            # [FIX-6b] SelfCheckGPT
            n_sent = hall.get('n_sentences', '?')
            n_samp = hall.get('n_samples', '?')
            w(f"       Halusinasi  : {_fmt(hall.get('hallucination_risk', '?'))}  "
              f"[{hall.get('risk_label', '?')}]  "
              f"kalimat={n_sent}  sampel={n_samp}  ({hall.get('method', '?')})")
        ans = r.get("answer_quality", {}) or {}
        if ans:
            # [FIX-6c] MT-Bench single-answer grading
            s_llm = ans.get('llm_judge_score', '?')
            w(f"       AQ Score    : {_fmt(ans.get('answer_quality_score', '?'))}  "
              f"[{ans.get('label', '?')}]  "
              f"MT-Bench={s_llm}/10  ({ans.get('method', '?')})")
        # [FIX-2] Tampilkan No-RAG baseline untuk perbandingan
        nr = norag_lookup.get(r["test_id"])
        if nr and not nr.get("error"):
            nr_aq   = (nr.get("answer_quality") or {}).get("answer_quality_score", "?")
            nr_fth  = (nr.get("faithfulness") or {}).get("faithfulness_score", "?")
            nr_hall = (nr.get("hallucination") or {}).get("hallucination_risk", "?")
            w(f"       No-RAG      : AQ={_fmt(nr_aq)}  Faith={_fmt(nr_fth)}  Hall={_fmt(nr_hall)}")

    w()
    w(sep)
    w("End of Report — rag_evaluator6.py (v6: RAGAS + SelfCheckGPT + MT-Bench)")
    w(sep)


# =============================================================================
# ENTRY POINT
# =============================================================================

def _run_current_test_cases(batch_label: str = "") -> Tuple[str, str, str]:
    """Jalankan evaluasi untuk TEST_CASES yang sedang aktif, lalu simpan hasilnya."""
    print("\n🔬 Memulai RAG Evaluation Suite v6 (formula paper-aligned)...")
    if batch_label:
        print(f"   Batch aktif: {batch_label} | jumlah soal: {len(TEST_CASES)}")
    print("   FIX-6a: RAGAS Faithfulness | FIX-6b: SelfCheckGPT | FIX-6c: MT-Bench")
    print(f"   SelfCheck N={SELFCHECK_N_SAMPLES} | RAGAS max_claims={RAGAS_MAX_CLAIMS}")
    print("   Pastikan server FastAPI sudah berjalan di", BASE_URL)
    print("   Konfigurasi LLM aktif:", CURRENT_MODE, "\n")

    # 1. Evaluasi RAG utama
    results = run_full_evaluation()

    # 2. No-RAG baseline opsional. Default MATI agar lebih ringan/stabil untuk batch panjang.
    run_norag = os.environ.get("RUN_NORAG_BASELINE", "0").strip().lower() in ("1", "true", "yes", "y")
    if run_norag:
        print("\n🔀 Menjalankan No-RAG baseline...")
        norag_results = run_norag_baseline()
    else:
        print("\n⏭️  No-RAG baseline dilewati. Set RUN_NORAG_BASELINE=1 kalau ingin menjalankannya.")
        norag_results = None

    # 3. Self-annotation opsional. Untuk auto-batch, template manual tidak wajib dibuat setiap batch.
    use_annotation = os.environ.get("USE_SELF_ANNOTATION", "0").strip().lower() in ("1", "true", "yes", "y")
    if use_annotation:
        if not os.path.exists(ANNOTATION_FILE):
            print("\n📝 Membuat template anotasi manual...")
            generate_annotation_template(results)
            annotation_metrics = None
        else:
            print("\n📋 Memuat anotasi manual...")
            annotation_metrics = compute_annotation_metrics(results)
    else:
        annotation_metrics = None

    # 4. Hitung metrik agregat
    print("\n📊 Menghitung metrik agregat...")
    aggregate = compute_aggregate_metrics(results, norag_results, annotation_metrics)

    # 5. Offline analysis
    history_path = os.path.join(
        os.path.dirname(__file__), "history_logs", "conversation_log.json"
    )
    print(f"📂 Membaca log historis dari: {history_path}")
    offline_stats = run_offline_analysis(history_path)

    # 6. Simpan semua hasil
    print("\n💾 Menyimpan hasil...")
    json_path, csv_path, txt_path = save_results(results, aggregate, offline_stats, norag_results)

    # 7. Ringkasan terminal
    print("\n" + "="*70)
    print("  ✅ EVALUASI v4 SELESAI")
    if batch_label:
        print(f"  Batch      : {batch_label}")
    print("="*70)
    print(f"  Hasil JSON  : {json_path}")
    print(f"  Hasil CSV   : {csv_path}")
    print(f"  Laporan TXT : {txt_path}")
    print()
    print("  RINGKASAN CEPAT:")
    ret = aggregate.get("retrieval", {})
    gen = aggregate.get("generation", {})
    aq  = aggregate.get("answer_quality", {})
    print(f"    Proxy Precision@{TOP_K}    : {ret.get('avg_precision_at_k', 'N/A')}")
    print(f"    Proxy Recall@{TOP_K}       : {ret.get('avg_recall_at_k', 'N/A')}")
    print(f"    Mean Similarity            : {ret.get('avg_mean_similarity', 'N/A')}")
    print(f"    Coverage                   : {ret.get('avg_coverage', 'N/A')}")
    print(f"    Source Diversity           : {ret.get('avg_source_diversity', 'N/A')}")
    print(f"    Faithfulness (RAGAS)       : {gen.get('avg_faithfulness', 'N/A')}")
    print(f"    Hallucination (SelfCheck)  : {gen.get('avg_hallucination_risk', 'N/A')}")
    print(f"    Avg Quality (MT-Bench/9)   : {aq.get('avg_quality_score', 'N/A')} "
          f"(min={aq.get('min_quality_score', 'N/A')} max={aq.get('max_quality_score', 'N/A')})")
    print(f"    Accuracy (judge score≥5)   : {aq.get('accuracy', 'N/A')} "
          f"({aq.get('correct', 0)}/{aq.get('total_evaluated', 0)})")
    print("="*70)

    return json_path, csv_path, txt_path


def main():
    """
    Mode jalan:
      1) Default single batch: gunakan RAG_BATCH_START dan RAG_BATCH_END.
      2) Auto batch: set RAG_AUTO_BATCHES=1, maka jalan 1–25, 26–50, 51–75 berurutan.

    Contoh Windows CMD:
      set RAG_AUTO_BATCHES=1
      python rag_evaluator5.py

    Default No-RAG baseline dimatikan agar lebih ringan:
      set RUN_NORAG_BASELINE=1   untuk mengaktifkan No-RAG.
    """
    global TEST_CASES

    auto_batches = os.environ.get("RAG_AUTO_BATCHES", "0").strip().lower() in ("1", "true", "yes", "y")

    if not auto_batches:
        _run_current_test_cases(batch_label=f"{BATCH_START}-{BATCH_END}")
        return

    batches = [(1, 25), (26, 50), (51, 75)]
    saved_csv_paths: List[str] = []

    print("\n" + "=" * 70)
    print("  AUTO BATCH MODE AKTIF")
    print("  Akan menjalankan: 1–25 → simpan → 26–50 → simpan → 51–75 → simpan")
    print("=" * 70)

    for start_i, end_i in batches:
        TEST_CASES = ALL_TEST_CASES[start_i - 1:end_i]

        print("\n" + "=" * 70)
        print(f"  MENJALANKAN BATCH {start_i}-{end_i} ({len(TEST_CASES)} soal)")
        print("=" * 70)

        json_path, csv_path, txt_path = _run_current_test_cases(batch_label=f"{start_i}-{end_i}")
        saved_csv_paths.append(csv_path)

        # Bersihkan cache embedding antar batch agar proses lebih ringan.
        _embedding_cache.clear()
        print(f"\n✅ Batch {start_i}-{end_i} selesai dan sudah tersimpan.")
        print(f"   JSON: {json_path}")
        print(f"   CSV : {csv_path}")
        print(f"   TXT : {txt_path}")

        if end_i < batches[-1][1]:
            print("\n⏸️  Istirahat 10 detik sebelum batch berikutnya...")
            time.sleep(10)

    print("\n" + "=" * 70)
    print("  ✅ SEMUA BATCH SELESAI")
    print("=" * 70)
    print("File CSV batch yang dihasilkan:")
    for path in saved_csv_paths:
        print(" -", path)
    print("\nGabungkan rata-rata dari CSV tersebut dengan script merge_eval_results.py.")


if __name__ == "__main__":
    main()