"""
compare_llms.py
═══════════════
Comparative evaluation harness for six LLMs on the PLS-CT RL system.

Design (decided with thesis author):
  • TUTOR role  → varies (the LLM under test generates explanations + follow-ups)
  • JUDGE role  → varies (the same LLM grades whether student answer is correct)
  • STUDENT     → FIXED, rule-based StudentSimulator from simulate_rl/profiles_new.py
                  (this removes one major confound; the student behaves identically
                   across all models given the same seed)

Why this design
───────────────
The RL agent receives `is_correct` from the judge. If the judge is the same model
as the tutor, the model is effectively grading its own output — a known LLM-as-judge
bias (Zheng et al. 2023; Panickssery et al. 2024). This is acknowledged in the
thesis Limitations section. The rule-based student gives us a ground-truth
`is_correct_truth` against which the judge's `is_correct_judge` can be measured,
so per-model judge agreement is reported alongside the RL metrics.

What gets logged per (model, seed) run
──────────────────────────────────────
  • Full step log (same schema as run_session, plus judge_agreement field)
  • KT-AUC, reward decomposition CV, OPE-DR — computed via evaluation/evaluator.py
  • Judge agreement rate (judge vs ground truth)
  • Latency & token cost estimates

Output
──────
  results/runs/{model_slug}_seed{seed}.json    — per-run detail
  results/summary.csv                          — model × metric table
  results/stats.json                           — Friedman + pairwise Wilcoxon

Usage
─────
  # Smoke test (single model, single seed, short run)
  python compare_llms.py --smoke

  # Full thesis run
  python compare_llms.py --full

  # Recommended revised thesis run: fixed judge + balanced simulated labels
  python compare_llms.py --full --fixed-judge-model grok-4.1-fast \
                         --fixed-judge-provider openrouter \
                         --label-source truth --balance-labels

  # Custom
  python compare_llms.py --models llama3.1:8b mistral:7b-instruct \\
                         --seeds 42 43 44 --questions 30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Project imports — these must work from the repo root
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))

from core.pedagogy_selector import SessionRegistry, SEEDING_QUESTIONS
from core.rl_metrics import LEARNING_TYPES
from simulate_rl.profiles_new import PROFILES, StudentSimulator
from evaluation.evaluator import (
    evaluate_kt_auc,
    evaluate_reward_decomposition,
    evaluate_ope_dr,
)


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT EXPERIMENT CONFIG (matches thesis Tabel models)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS: List[Dict[str, str]] = [
    {"name": "deepseek/deepseek-v3.2",                 "provider": "openrouter"},
    {"name": "google/gemma-3-27b-it",                 "provider": "openrouter"},
    {"name": "microsoft/phi-4",                       "provider": "openrouter"},
    {"name": "mistralai/mistral-small-3.2-24b-instruct", "provider": "openrouter"},
    {"name": "qwen/qwen3-32b",                        "provider": "openrouter"},
    {"name": "tencent/hy3-preview",                   "provider": "openrouter"},
]

DEFAULT_SEEDS    = [42, 43, 44, 45, 46, 47, 48, 49]  # 8 seeds — 1 per profile (see PROFILE_ROTATION)
DEFAULT_QUESTIONS = 50                       # 50Q gives 5 MLR refits, more chances
                                             # for policy convergence (5-consec rule),
                                             # and tighter OPE-DR estimates than 30Q.
                                             # Cost impact for OpenRouter is negligible.
DEFAULT_PROFILE   = "random"                 # neutral profile — forces exploration
DEFAULT_CATEGORY  = "Penggalang"

# When --vary-profiles is on (default), each seed gets a different student archetype.
# This produces diverse reward trajectories across runs, making KT-AUC and OPE-DR
# more informative across LLMs. (CV remains degenerate due to MLR circularity in
# rl_metrics.fit_weights — see thesis Limitations.)
#
# The rotation is length-8 to align with DEFAULT_SEEDS (1 seed per profile, no cycling).
# Profiles are organised by cognitive dimension of CT (Brennan & Resnick / Wing):
#
#   LT structure:
#     Cognitive:  Pengenalan Pola (P_)  |  Dekomposisi (T_/G_)
#     Strategy:   Algoritmik (xA)  |  Abstraksi (xT/TA)  |  Generalisasi (xG/PG)
#     Approach:   Rekursi (_R)  |  Iterasi (_I)
#
#   Profil:
#   • random          — uniform 0.50 across all LTs (exploration baseline)
#   • volatile        — high variance (std ~0.20); simulates inconsistent student;
#                       flat mean keeps KT-AUC label-balanced
#   • pattern_lean    — strong on all Pengenalan Pola LTs (PAR, TAR, PAI, TAI);
#                       weak on Dekomposisi/Generalisasi
#   • decomp_lean     — strong on all Dekomposisi LTs (TGR, TGI, PGR, PGI);
#                       weak on Pengenalan Pola
#   • recursive_lean  — strong on all Rekursi LTs (PAR, TAR, TGR, PGR);
#                       weak on all Iterasi LTs
#   • iterative_lean  — strong on all Iterasi LTs (PAI, TAI, TGI, PGI);
#                       weak on all Rekursi LTs
#   • abstract_lean   — strong on Abstraksi LTs (TAR, TAI);
#                       mid on others — tests whether agent detects narrow mastery
#   • generalize_lean — strong on Generalisasi LTs (PGR, PGI);
#                       mid on others
#
# NOTE: "improving" profile was intentionally removed.  Its monotonically rising
# correctness rate caused BENAR to dominate the label distribution, making
# KT-AUC statistically unstable (minority_rate < KT_AUC_MIN_MINORITY_RATE) and
# kt_auc_stable to be None for many runs.  All 8 replacement profiles maintain
# a roughly flat mean correctness, so KT-AUC is interpretable across all seeds
# without requiring --balance-labels overrides.
PROFILE_ROTATION = [
    "random",
    "volatile",
    "pattern_lean",
    "decomp_lean",
    "recursive_lean",
    "iterative_lean",
    "abstract_lean",
    "generalize_lean",
]

# KT-AUC stability filter.  AUC requires both classes (correct + incorrect) to be
# represented; when one class is rare, AUC becomes statistically unstable and a
# single misranking can swing the score wildly.  A run is considered "stable"
# when the minority class (whichever of correct/incorrect is smaller) makes up at
# least this fraction of the run.  Below the threshold, kt_auc_stable is set to
# None and the aggregate stats fall back to runs that pass the filter.
KT_AUC_MIN_MINORITY_RATE = 0.20

# Optional label-balance guard for thesis experiments.
# When enabled, the simulator output is only minimally adjusted after the
# warm-up period so neither BENAR nor SALAH dominates the whole run.
# This makes KT-AUC statistically interpretable without hiding the adjustment:
# every override is logged in the per-step JSON.
BALANCE_TARGET_MINORITY_RATE = 0.30
BALANCE_WARMUP_QUESTIONS = 10

# Policy convergence rule.  The agent is considered "converged" at the first
# question Q where it has selected the SAME learning type for this many
# consecutive questions ending at Q.  This mirrors the rule used in
# simulate_rl/plots.py:1132-1134 (5-consecutive same-LT streak).  Unlike the
# weight-based CV convergence in evaluator.py, this is behavioral and does not
# depend on the (circular) MLR refit.
POLICY_CONVERGENCE_STREAK = 5

RESULTS_DIR = Path("results140726")
RUNS_DIR    = RESULTS_DIR / "runs"


# ──────────────────────────────────────────────────────────────────────────────
# LLM CLIENT — swappable per model
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """Unified interface for Ollama + OpenRouter chat completions."""

    def __init__(self, model_name: str, provider: str,
                 openrouter_api_key: Optional[str] = None,
                 ollama_base_url: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 timeout: int = 60):
        self.model_name = model_name
        self.provider = provider.lower()
        self.temperature = temperature
        self.timeout = timeout
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.ollama_base_url = ollama_base_url
        self.call_count = 0
        self.total_latency = 0.0

        if self.provider == "openrouter" and not self.openrouter_api_key:
            raise RuntimeError(
                f"Model '{model_name}' requires OpenRouter, but OPENROUTER_API_KEY "
                "is not set. Export it before running."
            )

    def chat(self, prompt: str, max_tokens: int = 512) -> str:
        """Single-turn chat completion. Returns the model's text reply."""
        t0 = time.time()
        try:
            if self.provider == "ollama":
                text = self._chat_ollama(prompt, max_tokens)
            elif self.provider == "openrouter":
                text = self._chat_openrouter(prompt, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        finally:
            self.call_count += 1
            self.total_latency += time.time() - t0
        return text

    def _chat_ollama(self, prompt: str, max_tokens: int) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self.ollama_base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return (data.get("response") or "").strip()

    def _chat_openrouter(self, prompt: str, max_tokens: int) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            # Ask OpenRouter to surface reasoning tokens so we can parse them
            # when content="" (standard for o-series / gpt-oss reasoning models).
            "include_reasoning": True,
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        msg = data["choices"][0]["message"]

        # Primary: use content field (non-reasoning models always populate this)
        content = (msg.get("content") or "").strip()
        if content:
            return content

        # Fallback: reasoning models (e.g. gpt-oss-120b, o3, o4-mini) on
        # OpenRouter return content="" and put the actual reply in one of:
        #   msg["reasoning"]           — OpenRouter standard field
        #   msg["reasoning_content"]   — alternative key used by some providers
        # We surface whichever is present so parse_judge_verdict can find
        # BENAR/SALAH inside the chain-of-thought.
        for key in ("reasoning", "reasoning_content"):
            reasoning = (msg.get(key) or "").strip()
            if reasoning:
                return reasoning

        return ""


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS — minimal, Indonesian, matched to thesis context
# ──────────────────────────────────────────────────────────────────────────────

TUTOR_PROMPT = """Anda adalah tutor mata kuliah Computational Thinking.
Topik fokus pada aspek pembelajaran: {lt_label}.

Berikan penjelasan singkat (maksimal 3 kalimat) tentang konsep berikut,
kemudian ajukan SATU pertanyaan lanjutan untuk menguji pemahaman mahasiswa.

Konteks soal: {question}

Format respons:
PENJELASAN: <penjelasan singkat>
PERTANYAAN: <satu pertanyaan lanjutan>"""

JUDGE_PROMPT = """Anda adalah penilai jawaban mahasiswa pada mata kuliah Computational Thinking.

Pertanyaan: {question}
Jawaban yang diharapkan: {reference}
Jawaban mahasiswa: {student_answer}

Tugas: tentukan apakah jawaban mahasiswa BENAR atau SALAH dibandingkan jawaban
yang diharapkan. Pertimbangkan kebenaran konsep, bukan persis kata-kata.

Jawab HANYA dengan satu kata: BENAR atau SALAH."""

# Mock student-answer templates — keep it simple and deterministic so the judge
# is the only LLM that interprets them. We use the ground-truth `is_correct_truth`
# from StudentSimulator to decide which template to emit. This is a methodological
# simplification: the judge sees text, not a label, so it still exercises grading.

CORRECT_ANSWER_TEMPLATES = [
    # Mirror the reference format: "X adalah Y. Contoh: ..." so the judge
    # can clearly verify conceptual correctness.
    "{topic} adalah {concept}. Contoh penerapannya: ketika menghadapi soal "
    "yang memerlukan {concept}, kita menggunakan pendekatan ini untuk "
    "menyelesaikan masalah secara sistematis dan terstruktur.",

    "Menurut pemahaman saya, {topic} mengacu pada {concept}. "
    "Dalam praktik Computational Thinking, konsep ini digunakan untuk "
    "memecah dan menyelesaikan masalah yang membutuhkan {concept} "
    "secara efisien dan bertahap.",

    "Konsep {topic} berarti {concept}. "
    "Sebagai contoh, pendekatan ini diterapkan saat kita perlu "
    "menyelesaikan masalah yang membutuhkan {concept}, "
    "sehingga solusi dapat ditemukan dengan cara yang logis dan terstruktur.",

    "{topic} dalam Computational Thinking didefinisikan sebagai {concept}. "
    "Penerapannya sangat berguna ketika kita menghadapi masalah yang "
    "memerlukan {concept} — dengan pendekatan ini kita dapat "
    "menguraikan masalah dan menemukan solusi secara sistematis.",
]
INCORRECT_ANSWER_TEMPLATES = [
    # Clearly wrong: off-topic, contradicts the concept, no example.
    "Saya rasa {topic} adalah tentang menulis kode program sebanyak mungkin "
    "tanpa memperhatikan struktur atau logika tertentu.",

    "{topic} menurut saya berarti menghafal semua langkah penyelesaian "
    "soal tanpa perlu memahami konsepnya, cukup ikuti rumus yang ada.",

    "Sepertinya {topic} sama saja dengan trial-and-error, yaitu mencoba "
    "semua kemungkinan jawaban secara acak sampai menemukan yang benar.",

    "Tidak terlalu yakin, tapi {topic} sepertinya berkaitan dengan "
    "kemampuan mengetik cepat dan menghafal sintaks bahasa pemrograman, "
    "bukan tentang cara berpikir atau strategi pemecahan masalah.",
]

# Simple per-LT topic anchors so the prompts have CT content
LT_TOPICS = {
    "PAR": ("Pengenalan Pola - Algoritmik - Rekursi",
            "kemampuan mengenali pola berulang dan menerapkan rekursi"),
    "TAR": ("Pengenalan Pola - Abstraksi - Rekursi",
            "menyaring detail penting dari pola yang berulang"),
    "PAI": ("Pengenalan Pola - Algoritmik - Iterasi",
            "mengidentifikasi pola untuk diselesaikan dengan iterasi"),
    "TAI": ("Pengenalan Pola - Abstraksi - Iterasi",
            "abstraksi pola yang dipecahkan secara iteratif"),
    "TGR": ("Dekomposisi - Algoritmik - Rekursi",
            "memecah masalah lalu menyelesaikannya secara rekursif"),
    "TGI": ("Dekomposisi - Algoritmik - Iterasi",
            "memecah masalah lalu menyelesaikan tiap bagian secara iteratif"),
    "PGR": ("Pengenalan Pola - Generalisasi - Rekursi",
            "menggeneralisasi pola dengan pendekatan rekursif"),
    "PGI": ("Pengenalan Pola - Generalisasi - Iterasi",
            "menggeneralisasi pola dengan pendekatan iteratif"),
}


def make_question_for(lt: str, q_num: int) -> Tuple[str, str]:
    """Return (question_text, reference_answer) for this LT and question number."""
    topic, concept = LT_TOPICS.get(lt, ("Computational Thinking", "berpikir komputasional"))
    question = (
        f"Soal #{q_num}: Jelaskan konsep '{topic}' dalam konteks Computational "
        f"Thinking, dan berikan contoh penerapannya."
    )
    reference = (
        f"{topic} adalah {concept}. Contoh: menerapkannya untuk menyelesaikan "
        f"soal yang membutuhkan {concept}."
    )
    return question, reference


def make_student_answer(lt: str, is_correct_truth: bool) -> str:
    """Materialise the ground-truth correctness into an answer string for the judge."""
    topic, concept = LT_TOPICS.get(lt, ("Computational Thinking", "berpikir komputasional"))
    pool = CORRECT_ANSWER_TEMPLATES if is_correct_truth else INCORRECT_ANSWER_TEMPLATES
    return random.choice(pool).format(topic=topic, concept=concept)


def parse_judge_verdict(text: str) -> Optional[bool]:
    """
    Parse BENAR/SALAH from judge reply.  Returns None if truly unparseable.

    Robust to:
      • Verbose preambles / reasoning ("Mari saya analisis... BENAR")
      • Markdown formatting (**BENAR**, `BENAR`, # BENAR)
      • Multi-line replies (verdict on any line)
      • Reasoning-style models that put the answer at the end
      • English fallbacks (CORRECT/INCORRECT/TRUE/FALSE/YES/NO)
      • Final-line verdict pattern ("...therefore the answer is BENAR")
    """
    if not text:
        return None
    t = text.strip().upper()
    if not t:
        return None

    # Strip common markdown/punctuation noise
    cleaned = re.sub(r"[*_`#>\[\](){}]", " ", t)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Find ALL occurrences of BENAR and SALAH across the entire reply,
    # not just in the first 30 characters.  Reasoning models often put the
    # final verdict at the end of a long chain-of-thought.
    benar_count = len(re.findall(r"\bBENAR\b", cleaned))
    salah_count = len(re.findall(r"\bSALAH\b", cleaned))

    if benar_count > 0 and salah_count == 0:
        return True
    if salah_count > 0 and benar_count == 0:
        return False
    if benar_count > 0 and salah_count > 0:
        # Both appeared — trust the LAST occurrence (reasoning-style models
        # state intermediate hypotheses then a final answer)
        last_benar = cleaned.rfind("BENAR")
        last_salah = cleaned.rfind("SALAH")
        return last_benar > last_salah

    # English fallbacks — use word boundaries to avoid CORRECT matching inside INCORRECT
    for pos_word, neg_word in [("CORRECT", "INCORRECT"), ("TRUE", "FALSE"), ("YES", "NO")]:
        pos_match = list(re.finditer(rf"\b{pos_word}\b", cleaned))
        neg_match = list(re.finditer(rf"\b{neg_word}\b", cleaned))
        pos_p = pos_match[-1].start() if pos_match else -1
        pos_n = neg_match[-1].start() if neg_match else -1
        if pos_p >= 0 and pos_n < 0:
            return True
        if pos_n >= 0 and pos_p < 0:
            return False
        if pos_p >= 0 and pos_n >= 0:
            return pos_p > pos_n

    return None


def compute_cv_rewards(rewards: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    CV of the reward signal itself: std(r) / |mean(r)|.

    Matches the convention used in the supervisor's table.  Unlike the
    weight-based CV in evaluator.py, this is NOT affected by the MLR
    circularity issue, because it is computed directly from observed rewards.

    Returns
    -------
    (cv, mean, std).  cv is None when |mean| is too small for a stable ratio.
    """
    if len(rewards) < 2:
        return None, (rewards[0] if rewards else None), None
    arr = np.asarray(rewards, dtype=float)
    mean = float(arr.mean())
    std  = float(arr.std(ddof=1))
    if abs(mean) < 1e-9:
        return None, mean, std
    return std / abs(mean), mean, std


def compute_policy_convergence(
    step_log: List[Dict],
    streak: int = POLICY_CONVERGENCE_STREAK,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Detect the first question Q at which the agent has selected the SAME
    learning type for `streak` consecutive questions ending at Q.

    Mirrors the rule used in simulate_rl/plots.py:1132-1134.

    Returns
    -------
    (converged, convergence_q, convergence_lt).
    """
    lts = [s.get("lt") or s.get("learning_type") for s in step_log]
    if len(lts) < streak:
        return False, None, None
    for i in range(streak - 1, len(lts)):
        window = lts[i - streak + 1 : i + 1]
        if all(x is not None for x in window) and len(set(window)) == 1:
            # i is 0-indexed; convergence question number is i+1
            return True, i + 1, window[0]
    return False, None, None


def apply_label_balance_guard(
    raw_truth: bool,
    previous_truths: List[bool],
    target_minority_rate: float = BALANCE_TARGET_MINORITY_RATE,
    warmup_questions: int = BALANCE_WARMUP_QUESTIONS,
) -> Tuple[bool, bool]:
    """
    Keep the simulated ground-truth labels from becoming too one-sided.

    This is not used to improve model scores. It only prevents KT-AUC from
    becoming undefined/unstable when almost every simulated answer is BENAR or
    almost every answer is SALAH. The adjustment starts after a warm-up period
    and is logged as `balance_override=True` in the step log.

    Returns
    -------
    (final_truth, was_overridden).
    """
    if len(previous_truths) < warmup_questions:
        return raw_truth, False

    pos_rate = sum(previous_truths) / len(previous_truths)
    max_rate = 1.0 - target_minority_rate

    # If BENAR already dominates, force the next label to SALAH when needed.
    if pos_rate > max_rate and raw_truth is True:
        return False, True

    # If SALAH already dominates, force the next label to BENAR when needed.
    if pos_rate < target_minority_rate and raw_truth is False:
        return True, True

    return raw_truth, False


# ──────────────────────────────────────────────────────────────────────────────
# CORE LOOP — one session for one (model, seed)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    model: str
    provider: str
    judge_model: str
    judge_provider: str
    label_source: str
    balance_labels: bool
    balance_overrides: int
    seed: int
    n_questions: int
    profile: str
    category: str
    timestamp: str

    # RL metrics
    kt_auc: Optional[float]
    kt_auc_stable: Optional[float]   # kt_auc, but None when label balance too skewed
    kt_verdict: str
    # CV dari bobot α/β/γ antar MLR refit (v3: ini yang benar untuk stabilitas parameter)
    # Referensi: Reed et al. (2002). CV < 0.30 → konvergen.
    # Jika None: bobot tidak berubah dari inisialisasi (mean≈0 → CV tidak terdefinisi).
    cv_alpha: Optional[float]
    cv_beta: Optional[float]
    cv_gamma: Optional[float]
    cv_max: Optional[float]
    converged: Optional[bool]
    # CV reward per-step (dilaporkan sebagai diagnostic, BUKAN metrik stabilitas utama).
    # CV reward selalu >>0.30 karena μ_reward kecil (~0.06) — ini artefak matematis.
    # Untuk stabilitas sistem, gunakan cv_alpha/cv_beta/cv_gamma di atas.
    cv_rewards: Optional[float]
    reward_mean: Optional[float]
    reward_std:  Optional[float]
    ope_dr_value: Optional[float]
    ope_baseline: Optional[float]

    # Policy convergence — independent of CV.  First question at which the agent
    # selects the SAME learning type for POLICY_CONVERGENCE_STREAK questions in
    # a row.  Mirrors the convergence rule in simulate_rl/plots.py (lines 1132-1134).
    policy_converged: bool                 # did it converge at all?
    policy_convergence_q: Optional[int]    # question number where streak first completed
    policy_convergence_lt: Optional[str]   # which LT it converged on
    final_lt: Optional[str]                # last LT chosen (sanity check)

    # Label balance — diagnostic for KT-AUC reliability
    n_correct: int
    n_incorrect: int
    pos_rate: float                  # n_correct / n_total
    minority_rate: float             # min(pos_rate, 1-pos_rate)
    kt_auc_passes_filter: bool       # minority_rate >= KT_AUC_MIN_MINORITY_RATE

    # Judge / LLM quality
    judge_agreement: float       # fraction of judge verdicts matching ground truth
    judge_parse_failures: int    # how often we couldn't parse BENAR/SALAH
    total_tutor_calls: int
    total_judge_calls: int
    tutor_latency_s: float
    judge_latency_s: float

    # Errors
    errors: List[str]


def run_one(
    model_name: str,
    provider: str,
    seed: int,
    n_questions: int = DEFAULT_QUESTIONS,
    profile: str = DEFAULT_PROFILE,
    category: str = DEFAULT_CATEGORY,
    verbose: bool = True,
    openrouter_api_key: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    judge_provider: Optional[str] = None,
    label_source: str = "judge",
    balance_labels: bool = False,
    balance_target_minority_rate: float = BALANCE_TARGET_MINORITY_RATE,
) -> Tuple[RunResult, List[Dict]]:
    """Run one model with one seed and return (summary, step_log)."""
    # Deterministic across student + LT-template choice
    random.seed(seed)
    np.random.seed(seed)

    tutor = LLMClient(model_name, provider, openrouter_api_key)
    effective_judge_model = judge_model_name or model_name
    effective_judge_provider = (judge_provider or provider).lower()
    judge = LLMClient(effective_judge_model, effective_judge_provider, openrouter_api_key, temperature=0.0)
    # Fresh registry + agent per run — critical for independence
    registry = SessionRegistry()
    session_id = f"{model_name}_seed{seed}"
    # Sanitise for filesystem
    safe_session_id = session_id.replace(":", "_").replace("/", "_")
    agent = registry.get_agent(safe_session_id, category=category)

    student = StudentSimulator(profile, category=category, ai_sim=False)

    step_log: List[Dict] = []
    judge_agreements = 0
    judge_parse_failures = 0
    errors: List[str] = []
    tutor_calls = 0
    judge_calls = 0
    balanced_truths_so_far: List[bool] = []
    balance_overrides = 0

    if label_source not in {"judge", "truth"}:
        raise ValueError("label_source must be either 'judge' or 'truth'")

    for q_num in range(1, n_questions + 1):
        try:
            # 1. RL agent selects LT
            lt, _ = agent.select_action()

            # 2. Build question for this LT
            question, reference = make_question_for(lt, q_num)

            # 3. TUTOR LLM generates an explanation + follow-up
            #    (We don't use the follow-up further — this just exercises the tutor role
            #     to measure latency and ensure the model is actually invoked.)
            topic, _ = LT_TOPICS.get(lt, ("Computational Thinking", ""))
            try:
                tutor.chat(
                    TUTOR_PROMPT.format(lt_label=topic, question=question),
                    max_tokens=256,
                )
                tutor_calls += 1
            except Exception as exc:
                errors.append(f"q{q_num} tutor: {exc}")
                # Continue — RL update still happens with rule-based ground truth

            # 4. Rule-based student decides ground-truth correctness.
            #    Optional balance guard prevents KT-AUC from becoming meaningless
            #    when one label class dominates. Overrides are transparent and logged.
            raw_truth, n_attempt, t_ans = student.answer(lt)
            if balance_labels:
                is_correct_truth, was_balanced = apply_label_balance_guard(
                    raw_truth,
                    balanced_truths_so_far,
                    target_minority_rate=balance_target_minority_rate,
                )
                if was_balanced:
                    balance_overrides += 1
            else:
                is_correct_truth, was_balanced = raw_truth, False
            balanced_truths_so_far.append(is_correct_truth)

            # 5. Materialise an answer string for the judge to grade
            student_answer = make_student_answer(lt, is_correct_truth)

            # 6. JUDGE LLM grades the answer
            is_correct_for_rl = is_correct_truth  # fallback if judge fails
            judge_verdict: Optional[bool] = None
            judge_text: str = ""
            try:
                judge_text = judge.chat(
                    JUDGE_PROMPT.format(
                        question=question,
                        reference=reference,
                        student_answer=student_answer,
                    ),
                    # 512 tokens: reasoning models (gpt-oss-120b, o3, o4-mini)
                    # emit a chain-of-thought before their final BENAR/SALAH
                    # verdict. 64 was too small — they'd get cut off mid-think
                    # and return content="" with a truncated reasoning field.
                    # parse_judge_verdict already handles finding the last
                    # BENAR/SALAH anywhere in a long response.
                    max_tokens=512,
                )
                judge_calls += 1
                judge_verdict = parse_judge_verdict(judge_text)
                if judge_verdict is None:
                    judge_parse_failures += 1
                    # Fall back to ground truth so RL keeps a signal
                    is_correct_for_rl = is_correct_truth
                else:
                    is_correct_for_rl = judge_verdict
                    if judge_verdict == is_correct_truth:
                        judge_agreements += 1
            except Exception as exc:
                errors.append(f"q{q_num} judge: {exc}")
                judge_parse_failures += 1

            # 7. Feed the selected label source into the RL agent.
            #    label_source="judge" preserves the original design.
            #    label_source="truth" makes RL evaluation independent from judge bias,
            #    while judge agreement is still reported separately.
            if label_source == "truth":
                is_correct_for_rl = is_correct_truth

            mastery_code = f"{agent.mastery_levels[lt]}{lt}"
            step = agent.record_response(
                learning_type=lt,
                is_correct=is_correct_for_rl,
                n_attempt=n_attempt,
                t_answer_seconds=t_ans,
                mastery_code=mastery_code,
            )

            # 8. Augment step with our diagnostics — include raw judge text so
            #    parse failures are diagnosable post-hoc
            step["q_num"] = q_num
            step["lt"] = lt
            step["raw_is_correct_truth"] = raw_truth
            step["is_correct_truth"] = is_correct_truth
            step["balance_override"] = was_balanced
            step["label_source"] = label_source
            step["is_correct_judge"] = judge_verdict
            step["judge_agrees"] = (judge_verdict == is_correct_truth)
            step["judge_raw_text"] = judge_text[:500] if judge_text else ""  # cap to keep JSON small
            step_log.append(step)

            if verbose and q_num % 5 == 0:
                print(f"    q{q_num:>3}/{n_questions}  lt={lt}  "
                      f"truth={'✓' if is_correct_truth else '✗'}  "
                      f"judge={'✓' if judge_verdict else ('✗' if judge_verdict is False else '?')}  "
                      f"reward={step['reward']:+.3f}  ε={agent.epsilon:.3f}")

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            errors.append(f"q{q_num} loop: {exc}\n{traceback.format_exc()}")
            if verbose:
                print(f"    q{q_num}: ERROR — {exc}")

    # ─── Compute the 3 RL evaluation metrics ──────────────────────────────────
    try:
        kt = evaluate_kt_auc(step_log)
    except Exception as exc:
        kt = {"auc": None, "verdict": f"ERROR: {exc}"}
        errors.append(f"kt_auc: {exc}")

    try:
        rd = evaluate_reward_decomposition(step_log, agent)
    except Exception as exc:
        rd = {}
        errors.append(f"reward_decomp: {exc}")

    try:
        dr = evaluate_ope_dr(step_log, agent)
    except Exception as exc:
        dr = {}
        errors.append(f"ope_dr: {exc}")

    # Robust extraction — keys come from evaluation/evaluator.py return dicts
    stability = rd.get("stability", {}) if isinstance(rd, dict) else {}
    cv_alpha = stability.get("cv_alpha")
    cv_beta  = stability.get("cv_beta")
    cv_gamma = stability.get("cv_gamma")

    # v3: CV bobot MLR (bukan reward). None berarti bobot tidak berubah dari
    # inisialisasi (mean≈0 → CV tidak terdefinisi, tapi bukan berarti tidak stabil).
    # Untuk cv_max: None dianggap 0.0 (tidak ada variasi = paling konvergen).
    cv_vals_defined = [v for v in (cv_alpha, cv_beta, cv_gamma) if v is not None]
    cv_max = max(cv_vals_defined) if cv_vals_defined else 0.0  # 0.0 = semua stabil
    converged = stability.get("overall_stable")
    if converged is None:
        # Fallback: jika evaluator tidak set overall_stable, inferensi dari CV
        # Semua None → tidak ada fluktuasi → konvergen
        converged = (cv_max < 0.30) if cv_vals_defined else True

    ope_dr_value = _safe_get(dr, "v_hat_dr")
    ope_baseline = _safe_get(dr, "v_hat_logged")

    n_judge_attempts = max(1, judge_calls)
    judge_agreement_rate = judge_agreements / n_judge_attempts

    # Label balance diagnostics — use the labels the RL agent actually saw
    # (is_correct in step_log), since that's what KT-AUC was computed against.
    labels = [s.get("is_correct") for s in step_log if s.get("is_correct") is not None]
    n_correct   = sum(1 for x in labels if x)
    n_incorrect = sum(1 for x in labels if not x)
    n_total     = n_correct + n_incorrect
    pos_rate      = (n_correct / n_total) if n_total else 0.0
    minority_rate = min(pos_rate, 1.0 - pos_rate) if n_total else 0.0
    passes_filter = (minority_rate >= KT_AUC_MIN_MINORITY_RATE)

    kt_auc_raw = _safe_get(kt, "auc")
    kt_auc_stable = kt_auc_raw if (passes_filter and kt_auc_raw is not None) else None

    # CV reward per-step — dilaporkan sebagai diagnostic saja (bukan metrik stabilitas).
    # Nilai selalu >>0.30 karena μ_reward kecil (~0.06); ini artefak matematis.
    # Untuk stabilitas sistem, gunakan cv_alpha/cv_beta/cv_gamma dari bobot MLR.
    rewards = [s.get("reward") for s in step_log if isinstance(s.get("reward"), (int, float))]
    cv_rewards, reward_mean, reward_std = compute_cv_rewards(rewards)

    # Policy convergence — 5-consecutive same-LT rule
    pol_conv, pol_conv_q, pol_conv_lt = compute_policy_convergence(step_log)
    final_lt = step_log[-1].get("lt") or step_log[-1].get("learning_type") if step_log else None

    result = RunResult(
        model=model_name,
        provider=provider,
        judge_model=effective_judge_model,
        judge_provider=effective_judge_provider,
        label_source=label_source,
        balance_labels=balance_labels,
        balance_overrides=balance_overrides,
        seed=seed,
        n_questions=n_questions,
        profile=profile,
        category=category,
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        kt_auc=kt_auc_raw,
        kt_auc_stable=kt_auc_stable,
        kt_verdict=str(_safe_get(kt, "verdict") or ""),
        cv_alpha=cv_alpha,
        cv_beta=cv_beta,
        cv_gamma=cv_gamma,
        cv_max=cv_max,
        converged=converged,
        cv_rewards=cv_rewards,
        reward_mean=reward_mean,
        reward_std=reward_std,
        ope_dr_value=ope_dr_value,
        ope_baseline=ope_baseline,
        policy_converged=pol_conv,
        policy_convergence_q=pol_conv_q,
        policy_convergence_lt=pol_conv_lt,
        final_lt=final_lt,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        pos_rate=pos_rate,
        minority_rate=minority_rate,
        kt_auc_passes_filter=passes_filter,
        judge_agreement=judge_agreement_rate,
        judge_parse_failures=judge_parse_failures,
        total_tutor_calls=tutor_calls,
        total_judge_calls=judge_calls,
        tutor_latency_s=tutor.total_latency,
        judge_latency_s=judge.total_latency,
        errors=errors,
    )

    return result, step_log


def _safe_get(d: Any, *keys: str) -> Any:
    """Return d[k] for the first k in keys that exists in d; else None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENT DRIVER + STATS
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    models: List[Dict[str, str]],
    seeds: List[int],
    n_questions: int,
    profile: Optional[str] = None,
    profile_rotation: Optional[List[str]] = None,
    category: str = DEFAULT_CATEGORY,
    verbose: bool = True,
    openrouter_api_key: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    judge_provider: Optional[str] = None,
    label_source: str = "judge",
    balance_labels: bool = False,
    balance_target_minority_rate: float = BALANCE_TARGET_MINORITY_RATE,
) -> List[RunResult]:
    """
    Run the LLM comparison experiment.

    Profile selection:
      - If `profile` is set (string), every seed uses that single profile.
      - Else if `profile_rotation` is set (list), seed i uses rotation[i % len(rotation)].
      - Else falls back to DEFAULT_PROFILE for every seed.

    The same (seed → profile) mapping is used across ALL models, so the matched-seed
    design holds: each model faces an identical sequence of student behaviors.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: List[RunResult] = []

    # Build seed → profile mapping once, deterministically
    if profile is not None:
        seed_to_profile = {s: profile for s in seeds}
    elif profile_rotation:
        seed_to_profile = {s: profile_rotation[i % len(profile_rotation)]
                           for i, s in enumerate(seeds)}
    else:
        seed_to_profile = {s: DEFAULT_PROFILE for s in seeds}

    print("Seed → profile mapping:")
    for s in seeds:
        print(f"  seed {s} → {seed_to_profile[s]}")
    print()

    for m in models:
        for seed in seeds:
            seed_profile = seed_to_profile[seed]
            tag = f"{m['name'].replace(':', '_').replace('/', '_')}_seed{seed}_{seed_profile}"
            print(f"\n━━━ {tag} ━━━")
            try:
                result, steps = run_one(
                    model_name=m["name"],
                    provider=m["provider"],
                    seed=seed,
                    n_questions=n_questions,
                    profile=seed_profile,
                    category=category,
                    verbose=verbose,
                    openrouter_api_key=openrouter_api_key,
                    judge_model_name=judge_model_name,
                    judge_provider=judge_provider,
                    label_source=label_source,
                    balance_labels=balance_labels,
                    balance_target_minority_rate=balance_target_minority_rate,
                )
            except Exception as exc:
                print(f"  FATAL: {exc}")
                traceback.print_exc()
                continue

            # Persist this run
            with open(RUNS_DIR / f"{tag}.json", "w", encoding="utf-8") as f:
                json.dump({"summary": asdict(result), "steps": steps},
                          f, indent=2, ensure_ascii=False, default=str)

            filter_flag = "✓" if result.kt_auc_passes_filter else "✗ (skipped)"
            cv_r = f"{result.cv_rewards:.3f}" if result.cv_rewards is not None else "None"
            cv_w = f"{result.cv_max:.4f}" if result.cv_max is not None else "None"
            conv_str = (f"Q{result.policy_convergence_q}({result.policy_convergence_lt})"
                        if result.policy_converged else "no")
            print(f"  ✓ kt_auc={result.kt_auc}  cv_bobot={cv_w}(α/β/γ)  cv_reward={cv_r}(diagnostic)  "
                  f"ope_dr={result.ope_dr_value}  "
                  f"judge_agree={result.judge_agreement:.2%}  "
                  f"conv={conv_str}  "
                  f"labels={result.n_correct}/{result.n_incorrect} {filter_flag}  "
                  f"balance_overrides={result.balance_overrides}  "
                  f"errors={len(result.errors)}")
            all_results.append(result)

    return all_results


def write_summary_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) if results else []
    # Drop 'errors' from csv (it's a list); kept in JSON
    fields = [f for f in fields if f != "errors"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k in fields}
            w.writerow(row)


def write_stats(results: List[RunResult], path: Path) -> Dict[str, Any]:
    """Friedman + pairwise Wilcoxon across models per metric."""
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except ImportError:
        print("⚠ scipy not installed — skipping statistical tests. "
              "Install with: pip install scipy")
        return {}

    by_model: Dict[str, Dict[str, List[float]]] = {}
    # Metrik pembanding antar model (berdasarkan analisis variabilitas empiris):
    # - kt_auc / kt_auc_stable : range 0.18 antar model → pembeda UTAMA
    # - policy_convergence_q   : range 6.4 pertanyaan → pembeda kuat
    # - judge_agreement        : range ~10% → pembeda sedang
    # - reward_mean            : range 0.007 → pembeda lemah tapi valid
    # - ope_dr_value           : selisih pure-exploit vs logging policy
    # - cv_max                 : selalu 0.0 → TIDAK dipakai sebagai pembeda,
    #                            dilaporkan sebagai temuan "semua model konvergen"
    metrics = ["kt_auc", "kt_auc_stable",
               "reward_mean", "reward_std",
               "ope_dr_value",
               "judge_agreement",
               "policy_convergence_q"]
    for r in results:
        by_model.setdefault(r.model, {m: [] for m in metrics})
        for m in metrics:
            v = getattr(r, m)
            if v is not None:
                by_model[r.model][m].append(float(v))

    # Per-model filter coverage — how many runs survived the label-balance check
    filter_coverage: Dict[str, Dict[str, int]] = {}
    for r in results:
        d = filter_coverage.setdefault(r.model, {"total": 0, "passed": 0})
        d["total"] += 1
        if r.kt_auc_passes_filter:
            d["passed"] += 1

    # Per-model policy convergence summary
    convergence_summary: Dict[str, Dict[str, Any]] = {}
    for r in results:
        d = convergence_summary.setdefault(
            r.model, {"total": 0, "converged": 0, "conv_qs": []}
        )
        d["total"] += 1
        if r.policy_converged:
            d["converged"] += 1
            if r.policy_convergence_q is not None:
                d["conv_qs"].append(r.policy_convergence_q)
    for m, d in convergence_summary.items():
        d["convergence_rate"] = d["converged"] / d["total"] if d["total"] else 0.0
        d["mean_conv_q"] = float(np.mean(d["conv_qs"])) if d["conv_qs"] else None

    out: Dict[str, Any] = {
        "kt_auc_filter": {
            "min_minority_rate": KT_AUC_MIN_MINORITY_RATE,
            "per_model_coverage": filter_coverage,
        },
        "policy_convergence": {
            "streak": POLICY_CONVERGENCE_STREAK,
            "per_model": convergence_summary,
        },
        "per_metric": {},
    }
    for metric in metrics:
        # Matched samples: each seed contributes one value per model
        models_with_data = [m for m, d in by_model.items()
                            if len(d[metric]) >= 3]
        if len(models_with_data) < 3:
            out["per_metric"][metric] = {"note": "need ≥3 models with ≥3 seeds"}
            continue
        # Align by min-length across models
        min_n = min(len(by_model[m][metric]) for m in models_with_data)
        arrays = [by_model[m][metric][:min_n] for m in models_with_data]
        try:
            stat, p = friedmanchisquare(*arrays)
            entry = {
                "friedman_stat": float(stat),
                "friedman_p":    float(p),
                "n_per_model":   min_n,
                "models":        models_with_data,
                "means":         {m: float(np.mean(by_model[m][metric][:min_n]))
                                  for m in models_with_data},
                "stds":          {m: float(np.std(by_model[m][metric][:min_n], ddof=1))
                                  if min_n > 1 else 0.0
                                  for m in models_with_data},
            }
            # Pairwise Wilcoxon (if Friedman significant or just informative)
            pairs: Dict[str, Any] = {}
            for i, a in enumerate(models_with_data):
                for b in models_with_data[i+1:]:
                    xs = by_model[a][metric][:min_n]
                    ys = by_model[b][metric][:min_n]
                    try:
                        w_stat, w_p = wilcoxon(xs, ys)
                        pairs[f"{a} vs {b}"] = {"w": float(w_stat), "p": float(w_p)}
                    except Exception as exc:
                        pairs[f"{a} vs {b}"] = {"error": str(exc)}
            entry["pairwise_wilcoxon"] = pairs
            out["per_metric"][metric] = entry
        except Exception as exc:
            out["per_metric"][metric] = {"error": str(exc)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="Single model, single seed, 8 questions — for testing")
    p.add_argument("--full",  action="store_true",
                   help="Run the default thesis matrix (6 models × 5 seeds × 30 Q)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Override models, e.g. --models llama3.1:8b mistral:7b")
    p.add_argument("--providers", nargs="+", default=None,
                   help="Provider per model (ollama|openrouter), aligned to --models")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--questions", type=int, default=None)
    p.add_argument("--profile", type=str, default=None,
                   choices=sorted(PROFILES.keys()),
                   help="Force a SINGLE profile across all seeds. If omitted, "
                        "profiles rotate per seed (PROFILE_ROTATION).")
    p.add_argument("--no-vary-profiles", action="store_true",
                   help="Disable profile rotation and use the legacy default "
                        "(random profile for every seed). Equivalent to --profile random.")
    p.add_argument("--category", type=str, default=DEFAULT_CATEGORY)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--openrouter-key", type=str, default=None)
    p.add_argument("--fixed-judge-model", type=str, default=None,
                   help="Use one fixed judge model for all tutor models, e.g. grok-4.1-fast")
    p.add_argument("--fixed-judge-provider", type=str, default=None,
                   choices=["ollama", "openrouter"],
                   help="Provider for --fixed-judge-model")
    p.add_argument("--label-source", type=str, default="judge", choices=["judge", "truth"],
                   help="Label used by the RL agent: judge verdict or simulator ground truth")
    p.add_argument("--balance-labels", action="store_true",
                   help="Enable transparent label-balance guard for stable KT-AUC")
    p.add_argument("--balance-target-minority-rate", type=float,
                   default=BALANCE_TARGET_MINORITY_RATE,
                   help="Minimum target share for the minority class when --balance-labels is enabled")
    args = p.parse_args()

    # Resolve config
    if args.smoke:
        models = [DEFAULT_MODELS[0]]  # llama3.1:8b — local Ollama
        seeds = [42]
        n_q = 8
    elif args.full:
        models = DEFAULT_MODELS
        seeds = args.seeds if args.seeds else DEFAULT_SEEDS  # honour explicit --seeds
        n_q = DEFAULT_QUESTIONS
    else:
        if args.models:
            providers = args.providers or ["ollama"] * len(args.models)
            if len(providers) != len(args.models):
                p.error("--providers length must match --models")
            models = [{"name": m, "provider": pr}
                      for m, pr in zip(args.models, providers)]
        else:
            models = DEFAULT_MODELS
        seeds = args.seeds or DEFAULT_SEEDS
        n_q = args.questions or DEFAULT_QUESTIONS

    # Resolve profile strategy
    if args.profile:
        forced_profile = args.profile
        rotation = None
        profile_desc = f"forced single profile: {args.profile}"
    elif args.no_vary_profiles:
        forced_profile = "random"
        rotation = None
        profile_desc = "legacy: random profile for all seeds"
    elif args.smoke:
        forced_profile = "random"
        rotation = None
        profile_desc = "smoke: random profile"
    else:
        forced_profile = None
        rotation = PROFILE_ROTATION
        profile_desc = f"rotating: {PROFILE_ROTATION}"

    print(f"Models   : {[m['name'] for m in models]}")
    print(f"Seeds    : {seeds}")
    print(f"Questions: {n_q}")
    print(f"Profile  : {profile_desc}")
    print(f"Judge    : {args.fixed_judge_model or 'same as tutor'}")
    print(f"Labels   : {args.label_source}" +
          (f"; balanced minority target ≥ {args.balance_target_minority_rate:.0%}" if args.balance_labels else ""))
    print(f"Output   : {RESULTS_DIR.resolve()}")
    print()

    results = run_experiment(
        models=models,
        seeds=seeds,
        n_questions=n_q,
        profile=forced_profile,
        profile_rotation=rotation,
        category=args.category,
        verbose=not args.quiet,
        openrouter_api_key=args.openrouter_key,
        judge_model_name=args.fixed_judge_model,
        judge_provider=args.fixed_judge_provider,
        label_source=args.label_source,
        balance_labels=args.balance_labels,
        balance_target_minority_rate=args.balance_target_minority_rate,
    )

    if not results:
        print("\nNo successful runs — nothing to summarise.")
        return

    summary_csv = RESULTS_DIR / "summary.csv"
    stats_json = RESULTS_DIR / "stats.json"
    write_summary_csv(results, summary_csv)
    stats = write_stats(results, stats_json)

    print(f"\n━━━ DONE ━━━")
    print(f"Per-run JSON : {RUNS_DIR}/")
    print(f"Summary CSV  : {summary_csv}")
    print(f"Stats JSON   : {stats_json}")

    # Label-balance filter coverage report
    cov = stats.get("kt_auc_filter", {}).get("per_model_coverage", {})
    if cov:
        thresh = stats["kt_auc_filter"]["min_minority_rate"]
        print(f"\nKT-AUC stability filter (minority class ≥ {thresh:.0%}):")
        for m, d in cov.items():
            tot, pas = d["total"], d["passed"]
            pct = (pas / tot * 100) if tot else 0
            note = "" if pas == tot else "  ← some runs dropped from kt_auc_stable"
            print(f"  {m:<28}  {pas}/{tot} runs passed ({pct:.0f}%){note}")

    # Policy convergence report (5-consec same-LT rule)
    pol = stats.get("policy_convergence", {}).get("per_model", {})
    if pol:
        streak = stats["policy_convergence"]["streak"]
        print(f"\nPolicy convergence ({streak} consecutive same-LT):")
        for m, d in pol.items():
            conv_q = f"mean Q{d['mean_conv_q']:.1f}" if d['mean_conv_q'] is not None else "—"
            print(f"  {m:<28}  {d['converged']}/{d['total']} converged "
                  f"({d['convergence_rate']:.0%})  {conv_q}")

    # ── Ringkasan metrik pembanding antar model ───────────────────────────
    METRIC_LABELS = {
        "kt_auc":               "KT-AUC            [PEMBANDING UTAMA]",
        "kt_auc_stable":        "KT-AUC (filtered) [PEMBANDING UTAMA]",
        "reward_mean":          "Reward Mean        [pembanding sekunder]",
        "reward_std":           "Reward Std         [diagnostic]",
        "ope_dr_value":         "OPE-DR Value       [pembanding sekunder]",
        "judge_agreement":      "Judge Agreement    [pembanding sekunder]",
        "policy_convergence_q": "Convergence (Q)    [pembanding sekunder]",
    }
    NOT_USEFUL = {"cv_max", "cv_rewards"}

    print("\n" + "="*65)
    print("RINGKASAN METRIK PEMBANDING ANTAR MODEL")
    print("="*65)
    print("Catatan: cv_max selalu 0.0 (semua model konvergen) — tidak ditampilkan")
    for metric, entry in stats.get("per_metric", {}).items():
        if "means" not in entry or metric in NOT_USEFUL:
            continue
        label = METRIC_LABELS.get(metric, metric)
        fp  = entry.get("friedman_p")
        sig = " ★ signifikan" if isinstance(fp, float) and fp < 0.05 else ""
        fp_str = f"Friedman p={fp:.4f}{sig}" if isinstance(fp, float) else ""
        print(f"\n  {label}")
        if fp_str:
            print(f"  {fp_str}")
        ascending = metric in {"policy_convergence_q", "reward_std"}
        ranked = sorted(
            entry["models"],
            key=lambda m: entry["means"].get(m, 0),
            reverse=not ascending,
        )
        for rank, m in enumerate(ranked, 1):
            mean = entry["means"][m]
            std  = entry["stds"][m]
            marker = " ← terbaik" if rank == 1 else ""
            print(f"    #{rank} {m:<26}  {mean:+.4f} ± {std:.4f}{marker}")

if __name__ == "__main__":
    main()
