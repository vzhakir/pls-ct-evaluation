# pls-ct-evaluation

Kumpulan kode evaluasi untuk chatbot Personalized Learning System (PLS) mata kuliah Computational Thinking (KOM2102) di IPB University. Ketiga folder di repo ini menguji tiga pendekatan berbeda untuk chatbot LLM yang sama, yaitu RAG, Prompt Engineering, dan Reinforcement Learning, dan masing masing menjadi bagian dari skripsi terpisah.

---

## Struktur Repo

```
pls-ct-evaluation/
│
├── pls-pe/     ← Evaluasi Prompt Engineering (zero-shot, few-shot, CoT)
│                  Metrik: ROUGE-1/2/L, BERTScore F1
│
├── pls-rag/    ← Evaluasi RAG (retrieval + generation)
│                  Metrik: Faithfulness (RAGAS), Hallucination Risk (SelfCheckGPT),
│                          Answer Quality (MT-Bench), Precision@K, Recall@K, Coverage
│
└── pls-rl/     ← Evaluasi Reinforcement Learning (adaptive tutor policy)
                   Metrik: KT-AUC, Reward Decomposition CV, OPE Doubly Robust
```

Tiap folder punya `README.md` sendiri dengan detail lengkap cara setup, cara menjalankan, dan penjelasan tiap metrik. README ini hanya memberi gambaran umum dan penghubung antar folder.

---

## Ringkasan Tiap Folder

### `pls-pe` — Prompt Engineering Evaluation

Menguji tiga strategi prompt (zero-shot, few-shot, Chain-of-Thought) dengan konteks retrieval yang sama, lalu membandingkan jawaban model terhadap ground truth memakai ROUGE dan BERTScore. Berisi backend FastAPI (`api.py`), generator jawaban (`test_prompt.py`), dan evaluator (`prompt_eval.py`).

### `pls-rag` — RAG Evaluation Suite

Mengevaluasi sistem RAG dari sisi retrieval (relevansi chunk yang diambil) maupun generation (faktualitas dan kualitas jawaban), dibandingkan dengan baseline tanpa RAG. Backend FastAPI (`api.py`) sama arsitekturnya dengan `pls-pe`, dengan evaluator utama di `rag.py` dan 75 test case di `test_cases.py`.

### `pls-rl` — RL Evaluation Harness

Mengevaluasi kebijakan RL yang memilih jalur pembelajaran (learning trajectory) secara adaptif berdasarkan performa mahasiswa, disimulasikan lewat `simulate_rl/` dan dibandingkan lintas enam model LLM lewat `compare.py`.

---

## Arsitektur yang Dipakai Bersama

`pls-pe` dan `pls-rag` masing masing punya salinan `api.py` yang hampir identik, backend FastAPI dengan RAG selalu aktif dan 48 kombinasi tipe kognitif (format `{1-6}{P|T}{A|G}{I|R}`, mewakili level materi, gaya Personal/Team, Auditori/Ganda, dan Individual/Rutin). Backend ini mendukung tiga penyedia LLM secara hybrid (Ollama lokal, OpenRouter, ChatAnywhere) dengan fallback otomatis, dan menyediakan endpoint `/chat`, `/retrieve`, `/evaluate`, `/history`, serta `/download-history`.

Folder `materials/` di `pls-pe` dan `pls-rag` berisi konten yang sama persis, materi CT per kode LT dan level, plus file ground truth (`GT_CT01` sampai `GT_CT13`, `GT_DETAIL_PT01` sampai `PT13`, `GT_SUBTOPIK_01` sampai `14`, dan beberapa `ground_truth_KOM2102_*`). `pls-rl` punya `materials/` sendiri dengan materi serupa, disesuaikan untuk kebutuhan simulasi RL.

---

## Mulai Dari Mana

Untuk menjalankan salah satu evaluasi, masuk ke foldernya masing masing dan ikuti README di dalamnya.

```bash
cd pls-pe   && cat README.md   # evaluasi Prompt Engineering
cd pls-rag  && cat README.md   # evaluasi RAG
cd pls-rl   && cat README.md   # evaluasi RL
```

Ketiganya independen satu sama lain, install dependency dan jalankan backend FastAPI masing masing folder secara terpisah (kecuali `pls-rl` yang tidak butuh backend FastAPI, karena kebijakan RL disimulasikan langsung lewat `simulate_rl/`).
