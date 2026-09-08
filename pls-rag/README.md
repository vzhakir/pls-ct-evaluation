# CSIPBLLM — RAG Evaluation
Kode evaluasi RL untuk skripsi: "Evaluasi Large Language Model Berdasarkan RAG, Prompt Engineering, dan Reinforcement Learning" Vergiawan Zhaki Rasendria — G6401221101 — IPB University

---

## Struktur Folder

```
pls-rag/
│
├── api.py                    ← Backend FastAPI (RAG selalu aktif, 48 tipe kognitif)
│                                 rag.py mengimpor konfigurasi LLM langsung dari file ini
│
├── rag.py                     ← JALANKAN INI untuk evaluasi penuh
│
├── test_cases.py               ← 75 test case (eval-001 sampai eval-075)
│                                 schema: query, relevant_keywords, cognitive,
│                                 session_id, query_type, context_note
│
├── materials/                  ← Materi CT per LT dan level, plus referensi ground truth
│   ├── 1PAI.txt ... 6TGR.txt        Materi per kode LT (level 1-6)
│   ├── GT_CT01.txt ... GT_CT13.txt  Jawaban referensi per nomor soal
│   ├── GT_DETAIL_PT01 ... PT13      Materi rinci per pertemuan
│   ├── GT_SUBTOPIK_01 ... 14        Materi rinci per subtopik CT
│   └── ground_truth_KOM2102_*.txt   Ground truth tambahan per topik
│
├── static/                     ← Frontend sederhana untuk tutor interaktif
│   ├── index.html
│   ├── script.js
│   └── style.css
│
└── (tidak ada requirements.txt sendiri, lihat bagian Setup)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set API key sesuai penyedia LLM yang dipakai untuk chat maupun untuk judge (MT-Bench dan RAGAS memakai LLM sebagai verifier).

```bash
# OpenRouter, dipakai untuk model chat (OR_MODEL) dan model judge (JUDGE_MODEL_OR)
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx

# Ollama tidak butuh API key, cukup pastikan servernya sedang berjalan
ollama serve
```

---

## Cara Menjalankan

### 1. Jalankan backend RAG

`rag.py` mengimpor konfigurasi dan fungsi LLM langsung dari `api.py`, dan evaluasi retrieval memanggil endpoint `/chat` dan `/evaluate` milik backend ini.

```bash
uvicorn api:app --reload --port 8000
```

### 2. Jalankan evaluasi

Mode default menjalankan satu batch soal (soal 1 sampai 25 dari `test_cases.py`).

```bash
python rag.py
```

Rentang batch, mode auto batch, dan fitur opsional diatur lewat environment variable.

```bash
# Ganti rentang batch (default 1-25)
export RAG_BATCH_START=1
export RAG_BATCH_END=25

# Jalankan otomatis tiga batch berurutan (1-25, 26-50, 51-75) dengan jeda 10 detik
export RAG_AUTO_BATCHES=1

# Aktifkan baseline No-RAG untuk pembanding (default mati agar lebih ringan)
export RUN_NORAG_BASELINE=1

# Aktifkan alur anotasi manual peneliti
export USE_SELF_ANNOTATION=1

python rag.py
```

Parameter formula evaluasi (SelfCheckGPT dan RAGAS) juga bisa diatur lewat environment variable, dengan nilai default yang dipakai untuk skripsi ini.

```bash
export SELFCHECK_N_SAMPLES=3        # jumlah sampel stochastic SelfCheckGPT
export SELFCHECK_TEMPERATURE=1.0    # temperature saat generate sampel
export RAGAS_MAX_CLAIMS=8           # maksimum klaim yang diverifikasi per jawaban
export RAGAS_CONTEXT_MAX_CHARS=4000
export MTBENCH_CONTEXT_MAX_CHARS=4000
export MTBENCH_TEMPERATURE=0.0      # temperature saat LLM judge memberi skor
```

### 3. Alur anotasi manual (opsional)

Jika `USE_SELF_ANNOTATION=1` dan file `self_annotations.json` belum ada, `rag.py` otomatis membuat template dari hasil evaluasi. Isi field `annotation` di file tersebut secara manual (1 untuk jawaban benar atau tidak halusinasi, 0 untuk salah atau halusinasi), lalu jalankan ulang `rag.py` agar `compute_annotation_metrics` menghitung metriknya.

---

## Metrik yang Dihitung

### Retrieval
- **Precision@K dan Recall@K** — proxy berbasis threshold skor cosine similarity (`RELEVANCE_THRESHOLD = 0.30`), bukan Precision@K/Recall@K murni.
- **Mean Similarity** — rata rata skor cosine seluruh chunk yang diambil.
- **Coverage** — proporsi chunk dengan skor cosine di atas `COVERAGE_THRESHOLD = 0.50`.
- **Source Diversity** — jumlah sumber unik dibagi K (`TOP_K = 6`).

### Generation
- **Faithfulness (RAGAS)** — jawaban dipecah menjadi klaim atomik oleh LLM, tiap klaim diverifikasi ke konteks RAG, skor = klaim yang didukung dibagi total klaim. Mengikuti Es et al. (2023), arXiv:2309.15217, bagian 3.1.
- **Hallucination Risk (SelfCheckGPT)** — jawaban digenerate ulang beberapa kali secara stochastic, tiap kalimat jawaban utama dibandingkan ke seluruh sampel lewat pendekatan mirip BERTScore berbasis embedding. Mengikuti Manakul et al. (2023), arXiv:2303.08896, persamaan 4.
- **Answer Quality (MT-Bench)** — LLM judge memberi skor 1 sampai 10 berdasarkan pertanyaan, jawaban, dan konteks RAG, lalu dinormalisasi ke 0 sampai 1. Mengikuti Zheng et al. (2023), arXiv:2306.05685, bagian 3.

### Baseline dan validasi tambahan
- **No-RAG baseline** — query yang sama dikirim langsung ke LLM tanpa konteks retrieval, untuk membuktikan kontribusi RAG.
- **Self-annotation** — penilaian manual peneliti sebagai pembanding terhadap skor otomatis.

---

## Output

Semua hasil tersimpan di `eval-v7-results/` dengan nama file memakai timestamp.

| File | Isi |
|------|-----|
| `rag_eval_<timestamp>.json` | Hasil lengkap termasuk metrik agregat, hasil per soal, baseline No-RAG, dan konfigurasi yang dipakai |
| `rag_eval_<timestamp>.csv` | Satu baris per soal, kolom retrieval dan generation berdampingan dengan kolom No-RAG |
| `rag_eval_<timestamp>_report.txt` | Laporan ringkasan terformat, hasil dari `_write_report` |

Mode `RAG_AUTO_BATCHES=1` menghasilkan tiga set file terpisah (satu per batch), dan menyarankan penggabungan lewat `merge_eval_results.py` (skrip terpisah, tidak disertakan di folder ini).

---

## Referensi Fungsi

### Konfigurasi dan pemuatan data
| Fungsi | Kegunaan |
|--------|----------|
| `_load_external_test_cases(path)` | Memuat `TEST_CASES` dari `test_cases.py` tanpa bergantung pada package path |
| `_normalize_test_case(tc, idx)` | Memvalidasi dan menormalkan satu test case ke schema baru |
| `get_reference_answer_for_eval(tc, rag_context)` | Mengambil referensi atau proxy untuk evaluasi kualitas jawaban |

### Similarity dan metrik retrieval
| Fungsi | Kegunaan |
|--------|----------|
| `get_embedding(text)` | Menghasilkan vektor embedding lewat `embeddings_model` dari `api.py` |
| `cosine_similarity(vec_a, vec_b)` | Menghitung cosine similarity antara dua vektor |
| `precision_at_k(scores, k, threshold)` | Proxy Precision@K berbasis skor retrieval |
| `recall_at_k(scores, total_relevant, k, threshold)` | Proxy Recall@K berbasis skor retrieval |
| `chunk_relevance_score(chunk_scores)` | Skor relevansi chunk kontinu |
| `mean_similarity(scores)` | Rata rata similarity hasil retrieval |
| `coverage_score(scores, threshold)` | Proporsi chunk dengan skor di atas threshold |
| `source_diversity(sources)` | Jumlah sumber unik dibagi K |

### Faithfulness (RAGAS)
| Fungsi | Kegunaan |
|--------|----------|
| `_llm_call_for_ragas(prompt, max_tokens)` | Memanggil LLM untuk keperluan RAGAS |
| `_decompose_claims(answer, max_claims)` | Memecah jawaban menjadi klaim atomik lewat LLM |
| `_verify_claim(claim, context_text)` | Memverifikasi satu klaim terhadap konteks lewat LLM |
| `evaluate_faithfulness(answer, retrieved_chunks)` | Menghitung skor Faithfulness berbasis klaim |

### Hallucination (SelfCheckGPT)
| Fungsi | Kegunaan |
|--------|----------|
| `_split_sentences(text)` | Memisahkan teks menjadi kalimat sederhana |
| `_generate_stochastic_samples(query, cognitive, session_id, n)` | Menggenerate N sampel stochastic dari LLM untuk query yang sama |
| `_bert_score_sentence_pair(sent_a, sent_b)` | Menghitung kemiripan dua kalimat sebagai aproksimasi BERTScore |
| `detect_hallucination(answer, retrieved_chunks, precomputed_faithfulness, query, cognitive, session_id)` | Mendeteksi risiko halusinasi lewat SelfCheckGPT-BERTScore |

### Answer Quality (MT-Bench)
| Fungsi | Kegunaan |
|--------|----------|
| `evaluate_answer_quality(llm_reply, reference_answer, eval_resp, fast_bool_threshold, question, rag_context)` | Menilai kualitas jawaban lewat MT-Bench single-answer grading |

### Pemanggilan backend dan baseline
| Fungsi | Kegunaan |
|--------|----------|
| `call_chat_api(message, cognitive, session_id)` | Memanggil endpoint `/chat` |
| `call_evaluate_api(answer, correct_answer, active_question, wrong_count, cognitive, session_id)` | Memanggil endpoint `/evaluate` |
| `call_chat_norag(message)` | Memanggil LLM langsung tanpa konteks RAG, untuk baseline |
| `evaluate_retrieval_direct(query, relevant_keywords, k, cognitive)` | Mengevaluasi kualitas retrieval lewat dua jalur |
| `run_norag_baseline()` | Menjalankan semua query ke LLM tanpa konteks RAG |

### Anotasi manual
| Fungsi | Kegunaan |
|--------|----------|
| `generate_annotation_template(rag_results)` | Membuat template JSON untuk anotasi manual peneliti |
| `load_annotations()` | Membaca file anotasi manual peneliti |
| `compute_annotation_metrics(rag_results)` | Menghitung metrik berdasarkan anotasi manual |

### Orkestrasi dan output
| Fungsi | Kegunaan |
|--------|----------|
| `run_full_evaluation()` | Menjalankan evaluasi RAG penuh untuk semua test case aktif |
| `run_offline_analysis(history_json_path)` | Menganalisis statistik dari log percakapan historis |
| `compute_aggregate_metrics(results, norag_results, annotation_metrics)` | Menghitung metrik agregat dari seluruh hasil |
| `save_results(results, aggregate, offline_stats, norag_results)` | Menyimpan hasil ke file JSON, CSV, dan TXT |
| `_write_report(f, aggregate, offline_stats, results, norag_results)` | Menulis laporan terformat ke file TXT |
| `_run_current_test_cases(batch_label)` | Menjalankan evaluasi untuk batch aktif lalu menyimpan hasilnya |
| `main()` | Titik masuk program, memilih mode single batch atau auto batch |

---

## Troubleshooting

**Koneksi ke backend gagal** — pastikan `uvicorn api:app` sedang berjalan di `http://127.0.0.1:8000` (nilai `BASE_URL` di `rag.py`) sebelum menjalankan `python rag.py`.

**`FileNotFoundError` terkait `test_cases.py`** — file itu harus berada di folder yang sama dengan `rag.py`, atau arahkan lewat environment variable `RAG_TEST_CASES_FILE`.

**Evaluasi lambat atau boros API call** — turunkan `SELFCHECK_N_SAMPLES` dan `RAGAS_MAX_CLAIMS`, atau matikan `RUN_NORAG_BASELINE` yang menggandakan jumlah panggilan LLM.

**Ingin melanjutkan anotasi manual** — cek apakah `self_annotations.json` sudah dibuat oleh `generate_annotation_template`, isi kolom `annotation` di file itu, baru jalankan ulang dengan `USE_SELF_ANNOTATION=1`.
