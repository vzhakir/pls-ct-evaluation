# PLS-CT Prompt Engineering Evaluation Harness

Kode evaluasi Prompt Engineering untuk skripsi
**"Evaluasi Large Language Model Berdasarkan RAG, Prompt Engineering, dan Reinforcement Learning"**
Vergiawan Zhaki Rasendria — G6401221101 — IPB University

---

## Struktur Folder

```
pls-pe/
│
├── api.py                   ← Backend FastAPI (RAG selalu aktif, 48 tipe kognitif)
│
├── test_prompt.py            ← JALANKAN INI untuk menghasilkan jawaban model
│   └── Menguji 3 strategi prompt dengan konteks retrieval yang sama
│       zero-shot, few-shot, dan Chain-of-Thought (CoT)
│
├── prompt_eval.py            ← Evaluator ROUGE dan BERTScore atas hasil di atas
│
├── soal_ct.json              ← 30 soal CT (diambil dari KOM2102_Eval_Template)
│
├── materials/                 ← Materi CT per LT dan level, plus referensi ground truth
│   ├── 1PAR.txt ... 6TGI.txt      Materi per kode LT (level 1-6)
│   └── GT_CT01.txt ... GT_CT13.txt  Jawaban referensi per nomor soal
│
├── static/                    ← Frontend sederhana untuk tutor interaktif
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Program ini mendukung tiga penyedia LLM secara hybrid. Isi salah satu sesuai yang dipakai.

```bash
# OpenRouter
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx

# ChatAnywhere
export OPENAI_API_KEY=sk-xxxxxxxx

# Ollama tidak butuh API key, cukup pastikan servernya sedang berjalan
ollama serve
```

---

## Alur Kerja

### 1. Jalankan backend RAG

Semua strategi prompt mengambil konteks lewat endpoint `/retrieve`, jadi backend harus aktif lebih dulu.

```bash
uvicorn api:app --reload --port 8000
```

Endpoint `/chat` dan `/evaluate` dipakai untuk tutor adaptif di halaman web (`static/index.html`) dan tidak dilewati oleh evaluasi prompt engineering, agar hasilnya tidak tercampur dengan instruksi tutor atau cognitive profile.

### 2. Hasilkan jawaban model untuk tiap strategi prompt

```bash
python test_prompt.py \
  --model-name llama3.1:8b \
  --provider ollama \
  --strategies all \
  --retrieval-base-url http://127.0.0.1:8000 \
  --save-prompts --save-json
```

Argumen penting lain yang tersedia meliputi `--start` dan `--end` untuk rentang nomor soal, `--cognitive` untuk kode tipe kognitif yang dipakai saat retrieval, `--top-k` untuk jumlah chunk RAG, serta `--provider openrouter` atau `--provider chatanywhere` untuk pindah penyedia LLM.

Hasil tersimpan otomatis di `hasil_prompt_engineering_retrieval/rag_on/<strategi>/<model>/`.

### 3. Evaluasi hasil dengan ROUGE dan BERTScore

Evaluasi semua model dan semua teknik sekaligus.

```bash
python prompt_eval.py --all \
  --reference-folder materials \
  --input-root hasil_prompt_engineering_retrieval \
  --output-root hasil_evaluasi_prompt_engineering_retrieval
```

Atau evaluasi satu kombinasi model dan teknik saja.

```bash
python prompt_eval.py \
  --model-name deepseek_deepseek-v3.2 \
  --prompt-technique cot \
  --reference-folder materials
```

---

## Output

| File | Isi |
|------|-----|
| `summary_all_<timestamp>.json` | Hasil lengkap semua model dan teknik dalam format JSON |
| `summary_all_<timestamp>.txt` | Ringkasan yang sama dalam format teks, mudah dibaca langsung |

Kedua file berada di `hasil_evaluasi_prompt_engineering_retrieval/rag_on/` dan berisi ROUGE-1, ROUGE-2, ROUGE-L, serta BERTScore F1 rata rata per model per teknik prompt.

---

## Strategi Prompt yang Diuji

- **Zero-shot** — model langsung menjawab dari soal dan konteks retrieval, tanpa contoh.
- **Few-shot** — beberapa contoh soal dan jawaban disertakan sebelum soal yang dievaluasi.
- **Chain-of-Thought (CoT)** — model diminta menjabarkan langkah penalaran sebelum menjawab.

Ketiganya memakai konteks RAG yang sama dari `/retrieve` agar perbandingan antar strategi adil, dan tidak melewati endpoint `/chat` supaya tidak tercampur dengan prompt tutor adaptif atau follow-up generator.

---

## Troubleshooting

**Folder `rag_on` tidak ditemukan saat evaluasi** — jalankan `test_prompt.py` lebih dulu dengan `--output-root` yang sama dengan yang dipakai di `prompt_eval.py`.

**`ModuleNotFoundError: rouge_score` atau `bert_score`** — jalankan ulang `pip install -r requirements.txt`.

**Retrieval gagal atau timeout** — pastikan `uvicorn api:app` sedang berjalan di port yang sama dengan `--retrieval-base-url`, dan folder `materials/` sudah berisi materi yang lengkap.

**Model 404 dari OpenRouter** — cek nama model persis di [openrouter.ai/models](https://openrouter.ai/models) dan sesuaikan nilai `--model-name`.
