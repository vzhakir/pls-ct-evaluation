# PLS-CT RL Evaluation Harness

Kode evaluasi RL untuk skripsi:
**"Evaluasi Large Language Model Berdasarkan RAG, Prompt Engineering, dan Reinforcement Learning"**
Vergiawan Zhaki Rasendria — G6401221101 — IPB University

---

## Struktur Folder

```
pls_ct_rl/
│
├── compare.py              ← JALANKAN INI untuk evaluasi penuh
│
├── core/                    ← Mesin RL (tidak perlu diubah)
│   ├── pedagogy_selector.py    RL agent + SessionRegistry
│   └── rl_metrics.py           Reward / mastery / performance / engagement
│
├── evaluation/              ← Tiga metrik evaluasi RL (v3, paper-aligned)
│   └── evaluator.py
│       ├── evaluate_kt_auc()          KT-AUC (Liu et al. 2022 pyKT)
│       ├── evaluate_reward_decomposition()  CV bobot α/β/γ (Juozapaitis 2019)
│       └── evaluate_ope_dr()          OPE Doubly Robust (Zhan et al. 2021)
│
├── simulate_rl/             ← Simulator mahasiswa berbasis aturan
│   └── profiles_new.py         StudentSimulator + profil perilaku
│
├── materials/               ← Materi CT per LT dan level (56 file)
│   ├── 1PAR.txt ... 6TGI.txt   Materi per kode LT (level 1-6)
│   └── *.md                    Materi topik CT umum
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Set API key OpenRouter (wajib, semua model default pakai OpenRouter):

```bash
# Linux / Mac
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx

# Windows
set OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
```

---

## Cara Menjalankan

Semua perintah dijalankan dari dalam folder `pls_ct_rl/`.

### 1. Smoke test — verifikasi setup (±2 menit, hemat API)

```bash
python compare.py --smoke
```

Menjalankan 1 model, 1 seed, 8 pertanyaan. Kalau berhasil, lanjut ke full run.

### 2. Full thesis run — 6 model × 8 seed × 50 pertanyaan

```bash
python compare.py --full
```

### 3. Custom subset

```bash
python compare.py \
  --models deepseek/deepseek-v3.2 qwen/qwen3-32b \
  --providers openrouter openrouter \
  --seeds 42 43 44 \
  --questions 30
```

### 4. Dengan fixed judge (mengurangi self-evaluation bias)

```bash
python compare.py --full \
  --fixed-judge-model x-ai/grok-4-fast \
  --fixed-judge-provider openrouter \
  --label-source truth
```

---

## Output

Setelah selesai, buka `results/`:

| File | Isi |
|------|-----|
| `summary.csv` | Tabel utama untuk skripsi, 1 baris per run |
| `stats.json` | Friedman test + pairwise Wilcoxon per metrik |
| `runs/*.json` | Detail lengkap per run (step log + summary) |

### Kolom penting di summary.csv

| Kolom | Keterangan | Target |
|-------|-----------|--------|
| `kt_auc` | KT-AUC (composite predictor v3) | ≥ 0.72 |
| `kt_auc_stable` | KT-AUC setelah filter minority rate | ≥ 0.72 |
| `cv_max` | max CV bobot α/β/γ dari MLR refit | < 0.30 |
| `cv_rewards` | CV reward per-step (diagnostic saja) | — |
| `ope_dr_value` | V̂_DR: estimasi nilai pure-exploit policy | > `ope_baseline` |
| `ope_baseline` | V̂_logged: nilai logging policy | — |
| `judge_agreement` | Seberapa sering judge setuju ground truth | ≥ 0.70 |
| `policy_converged` | Apakah agent konvergen ke satu LT | True |
| `policy_convergence_q` | Pertanyaan ke berapa agent konvergen | — |

---

## Metrik Evaluasi RL — Referensi Paper

### KT-AUC
- **Protokol**: Liu et al. (2022). pyKT: A Python Library to Benchmark DLKT Models. *NeurIPS 2022*.
- **Predictor**: composite `0.3×mastery_score + 0.5×performance + 0.2×q_pred` (pilihan desain sistem)
- **Threshold 0.72**: Corbett & Anderson (1994). Knowledge Tracing. *UMUAI 4(4)*.
- **Threshold 0.80**: Piech et al. (2015). Deep Knowledge Tracing. *NeurIPS 2015*.

### Reward Decomposition CV
- **Formula RD**: Juozapaitis et al. (2019). Explainable RL via Reward Decomposition. *IJCAI/ECAI XAI Workshop*.
- **Linear scalarization**: Roijers et al. (2013). A Survey of Multi-Objective Sequential Decision-Making. *JAIR 48*.
- **CV threshold 30%**: Reed et al. (2002). *Clin Diagn Lab Immunol 9(6):1235–1239*.

### OPE Doubly Robust
- **Formula**: Zhan et al. (2021). Off-Policy Evaluation via Adaptive Weighting. *KDD 2021*. Eq.2.
- **Adaptasi RL**: Jiang & Li (2016). Doubly Robust Off-policy Value Evaluation for RL. *ICML 2016*.

---

## Troubleshooting

**`ModuleNotFoundError`** — pastikan menjalankan dari dalam folder `pls_ct_rl/`, bukan dari luar.

**Model 404 dari OpenRouter** — slug model bisa berbeda. Cek di [openrouter.ai/models](https://openrouter.ai/models) dan sesuaikan di `DEFAULT_MODELS` (baris ~91) di `compare7.py`.

**Timeout** — tambahkan `--questions 20` untuk memperpendek saat testing.
