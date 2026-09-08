#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
prompt_engineering_retrieval_eval.py

Evaluasi Prompt Engineering DENGAN konteks retrieval.

Tujuan:
- Menguji 3 strategi prompt engineering:
  1) zero-shot
  2) few-shot
  3) Chain-of-Thought (CoT)
- Semua strategi memakai konteks retrieval yang sama dari endpoint /retrieve.
- Program ini TIDAK memanggil endpoint /chat agar tidak tercampur dengan prompt tutor adaptif,
  cognitive profile, instruksi "jangan langsung jawab final", atau follow-up generator.

Alur:
1. Baca soal dari soal_ct.json.
2. Untuk setiap soal, ambil konteks dari backend RAG melalui POST /retrieve.
3. Bentuk prompt sesuai strategi: zero-shot, few-shot, atau cot.
4. Kirim prompt langsung ke endpoint LLM OpenAI-compatible:
   - Ollama:      http://localhost:11434/v1
   - OpenRouter:  https://openrouter.ai/api/v1
   - ChatAnywhere atau endpoint lain juga bisa dipakai.
5. Simpan jawaban, prompt, dan metadata retrieval.

Contoh:
python prompt_engineering_retrieval_eval.py ^
  --questions-json soal_ct.json ^
  --model-name "llama3.1:8b" ^
  --strategies all ^
  --retrieval-base-url http://127.0.0.1:8000 ^
  --llm-base-url http://localhost:11434/v1 ^
  --api-key ollama ^
  --output-root hasil_prompt_engineering_retrieval ^
  --save-prompts --save-json

Catatan:
- Pastikan backend ollamaapi.py sudah berjalan agar endpoint /retrieve tersedia.
- Pastikan model LLM dapat diakses melalui endpoint OpenAI-compatible.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ============================================================
# UTILITAS
# ============================================================

def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", str(name).strip(), flags=re.UNICODE)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "model"


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_strategies(value: str) -> List[str]:
    if value.lower().strip() == "all":
        return ["zero-shot", "few-shot", "cot"]

    alias = {
        "zero": "zero-shot",
        "zeroshot": "zero-shot",
        "zero-shot": "zero-shot",
        "no-shot": "zero-shot",
        "noshot": "zero-shot",
        "few": "few-shot",
        "fewshot": "few-shot",
        "few-shot": "few-shot",
        "cot": "cot",
        "chain-of-thought": "cot",
        "chain_of_thought": "cot",
    }

    out = []
    for part in value.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in alias:
            raise ValueError(
                f"Strategi tidak valid: {part}. Gunakan: all, zero-shot, few-shot, cot"
            )
        norm = alias[key]
        if norm not in out:
            out.append(norm)

    if not out:
        raise ValueError("Tidak ada strategi yang dipilih.")
    return out


# ============================================================
# LOAD SOAL
# ============================================================

def load_questions(path: str, start: int, end: int) -> List[Dict[str, Any]]:
    raw = read_json(Path(path))

    if isinstance(raw, dict):
        data = raw.get("questions") or raw.get("data")
        if data is None and all(isinstance(v, dict) for v in raw.values()):
            data = list(raw.values())
    else:
        data = raw

    if not isinstance(data, list):
        raise ValueError("Format soal harus list, atau dict dengan key 'questions'/'data'.")

    questions: List[Dict[str, Any]] = []

    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            nomor = idx
            soal = item
            meta = {}
        elif isinstance(item, dict):
            nomor = (
                item.get("nomor")
                or item.get("no")
                or item.get("id")
                or item.get("number")
                or idx
            )
            soal = (
                item.get("soal")
                or item.get("question")
                or item.get("pertanyaan")
                or item.get("text")
                or item.get("query")
            )
            meta = {k: v for k, v in item.items() if k not in {"soal", "question", "pertanyaan", "text", "query"}}
        else:
            continue

        try:
            nomor = int(nomor)
        except Exception:
            nomor = idx

        if soal and start <= nomor <= end:
            questions.append({
                "nomor": nomor,
                "soal": str(soal).strip(),
                "meta": meta,
            })

    questions.sort(key=lambda x: x["nomor"])

    if not questions:
        raise ValueError(f"Tidak ada soal pada rentang {start}-{end}.")

    return questions


# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================

DEFAULT_FEWSHOT_EXAMPLES = [
    {
        "question": "Jelaskan elemen dekomposisi dalam masalah menyusun jadwal belajar mingguan.",
        "answer": (
            "Dekomposisi dilakukan dengan memecah masalah besar menjadi beberapa bagian kecil, "
            "misalnya daftar mata kuliah, tenggat tugas, durasi belajar, tingkat kesulitan, dan waktu kosong. "
            "Setelah itu, setiap bagian diatur menjadi jadwal belajar yang lebih mudah dikelola."
        ),
    },
    {
        "question": "Tuliskan algoritme sederhana untuk menentukan apakah sebuah bilangan genap atau ganjil.",
        "answer": (
            "INPUT: sebuah bilangan n. PROSES: hitung sisa pembagian n dengan 2. "
            "Jika n mod 2 = 0, maka bilangan tersebut genap. Jika tidak, bilangan tersebut ganjil. "
            "OUTPUT: keterangan genap atau ganjil."
        ),
    },
    {
        "question": "Mengapa stack cocok digunakan untuk fitur undo?",
        "answer": (
            "Stack cocok untuk fitur undo karena menggunakan prinsip Last In First Out (LIFO). "
            "Aksi terakhir yang dilakukan pengguna akan menjadi aksi pertama yang dibatalkan, "
            "sehingga urutan pembatalan sesuai dengan urutan penggunaan fitur undo."
        ),
    },
]


def load_fewshot_examples(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return DEFAULT_FEWSHOT_EXAMPLES

    raw = read_json(Path(path))

    if isinstance(raw, dict):
        examples = raw.get("examples") or raw.get("data") or raw.get("fewshot_examples")
    else:
        examples = raw

    if not isinstance(examples, list):
        raise ValueError("File few-shot harus berupa list atau dict dengan key 'examples'/'data'.")

    cleaned: List[Dict[str, str]] = []
    for item in examples:
        if not isinstance(item, dict):
            continue
        q = item.get("question") or item.get("soal") or item.get("pertanyaan")
        a = item.get("answer") or item.get("jawaban") or item.get("response")
        if q and a:
            cleaned.append({"question": str(q).strip(), "answer": str(a).strip()})

    if not cleaned:
        raise ValueError("Tidak ada contoh few-shot valid di file.")
    return cleaned


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_context(
    retrieval_base_url: str,
    question: str,
    cognitive: str,
    k: int,
    timeout: int,
) -> Dict[str, Any]:
    url = retrieval_base_url.rstrip("/") + "/retrieve"
    payload = {
        "query": question,
        "cognitive": cognitive,
        "k": k,
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    chunks = data.get("chunks") or []
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        source = str(chunk.get("source", f"sumber_{i}")).strip()
        score = chunk.get("score", "")
        if text:
            context_parts.append(
                f"[Konteks {i} | sumber: {source} | skor: {score}]\n{text}"
            )

    context = "\n\n".join(context_parts).strip()
    if not context:
        context = "Konteks retrieval belum tersedia."

    return {
        "context": context,
        "raw": data,
    }


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_zero_shot_prompt(question: str, context: str) -> str:
    return f"""Anda adalah sistem penjawab soal Computational Thinking berbahasa Indonesia.

Gunakan konteks retrieval berikut sebagai sumber utama.
Jika konteks tidak memuat informasi yang cukup, jawab berdasarkan penalaran yang relevan, tetapi jangan membuat klaim yang tidak didukung.

KONTEKS RETRIEVAL:
{context}

SOAL:
{question}

TUGAS:
Jawab soal secara langsung, jelas, akademik, dan sesuai konteks.
Jangan menambahkan pertanyaan lanjutan.
Jangan menyebut bahwa Anda adalah AI.

JAWABAN:"""


def build_few_shot_prompt(question: str, context: str, examples: List[Dict[str, str]]) -> str:
    ex_blocks = []
    for i, ex in enumerate(examples, start=1):
        ex_blocks.append(
            f"""Contoh {i}
Soal:
{ex['question']}

Jawaban:
{ex['answer']}"""
        )

    examples_text = "\n\n".join(ex_blocks)

    return f"""Anda adalah sistem penjawab soal Computational Thinking berbahasa Indonesia.

Gunakan konteks retrieval berikut sebagai sumber utama.
Perhatikan pola jawaban pada contoh-contoh berikut, lalu jawab soal target dengan gaya yang serupa.

KONTEKS RETRIEVAL:
{context}

CONTOH FEW-SHOT:
{examples_text}

SOAL TARGET:
{question}

TUGAS:
Jawab soal target secara langsung, jelas, akademik, dan sesuai konteks.
Jangan menambahkan pertanyaan lanjutan.
Jangan menyebut bahwa Anda adalah AI.

JAWABAN:"""


def build_cot_prompt(question: str, context: str) -> str:
    return f"""Anda adalah sistem penjawab soal Computational Thinking berbahasa Indonesia.

Gunakan konteks retrieval berikut sebagai sumber utama.
Soal Computational Thinking sering membutuhkan penalaran bertahap, seperti dekomposisi, abstraksi, pengenalan pola, atau perancangan algoritme.

KONTEKS RETRIEVAL:
{context}

SOAL:
{question}

TUGAS:
Uraikan penalaran secara bertahap sebelum memberikan jawaban akhir.
Gunakan struktur berikut:
1. Analisis singkat soal
2. Langkah penyelesaian
3. Jawaban akhir

Pastikan jawaban akhir eksplisit dan tidak menambahkan pertanyaan lanjutan.
Jangan menyebut bahwa Anda adalah AI.

JAWABAN:"""


def build_prompt(strategy: str, question: str, context: str, examples: List[Dict[str, str]]) -> str:
    if strategy == "zero-shot":
        return build_zero_shot_prompt(question, context)
    if strategy == "few-shot":
        return build_few_shot_prompt(question, context, examples)
    if strategy == "cot":
        return build_cot_prompt(question, context)
    raise ValueError(f"Strategi tidak dikenal: {strategy}")


# ============================================================
# LLM CALL
# ============================================================

def call_llm(
    llm_base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: Optional[int],
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Memanggil endpoint OpenAI-compatible /chat/completions tanpa dependency openai.
    """

    url = llm_base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Anda menjawab soal Computational Thinking dalam Bahasa Indonesia. "
                    "Jawaban harus jelas, akademik, dan berfokus pada soal."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = max_tokens

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2).strip()


def build_extra_headers(provider: str) -> Dict[str, str]:
    if provider.lower() == "openrouter":
        return {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Prompt Engineering Retrieval Evaluation",
        }
    return {}


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluasi zero-shot, few-shot, dan CoT dengan konteks retrieval."
    )

    # Data
    parser.add_argument("--questions-json", default="soal_ct.json")
    parser.add_argument("--fewshot-json", default="")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=30)
    parser.add_argument("--strategies", default="all", help="all atau kombinasi: zero-shot,few-shot,cot")

    # Retrieval
    parser.add_argument("--retrieval-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cognitive", default="1PAR")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--retrieval-timeout", type=int, default=120)

    # LLM
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openrouter", "chatanywhere", "custom"])
    parser.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--llm-timeout", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=0)

    # Output
    parser.add_argument("--output-root", default="hasil_prompt_engineering_retrieval")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument(
        "--init-folders-only",
        action="store_true",
        help="Hanya buat struktur folder output lalu berhenti, tanpa memanggil retrieval/LLM.",
    )

    args = parser.parse_args()

    strategies = parse_strategies(args.strategies)
    questions = load_questions(args.questions_json, args.start, args.end)
    fewshot_examples = load_fewshot_examples(args.fewshot_json or None)

    # API key defaults
    api_key = args.api_key
    if not api_key:
        if args.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
        elif args.provider == "chatanywhere":
            api_key = os.getenv("OPENAI_API_KEY", "")
        elif args.provider == "ollama":
            api_key = "ollama"

    if not api_key:
        raise ValueError(
            "API key kosong. Isi --api-key atau set environment variable yang sesuai."
        )

    model_safe = safe_name(args.model_name)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "timestamp": datetime.now().isoformat(),
        "questions_json": args.questions_json,
        "fewshot_json": args.fewshot_json or None,
        "strategies": strategies,
        "retrieval_base_url": args.retrieval_base_url,
        "cognitive": args.cognitive,
        "top_k": args.top_k,
        "provider": args.provider,
        "llm_base_url": args.llm_base_url,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens if args.max_tokens > 0 else None,
        "total_questions": len(questions),
        "note": "Evaluasi prompt engineering dengan konteks retrieval. Tidak menggunakan endpoint /chat.",
    }
    write_json(output_root / f"run_config_{model_safe}.json", run_config)

    # Buat semua folder output sejak awal, sebelum ada request retrieval/LLM.
    # Ini membuat folder tetap terbentuk meskipun koneksi retrieval atau API gagal.
    prepared_dirs = {}
    for strategy_name in strategies:
        prepared_dir = output_root / "rag_on" / strategy_name / model_safe
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_dirs[strategy_name] = str(prepared_dir.resolve())

    readme_lines = [
        "Folder hasil evaluasi Prompt Engineering dengan konteks retrieval.",
        "",
        "Struktur:",
        "rag_on/<strategy>/<model>/",
        "",
        "Strategi yang dibuat:",
    ]
    for strategy_name, folder_path in prepared_dirs.items():
        readme_lines.append(f"- {strategy_name}: {folder_path}")
    readme_lines.extend([
        "",
        "Catatan:",
        "- Endpoint /chat tidak digunakan.",
        "- Konteks diambil dari /retrieve.",
        "- Prompt dibangun langsung oleh script untuk zero-shot, few-shot, dan CoT.",
    ])
    (output_root / "README_OUTPUT.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print("[INFO] Evaluasi Prompt Engineering dengan konteks retrieval", flush=True)
    print(f"[INFO] Jumlah soal       : {len(questions)}")
    print(f"[INFO] Strategi         : {', '.join(strategies)}")
    print(f"[INFO] Retrieval endpoint: {args.retrieval_base_url.rstrip('/')}/retrieve")
    print(f"[INFO] LLM endpoint      : {args.llm_base_url.rstrip('/')}/chat/completions")
    print(f"[INFO] Model             : {args.model_name}")
    print(f"[INFO] Output root       : {output_root}")
    print("[INFO] Endpoint /chat TIDAK digunakan.")
    print("[INFO] Folder output dibuat otomatis:")
    for strategy_name, folder_path in prepared_dirs.items():
        print(f"       - {strategy_name}: {folder_path}")
    print()

    if args.init_folders_only:
        print("[DONE] Struktur folder output sudah dibuat. Mode --init-folders-only aktif, evaluasi tidak dijalankan.")
        return 0

    extra_headers = build_extra_headers(args.provider)
    max_tokens = args.max_tokens if args.max_tokens > 0 else None

    all_results: List[Dict[str, Any]] = []

    for strategy in strategies:
        strategy_dir = output_root / "rag_on" / strategy / model_safe
        strategy_dir.mkdir(parents=True, exist_ok=True)

        print(f"[STRATEGY] {strategy}")

        for item in questions:
            nomor = item["nomor"]
            soal = item["soal"]

            answer_file = strategy_dir / f"soal{nomor}_{model_safe}_{strategy}.txt"
            prompt_file = strategy_dir / f"soal{nomor}_{model_safe}_{strategy}_PROMPT.txt"
            meta_file = strategy_dir / f"soal{nomor}_{model_safe}_{strategy}.json"

            if args.skip_existing and answer_file.exists():
                print(f"[SKIP] Soal {nomor} sudah ada -> {answer_file.name}")
                continue

            try:
                print(f"[RUN] {strategy} | Soal {nomor}")

                retrieval = retrieve_context(
                    retrieval_base_url=args.retrieval_base_url,
                    question=soal,
                    cognitive=args.cognitive,
                    k=args.top_k,
                    timeout=args.retrieval_timeout,
                )

                prompt = build_prompt(
                    strategy=strategy,
                    question=soal,
                    context=retrieval["context"],
                    examples=fewshot_examples,
                )

                answer = call_llm(
                    llm_base_url=args.llm_base_url,
                    api_key=api_key,
                    model_name=args.model_name,
                    prompt=prompt,
                    timeout=args.llm_timeout,
                    temperature=args.temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                )

                answer_file.write_text(answer.strip() + "\n", encoding="utf-8")

                if args.save_prompts:
                    prompt_file.write_text(prompt, encoding="utf-8")

                record = {
                    "timestamp": datetime.now().isoformat(),
                    "model": args.model_name,
                    "model_safe": model_safe,
                    "strategy": strategy,
                    "rag": True,
                    "top_k": args.top_k,
                    "cognitive": args.cognitive,
                    "nomor": nomor,
                    "soal": soal,
                    "question_meta": item.get("meta", {}),
                    "answer_file": str(answer_file),
                    "prompt_file": str(prompt_file) if args.save_prompts else None,
                    "answer": answer if args.save_json else None,
                    "retrieval": retrieval["raw"],
                }

                if args.save_json:
                    write_json(meta_file, record)

                all_results.append(record)
                print(f"[OK]  {answer_file.name}")

                if args.sleep > 0:
                    time.sleep(args.sleep)

            except KeyboardInterrupt:
                print("\n[STOP] Dihentikan pengguna.")
                return 130
            except Exception as exc:
                print(f"[ERROR] {strategy} | Soal {nomor}: {exc}")
                error_file = strategy_dir / f"soal{nomor}_{model_safe}_{strategy}_ERROR.txt"
                error_file.write_text(str(exc), encoding="utf-8")

        print()

    summary_file = output_root / f"summary_{model_safe}.json"
    summary = {
        "run_config": run_config,
        "result_count": len(all_results),
        "results": all_results if args.save_json else [
            {k: v for k, v in r.items() if k not in {"answer", "retrieval"}}
            for r in all_results
        ],
    }
    write_json(summary_file, summary)

    print("[DONE] Semua proses selesai.")
    print(f"[DONE] Summary: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
