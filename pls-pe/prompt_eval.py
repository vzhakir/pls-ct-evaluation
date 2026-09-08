import os
import re
import json
import argparse
import numpy as np
from datetime import datetime
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func


class NLPValidator:
    def __init__(self, lang="id"):
        self.rouge_eval = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )
        self.lang = lang

    def extract_soal_number(self, filename):
        """
        Mengambil nomor soal dari nama file.

        Contoh:
        - soal1.txt -> 1
        - soal1_deepseek_deepseek-v3.2_cot.txt -> 1
        - soal25_qwen_qwen3-32b_zero-shot.txt -> 25
        """
        match = re.search(r"soal(\d+)", filename.lower())
        if match:
            return int(match.group(1))
        return 999999

    def read_reference_files(self, folder_path):
        """
        Membaca file ground truth dari folder referensi.
        Biasanya formatnya:
        - soal1.txt
        - soal2.txt
        - soal3.txt
        """
        texts = []
        filenames = []

        if not os.path.exists(folder_path):
            return texts, filenames

        files = [
            file for file in os.listdir(folder_path)
            if file.endswith(".txt")
        ]

        files = sorted(files, key=self.extract_soal_number)

        for file in files:
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                texts.append(content)
                filenames.append(file)

        return texts, filenames

    def read_prediction_files(self, folder_path, model_name=None, prompt_technique=None):
        """
        Membaca file prediksi model.

        File yang dibaca:
        - soal1_deepseek_deepseek-v3.2_cot.txt
        - soal2_deepseek_deepseek-v3.2_cot.txt

        File yang diabaikan:
        - *_PROMPT.txt
        - *.json
        - file .txt lain yang bukan jawaban final
        """
        texts = []
        filenames = []

        if not os.path.exists(folder_path):
            return texts, filenames

        valid_files = []

        for file in os.listdir(folder_path):
            lower_file = file.lower()

            # hanya file .txt
            if not lower_file.endswith(".txt"):
                continue

            # abaikan file prompt
            if "_prompt.txt" in lower_file:
                continue

            # harus diawali soal
            if not lower_file.startswith("soal"):
                continue

            # kalau model_name diberikan, pastikan nama model ada di filename
            if model_name is not None:
                if model_name.lower() not in lower_file:
                    continue

            # kalau prompt_technique diberikan, pastikan teknik ada di filename
            if prompt_technique is not None:
                if prompt_technique.lower() not in lower_file:
                    continue

            valid_files.append(file)

        valid_files = sorted(valid_files, key=self.extract_soal_number)

        for file in valid_files:
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                texts.append(content)
                filenames.append(file)

        return texts, filenames

    def evaluate_multi_refs(self, predictions: list, references: list):
        """
        Evaluasi setiap jawaban model terhadap semua reference.
        ROUGE memakai skor maksimum dari semua ground truth.
        BERTScore juga dibandingkan terhadap banyak reference.
        """
        results = []

        for pred in predictions:
            rouge_results = {
                "rouge1": [],
                "rouge2": [],
                "rougeL": []
            }

            for ref in references:
                s = self.rouge_eval.score(ref, pred)

                rouge_results["rouge1"].append(s["rouge1"].fmeasure)
                rouge_results["rouge2"].append(s["rouge2"].fmeasure)
                rouge_results["rougeL"].append(s["rougeL"].fmeasure)

            _, _, f1 = bert_score_func(
                [pred],
                [references],
                lang=self.lang,
                verbose=False
            )

            results.append({
                "rouge1": max(rouge_results["rouge1"]),
                "rouge2": max(rouge_results["rouge2"]),
                "rougeL": max(rouge_results["rougeL"]),
                "bertscore_f1": f1.item()
            })

        return results


def evaluate_one_folder(
    model_name,
    prompt_technique,
    reference_folder,
    input_root,
    output_root,
    lang="id"
):
    validator = NLPValidator(lang=lang)

    rag_mode = "rag_on"

    prediction_folder = os.path.join(
        input_root,
        rag_mode,
        prompt_technique,
        model_name
    )

    output_dir = os.path.join(
        output_root,
        rag_mode,
        prompt_technique,
        model_name
    )

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"evaluasi_{timestamp}.json")
    txt_path = os.path.join(output_dir, f"evaluasi_{timestamp}.txt")

    refs, ref_names = validator.read_reference_files(reference_folder)

    preds, pred_names = validator.read_prediction_files(
        prediction_folder,
        model_name=model_name,
        prompt_technique=prompt_technique
    )

    print("\n" + "=" * 80)
    print(f"Evaluasi Model        : {model_name}")
    print(f"Teknik Prompt         : {prompt_technique}")
    print(f"Mode RAG              : {rag_mode}")
    print(f"Folder Referensi      : {reference_folder}")
    print(f"Folder Prediksi       : {prediction_folder}")
    print(f"Folder Output         : {output_dir}")
    print(f"Jumlah file referensi : {len(refs)}")
    print(f"Jumlah file prediksi  : {len(preds)}")
    print("=" * 80)

    if not refs:
        print(f"[GAGAL] Folder referensi kosong atau tidak ditemukan: {reference_folder}")
        return None

    if not preds:
        print(f"[GAGAL] Folder prediksi kosong atau tidak ditemukan: {prediction_folder}")
        print("Pastikan file jawaban final berbentuk seperti:")
        print(f"soal1_{model_name}_{prompt_technique}.txt")
        print("dan bukan file *_PROMPT.txt atau *.json")
        return None

    print("\nFile prediksi yang akan dievaluasi:")
    for name in pred_names:
        print(f"- {name}")

    print("\nMulai evaluasi...")
    print("-" * 80)

    skor_per_file = validator.evaluate_multi_refs(preds, refs)

    hasil_dokumentasi = []

    for name, skor in zip(pred_names, skor_per_file):
        data = {
            "file": name,
            "rouge1": skor["rouge1"],
            "rouge2": skor["rouge2"],
            "rougeL": skor["rougeL"],
            "bertscore_f1": skor["bertscore_f1"]
        }

        hasil_dokumentasi.append(data)

        print(
            f"File: {name} | "
            f"R1: {skor['rouge1']:.4f} | "
            f"R2: {skor['rouge2']:.4f} | "
            f"RL: {skor['rougeL']:.4f} | "
            f"BERT: {skor['bertscore_f1']:.4f}"
        )

    rata_rata = {
        "rouge1": float(np.mean([x["rouge1"] for x in hasil_dokumentasi])),
        "rouge2": float(np.mean([x["rouge2"] for x in hasil_dokumentasi])),
        "rougeL": float(np.mean([x["rougeL"] for x in hasil_dokumentasi])),
        "bertscore_f1": float(np.mean([x["bertscore_f1"] for x in hasil_dokumentasi]))
    }

    output_json = {
        "model": model_name,
        "rag_mode": rag_mode,
        "prompt_technique": prompt_technique,
        "tanggal_evaluasi": timestamp,
        "folder_referensi": reference_folder,
        "folder_prediksi": prediction_folder,
        "jumlah_file_referensi": len(refs),
        "jumlah_file_prediksi": len(preds),
        "file_prediksi_yang_dievaluasi": pred_names,
        "rata_rata": rata_rata,
        "hasil_per_file": hasil_dokumentasi
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Hasil Evaluasi Model: {model_name}\n")
        f.write(f"Mode RAG: {rag_mode}\n")
        f.write(f"Teknik Prompt: {prompt_technique}\n")
        f.write(f"Tanggal Evaluasi: {timestamp}\n")
        f.write(f"Folder Referensi: {reference_folder}\n")
        f.write(f"Folder Prediksi: {prediction_folder}\n")
        f.write(f"Jumlah File Referensi: {len(refs)}\n")
        f.write(f"Jumlah File Prediksi: {len(preds)}\n")
        f.write("=" * 80 + "\n\n")

        f.write("File Prediksi yang Dievaluasi:\n")
        f.write("-" * 80 + "\n")

        for name in pred_names:
            f.write(f"{name}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Hasil Per File:\n")
        f.write("-" * 80 + "\n")

        for item in hasil_dokumentasi:
            f.write(
                f"File: {item['file']} | "
                f"R1: {item['rouge1']:.4f} | "
                f"R2: {item['rouge2']:.4f} | "
                f"RL: {item['rougeL']:.4f} | "
                f"BERT: {item['bertscore_f1']:.4f}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("Rata-rata:\n")
        f.write(f"ROUGE-1: {rata_rata['rouge1']:.4f}\n")
        f.write(f"ROUGE-2: {rata_rata['rouge2']:.4f}\n")
        f.write(f"ROUGE-L: {rata_rata['rougeL']:.4f}\n")
        f.write(f"BERTScore F1: {rata_rata['bertscore_f1']:.4f}\n")

    print("=" * 80)
    print("Rata-rata:")
    print(f"ROUGE-1     : {rata_rata['rouge1']:.4f}")
    print(f"ROUGE-2     : {rata_rata['rouge2']:.4f}")
    print(f"ROUGE-L     : {rata_rata['rougeL']:.4f}")
    print(f"BERTScore F1: {rata_rata['bertscore_f1']:.4f}")

    print("\nDokumentasi berhasil disimpan:")
    print(f"JSON: {json_path}")
    print(f"TXT : {txt_path}")

    return output_json


def evaluate_all(
    reference_folder,
    input_root,
    output_root,
    lang="id"
):
    rag_mode = "rag_on"
    rag_mode_path = os.path.join(input_root, rag_mode)

    if not os.path.exists(rag_mode_path):
        print(f"[GAGAL] Folder rag_on tidak ditemukan: {rag_mode_path}")
        return

    all_results = []

    for prompt_technique in sorted(os.listdir(rag_mode_path)):
        technique_path = os.path.join(rag_mode_path, prompt_technique)

        if not os.path.isdir(technique_path):
            continue

        for model_name in sorted(os.listdir(technique_path)):
            model_path = os.path.join(technique_path, model_name)

            if not os.path.isdir(model_path):
                continue

            result = evaluate_one_folder(
                model_name=model_name,
                prompt_technique=prompt_technique,
                reference_folder=reference_folder,
                input_root=input_root,
                output_root=output_root,
                lang=lang
            )

            if result is not None:
                all_results.append(result)

    if not all_results:
        print("[INFO] Tidak ada hasil evaluasi yang berhasil diproses.")
        return

    summary_dir = os.path.join(output_root, rag_mode)
    os.makedirs(summary_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json_path = os.path.join(summary_dir, f"summary_all_{timestamp}.json")
    summary_txt_path = os.path.join(summary_dir, f"summary_all_{timestamp}.txt")

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("SUMMARY SEMUA HASIL EVALUASI PROMPT ENGINEERING RAG_ON\n")
        f.write("=" * 80 + "\n\n")

        for result in all_results:
            rata = result["rata_rata"]

            f.write(f"Model: {result['model']}\n")
            f.write(f"Teknik: {result['prompt_technique']}\n")
            f.write(f"Jumlah File Prediksi: {result['jumlah_file_prediksi']}\n")
            f.write(f"ROUGE-1: {rata['rouge1']:.4f}\n")
            f.write(f"ROUGE-2: {rata['rouge2']:.4f}\n")
            f.write(f"ROUGE-L: {rata['rougeL']:.4f}\n")
            f.write(f"BERTScore F1: {rata['bertscore_f1']:.4f}\n")
            f.write("-" * 80 + "\n")

    print("\n" + "=" * 80)
    print("Semua evaluasi selesai.")
    print(f"Summary JSON disimpan di: {summary_json_path}")
    print(f"Summary TXT  disimpan di: {summary_txt_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluator ROUGE dan BERTScore untuk hasil prompt engineering retrieval rag_on."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Nama folder model, contoh: deepseek_deepseek-v3.2"
    )

    parser.add_argument(
        "--prompt-technique",
        type=str,
        default=None,
        choices=["zero-shot", "few-shot", "cot"],
        help="Teknik prompt yang ingin dievaluasi."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluasi semua teknik dan semua model di dalam folder rag_on."
    )

    parser.add_argument(
        "--reference-folder",
        type=str,
        default="gt_ct",
        help="Folder ground truth/reference. Default: gt_ct"
    )

    parser.add_argument(
        "--input-root",
        type=str,
        default="hasil_prompt_engineering_retrieval",
        help="Root folder hasil prediksi. Default: hasil_prompt_engineering_retrieval"
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="hasil_evaluasi_prompt_engineering_retrieval",
        help="Root folder output evaluasi. Default: hasil_evaluasi_prompt_engineering_retrieval"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="id",
        help="Bahasa untuk BERTScore. Default: id"
    )

    args = parser.parse_args()

    if args.all:
        evaluate_all(
            reference_folder=args.reference_folder,
            input_root=args.input_root,
            output_root=args.output_root,
            lang=args.lang
        )
    else:
        if args.model_name is None or args.prompt_technique is None:
            print("[GAGAL] Kalau tidak pakai --all, wajib isi --model-name dan --prompt-technique.")
            print("\nContoh:")
            print(
                "python prompt_evaluator2.py "
                "--model-name deepseek_deepseek-v3.2 "
                "--prompt-technique cot"
            )
        else:
            evaluate_one_folder(
                model_name=args.model_name,
                prompt_technique=args.prompt_technique,
                reference_folder=args.reference_folder,
                input_root=args.input_root,
                output_root=args.output_root,
                lang=args.lang
            )