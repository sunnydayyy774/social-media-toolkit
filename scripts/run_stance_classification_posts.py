"""Classify post-level stance toward Japan constitutional revision.

This script reads the processed post analysis table, classifies only
include_in_analysis=true rows, and writes an incremental CSV that can be
resumed after interruption.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "analysis_inputs" / "posts_for_analysis_topic_labeled.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "stance_analysis" / "posts_stance.csv"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

STANCE_LABELS = {
    "oppose_revision",
    "support_revision",
    "neutral_descriptive",
    "unclear",
}

OUTPUT_COLUMNS = [
    "aweme_id",
    "create_time",
    "topic",
    "text",
    "text_for_stance",
    "stance_label",
    "stance_confidence",
    "stance_reason",
    "model_name",
    "include_in_analysis",
    "classification_status",
    "error",
    "updated_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-level stance classification with Qwen.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--text-column", default="text_clean_bertopic_with_asr")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Only print input counts; do not load model.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing successful rows in output.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def check_dependencies(load_model: bool) -> None:
    missing = [
        package
        for module, package in {
            "pandas": "pandas",
            "torch": "torch",
            "transformers": "transformers",
        }.items()
        if importlib.util.find_spec(module) is None
    ]
    if load_model and importlib.util.find_spec("accelerate") is None:
        logging.warning(
            "Package 'accelerate' is not installed. If model loading fails on the target machine, install accelerate."
        )
    if missing:
        raise SystemExit("Missing required dependencies: " + ", ".join(missing))


def clean_text_for_stance(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:max_chars]


def build_prompt(text: str) -> str:
    return f"""
You are classifying stance in Chinese Douyin video text.

Target issue: Japan's constitutional revision, especially revision of the Peace Constitution / Article 9, Japan's remilitarization, and Self-Defense Forces normalization.

Choose exactly one label:
- oppose_revision: the text criticizes, warns against, or frames Japan's constitutional revision/remilitarization as dangerous or wrong.
- support_revision: the text supports or justifies Japan's constitutional revision, remilitarization, or becoming a normal military power.
- neutral_descriptive: the text mainly reports or describes related facts without a clear stance.
- unclear: the stance toward Japan constitutional revision is unclear, too short, or not actually about the target issue.

Return only valid JSON with these keys:
{{
  "stance_label": "oppose_revision|support_revision|neutral_descriptive|unclear",
  "confidence": 0.0,
  "reason": "short English explanation"
}}

Text:
{text}
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model output")
    data = json.loads(match.group(0))
    label = str(data.get("stance_label", "")).strip()
    if label not in STANCE_LABELS:
        raise ValueError(f"Invalid stance_label: {label}")
    confidence = float(data.get("confidence", 0))
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", "")).strip()
    return {"stance_label": label, "confidence": confidence, "reason": reason}


def load_existing_success(output_path: Path, overwrite: bool) -> set[str]:
    if overwrite or not output_path.exists():
        return set()
    existing = pd.read_csv(output_path, dtype=str).fillna("")
    if "aweme_id" not in existing.columns or "classification_status" not in existing.columns:
        return set()
    return set(existing.loc[existing["classification_status"].eq("success"), "aweme_id"].astype(str))


def append_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def load_model_and_tokenizer(model_name: str, device: str, dtype_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32 if dtype_name == "auto" else torch_dtype,
            trust_remote_code=True,
        )
        model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "auto" else {"": device},
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def classify_one(model, tokenizer, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    import torch

    messages = [
        {"role": "system", "content": "You are a careful stance classification assistant."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_text], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return extract_json(response)


def main() -> None:
    args = parse_args()
    setup_logging()
    check_dependencies(load_model=not args.dry_run)

    if args.batch_size != 1:
        logging.warning("Qwen generation is processed one row at a time; --batch-size is currently kept for interface only.")

    posts = pd.read_csv(args.input, dtype=str).fillna("")
    required = {"aweme_id", "create_time", "topic", "text", args.text_column, "include_in_analysis"}
    missing = required - set(posts.columns)
    if missing:
        raise SystemExit("Input is missing required columns: " + ", ".join(sorted(missing)))

    selected = posts[posts["include_in_analysis"].str.lower().eq("true")].copy()
    selected["text_for_stance"] = selected[args.text_column].map(lambda value: clean_text_for_stance(value, args.max_chars))
    selected = selected[selected["text_for_stance"].str.len() > 0].copy()
    eligible_count = len(selected)

    completed = load_existing_success(args.output, args.overwrite)
    if completed:
        selected = selected[~selected["aweme_id"].astype(str).isin(completed)].copy()
    if args.max_items is not None:
        selected = selected.head(args.max_items).copy()

    logging.info("Input posts: %s", len(posts))
    logging.info("Rows with include_in_analysis=true and non-empty text: %s", eligible_count)
    logging.info("Already completed success rows skipped: %s", len(completed))
    logging.info("Rows to classify in this run: %s", len(selected))
    logging.info("Output CSV: %s", args.output.resolve())

    if args.dry_run:
        return

    logging.info("Loading model: %s", args.model_name)
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device, args.dtype)

    for index, row in enumerate(selected.itertuples(index=False), start=1):
        row_dict = row._asdict()
        now = datetime.now().isoformat(timespec="seconds")
        output_row = {
            "aweme_id": row_dict["aweme_id"],
            "create_time": row_dict["create_time"],
            "topic": row_dict["topic"],
            "text": row_dict["text"],
            "text_for_stance": row_dict["text_for_stance"],
            "stance_label": "",
            "stance_confidence": "",
            "stance_reason": "",
            "model_name": args.model_name,
            "include_in_analysis": row_dict["include_in_analysis"],
            "classification_status": "failed",
            "error": "",
            "updated_at": now,
        }
        try:
            prompt = build_prompt(row_dict["text_for_stance"])
            result = classify_one(model, tokenizer, prompt, args.max_new_tokens)
            output_row.update(
                {
                    "stance_label": result["stance_label"],
                    "stance_confidence": result["confidence"],
                    "stance_reason": result["reason"],
                    "classification_status": "success",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep row-level failures resumable.
            output_row["error"] = str(exc)

        append_rows(args.output, [output_row])
        if index % 10 == 0 or index == len(selected):
            logging.info("Processed %s/%s rows", index, len(selected))

    logging.info("Stance classification complete: %s", args.output.resolve())


if __name__ == "__main__":
    main()
