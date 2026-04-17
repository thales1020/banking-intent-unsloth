from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path

import pandas as pd
import torch
import unsloth
import yaml
from sklearn.metrics import accuracy_score, classification_report
from unsloth import FastLanguageModel


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_value(config: dict, key: str, default=None):
    if key in config:
        return config[key]

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    save_cfg = config.get("save", {})

    fallback_map = {
        "max_seq_length": model_cfg.get("max_seq_length", default),
        "test_csv": data_cfg.get("test_csv", "sample_data/test.csv"),
        "text_column": data_cfg.get("text_column", "text"),
        "label_column": data_cfg.get("label_column", "label"),
        "output_dir": save_cfg.get("output_dir", "saved_model"),
        "prompt_template": data_cfg.get("prompt_template", default),
    }
    return fallback_map.get(key, default)


def build_prompt_prefix(config: dict) -> str:
    template = get_value(config, "prompt_template")
    if template and "{label}" in template:
        prefix = template.replace("{label}", "")
        return prefix.rstrip()

    text_col = get_value(config, "text_column", "text")
    return f"Tin nhan: {{{text_col}}}\nY dinh:"


def normalize_prediction(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.split("\n", 1)[0].strip()
    cleaned = cleaned.strip(" -:\t").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def extract_label_name(value):
    if isinstance(value, str):
        return value.strip()
    return str(value)


def load_model_and_tokenizer(config: dict, repo_root: Path):
    save_dir = repo_root / get_value(config, "output_dir")
    seq_len = get_value(config, "max_seq_length")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(save_dir),
        max_seq_length=seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def batch_predict(model, tokenizer, texts: list[str], batch_size: int = 8) -> list[str]:
    predictions: list[str] = []
    device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for prompt_text, full_text in zip(batch_texts, decoded):
            if full_text.startswith(prompt_text):
                label_text = full_text[len(prompt_text) :]
            else:
                label_text = full_text
            predictions.append(normalize_prediction(label_text))

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BANKING77 model on the test split.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to config YAML (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="sample_data/test.csv",
        help="Path to test CSV (default: sample_data/test.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for batched generation (default: 8)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = load_config(repo_root / args.config)

    test_csv = repo_root / args.test_csv
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df = pd.read_csv(test_csv)
    text_col = get_value(config, "text_column")
    label_col = get_value(config, "label_column")

    if text_col not in df.columns:
        raise ValueError(f"Missing required text column: {text_col}")
    if label_col not in df.columns and "label_text" not in df.columns:
        raise ValueError(f"Missing required label column: {label_col}")

    if label_col in df.columns:
        y_true = [extract_label_name(value) for value in df[label_col].tolist()]
    else:
        y_true = [extract_label_name(value) for value in df["label_text"].tolist()]

    prompt_prefix = build_prompt_prefix(config)
    prompts = [prompt_prefix.format(text=str(text)) for text in df[text_col].tolist()]

    model, tokenizer = load_model_and_tokenizer(config, repo_root)
    y_pred = batch_predict(model, tokenizer, prompts, batch_size=args.batch_size)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
