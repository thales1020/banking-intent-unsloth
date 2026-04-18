from __future__ import annotations

import argparse
import difflib
import re
from collections import Counter
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

    checkpoint_cfg = config.get("checkpoint", {})
    test_cfg = config.get("test", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    save_cfg = config.get("save", {})

    fallback_map = {
        "max_seq_length": model_cfg.get("max_seq_length", default),
        "model_dir": checkpoint_cfg.get("model_dir", save_cfg.get("output_dir", "saved_model")),
        "test_csv": data_cfg.get("test_csv", "sample_data/test.csv"),
        "text_column": data_cfg.get("text_column", "text"),
        "label_column": data_cfg.get("label_column", "label"),
        "output_dir": save_cfg.get("output_dir", "saved_model"),
        "prompt_template": data_cfg.get("prompt_template", default),
        "batch_size": test_cfg.get("batch_size", 8),
        "max_new_tokens": test_cfg.get("max_new_tokens", 24),
        "temperature": test_cfg.get("temperature", 0.0),
        "do_sample": test_cfg.get("do_sample", False),
        "prompt_prefix": test_cfg.get("prompt_template", default),
    }
    return fallback_map.get(key, default)


def build_prompt_prefix(config: dict) -> str:
    template = get_value(config, "prompt_prefix") or get_value(config, "prompt_template")
    if template and "{label}" in template:
        prefix = template.replace("{label}", "")
        return prefix.rstrip()

    text_col = get_value(config, "text_column", "text")
    return f"Tin nhan: {{{text_col}}}\nY dinh:"


def normalize_prediction(text: str) -> str:
    # Tìm tất cả các cụm số trong văn bản trả về
    match = re.search(r'\d+', text)
    if match:
        return match.group() 
    return text.strip()


def canonical_label(text: str) -> str:
    cleaned = normalize_prediction(text).lower()
    cleaned = re.sub(r"[^a-z0-9_ ]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_label_name(value):
    if isinstance(value, str):
        return value.strip()
    return str(value)


def force_to_known_label(pred: str, known_lookup: dict[str, str], default_label: str) -> str:
    key = canonical_label(pred)
    if key in known_lookup:
        return known_lookup[key]

    # Try substring matching before fuzzy matching.
    candidates = [k for k in known_lookup if k and (k in key or key in k)]
    if candidates:
        best_key = max(candidates, key=len)
        return known_lookup[best_key]

    close = difflib.get_close_matches(key, list(known_lookup.keys()), n=1, cutoff=0.35)
    if close:
        return known_lookup[close[0]]

    return default_label


def load_model_and_tokenizer(config: dict, repo_root: Path):
    save_dir = repo_root / get_value(config, "model_dir")
    seq_len = get_value(config, "max_seq_length")
    adapter_config_path = save_dir / "adapter_config.json"

    try:
        if not save_dir.exists() or not save_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc mo hinh: {save_dir}")
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"Thieu file adapter: {adapter_config_path}. "
                "Hay dam bao ban da fine-tune va luu LoRA adapter dung thu muc."
            )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(save_dir),
            max_seq_length=seq_len,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None):
            model.generation_config.max_length = None
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Khong the load LoRA adapter tu thu muc {save_dir}. Chi tiet loi: {exc}"
        ) from exc

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Da load thanh cong LoRA adapters tu {save_dir} len mo hinh goc")
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
                max_new_tokens=batch_predict.max_new_tokens,
                do_sample=batch_predict.do_sample,
                temperature=batch_predict.temperature,
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
        default="configs/test.yaml",
        help="Path to config YAML (default: configs/test.yaml)",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Path to test CSV (default: read from configs/test.yaml)",
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

    test_csv_rel = args.test_csv if args.test_csv else get_value(config, "test_csv", "sample_data/test.csv")
    test_csv = repo_root / test_csv_rel
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

    label_counts = Counter(y_true)
    default_label = label_counts.most_common(1)[0][0]
    known_lookup = {canonical_label(label): label for label in sorted(set(y_true))}

    prompt_prefix = build_prompt_prefix(config)
    prompts = [prompt_prefix.format(text=str(text)) for text in df[text_col].tolist()]

    model, tokenizer = load_model_and_tokenizer(config, repo_root)
    batch_predict.max_new_tokens = get_value(config, "max_new_tokens")
    batch_predict.do_sample = get_value(config, "do_sample")
    batch_predict.temperature = get_value(config, "temperature")
    raw_predictions = batch_predict(
        model,
        tokenizer,
        prompts,
        batch_size=get_value(config, "batch_size", args.batch_size),
    )

    y_true_canonical = [canonical_label(label) for label in y_true]
    raw_pred_canonical = [canonical_label(pred) for pred in raw_predictions]
    raw_accuracy = accuracy_score(y_true_canonical, raw_pred_canonical)

    y_pred = [force_to_known_label(pred, known_lookup, default_label) for pred in raw_predictions]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Raw Accuracy (no label forcing): {raw_accuracy:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()
