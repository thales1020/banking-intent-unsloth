from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_train_value(config: dict, key: str, default=None):
    if key in config:
        return config[key]

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    save_cfg = config.get("save", {})

    fallback_map = {
        "max_seq_length": model_cfg.get("max_seq_length", default),
        "batch_size": training_cfg.get("per_device_train_batch_size", default),
        "max_steps": training_cfg.get("max_steps", default),
        "learning_rate": training_cfg.get("learning_rate", default),
        "output_dir": training_cfg.get("output_dir", save_cfg.get("output_dir", default)),
        "train_csv": data_cfg.get("train_csv", default),
        "text_column": data_cfg.get("text_column", default),
        "label_column": data_cfg.get("label_column", default),
        "output_text_column": data_cfg.get("output_text_column", "formatted_text"),
        "prompt_template": data_cfg.get("prompt_template", default),
        "seed": training_cfg.get("seed", default),
    }
    return fallback_map.get(key, default)


def build_prompt_dataset(config: dict, repo_root: Path) -> Dataset:
    csv_path = repo_root / get_train_value(config, "train_csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    text_col = get_train_value(config, "text_column")
    label_col = get_train_value(config, "label_column")
    output_text_col = get_train_value(config, "output_text_column", "formatted_text")
    template = get_train_value(config, "prompt_template")

    if text_col not in df.columns:
        raise ValueError(f"Missing required text column: {text_col}")

    # Fallback to human-readable labels if available from preprocessing.
    if label_col not in df.columns and "label_text" in df.columns:
        df[label_col] = df["label_text"]

    if label_col not in df.columns:
        raise ValueError(f"Missing required label column: {label_col}")

    formatted_text = [
        template.format(text=str(text), label=str(label))
        for text, label in zip(df[text_col], df[label_col])
    ]

    df["formatted_text"] = formatted_text
    if output_text_col != "formatted_text":
        df[output_text_col] = formatted_text

    return Dataset.from_pandas(df, preserve_index=False)


def build_model_and_tokenizer(config: dict):
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    seed = get_train_value(config, "seed")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    return model, tokenizer


def train(config: dict, repo_root: Path):
    dataset = build_prompt_dataset(config, repo_root)
    model, tokenizer = build_model_and_tokenizer(config)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = repo_root / get_train_value(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len = get_train_value(config, "max_seq_length")
    sft_config_params = inspect.signature(SFTConfig).parameters
    sft_config_kwargs = {
        "per_device_train_batch_size": get_train_value(config, "batch_size"),
        "gradient_accumulation_steps": 4,
        "max_steps": get_train_value(config, "max_steps"),
        "learning_rate": get_train_value(config, "learning_rate"),
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "output_dir": str(output_dir),
    }

    if "max_seq_length" in sft_config_params:
        sft_config_kwargs["max_seq_length"] = seq_len
    elif "max_length" in sft_config_params:
        sft_config_kwargs["max_length"] = seq_len

    args = SFTConfig(**sft_config_kwargs)

    sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": args,
    }

    if "processing_class" in sft_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "dataset_text_field" in sft_trainer_params:
        trainer_kwargs["dataset_text_field"] = "formatted_text"
    if "max_seq_length" in sft_trainer_params:
        trainer_kwargs["max_seq_length"] = seq_len

    trainer = SFTTrainer(**trainer_kwargs)

    trainer.train()

    save_dir = repo_root / config["save"]["output_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    print(f"Saved LoRA adapter and tokenizer to: {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BANKING77 classifier with Unsloth QLoRA.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to train config YAML (default: configs/train.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config

    config = load_config(config_path)
    train(config=config, repo_root=repo_root)


if __name__ == "__main__":
    main()
