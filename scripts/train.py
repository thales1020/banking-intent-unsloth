from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt_dataset(config: dict, repo_root: Path) -> Dataset:
    data_cfg = config["data"]
    csv_path = repo_root / data_cfg["train_csv"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]
    output_text_col = data_cfg["output_text_column"]
    template = data_cfg["prompt_template"]

    if text_col not in df.columns:
        raise ValueError(f"Missing required text column: {text_col}")

    # Fallback to human-readable labels if available from preprocessing.
    if label_col not in df.columns and "label_text" in df.columns:
        df[label_col] = df["label_text"]

    if label_col not in df.columns:
        raise ValueError(f"Missing required label column: {label_col}")

    df[output_text_col] = [
        template.format(text=str(text), label=str(label))
        for text, label in zip(df[text_col], df[label_col])
    ]

    return Dataset.from_pandas(df, preserve_index=False)


def build_model_and_tokenizer(config: dict):
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    seed = config["training"]["seed"]

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

    training_cfg = config["training"]
    data_cfg = config["data"]

    output_dir = repo_root / training_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        max_steps=training_cfg["max_steps"],
        warmup_steps=training_cfg.get("warmup_steps", 0),
        logging_steps=training_cfg.get("logging_steps", 5),
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        seed=training_cfg["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=data_cfg["output_text_column"],
        max_seq_length=config["model"]["max_seq_length"],
        args=training_args,
    )

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
