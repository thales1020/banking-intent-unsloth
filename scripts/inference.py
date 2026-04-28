from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel


class IntentClassification:
    def __init__(self, config_path: str = "configs/inference.yaml"):
        repo_root = Path(__file__).resolve().parents[1]
        cfg_path = repo_root / config_path

        with cfg_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_dir = repo_root / self.config["checkpoint"]["model_dir"]
        infer_cfg = self.config["inference"]

        adapter_config_path = model_dir / "adapter_config.json"
        try:
            if not model_dir.exists() or not model_dir.is_dir():
                raise FileNotFoundError(f"Khong tim thay thu muc mo hinh: {model_dir}")
            if not adapter_config_path.exists():
                raise FileNotFoundError(
                    f"Thieu file adapter: {adapter_config_path}. "
                    "Hay dam bao ban da fine-tune va luu LoRA adapter dung thu muc."
                )

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_dir),
                max_seq_length=infer_cfg["max_seq_length"],
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Khong the load LoRA adapter tu thu muc {model_dir}. Chi tiet loi: {exc}"
            ) from exc

        print(f"Da load thanh cong LoRA adapters tu {model_dir} len mo hinh goc")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, text: str) -> str:
        infer_cfg = self.config["inference"]
        prompt = infer_cfg["prompt_template"].format(text=text)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=infer_cfg["max_new_tokens"],
                do_sample=infer_cfg["do_sample"],
                temperature=infer_cfg["temperature"],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        intent = generated[len(prompt) :].strip()
        intent = intent.split("\n", 1)[0].strip(" -:\t")

        if not intent:
            return "unknown_intent"
        return intent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI demo for BANKING77 intent classification.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML (default: configs/inference.yaml)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single message to classify.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional ground-truth label for the single --text input.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive CLI mode.",
    )
    return parser.parse_args()


def run_interactive(classifier: IntentClassification) -> None:
    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            text = input("Nhap tin nhan> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit interactive mode.")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", "q"}:
            print("Exit interactive mode.")
            break

        # Support entering ground truth alongside text using a tab separator: "text<TAB>label"
        gt = None
        if "\t" in text:
            text, gt = text.split("\t", 1)
            text = text.strip()
            gt = gt.strip()

        predicted_intent = classifier(text)
        print(f"Predicted intent: {predicted_intent}")
        if gt:
            print(f"Ground truth: {gt}")


def run_from_stdin(classifier: IntentClassification) -> None:
    lines = [line.strip() for line in sys.stdin if line.strip()]
    if not lines:
        raise ValueError("No input found in stdin.")

    for text in lines:
        gt = None
        if "\t" in text:
            text, gt = text.split("\t", 1)
            text = text.strip()
            gt = gt.strip()

        predicted_intent = classifier(text)
        print(f"Input: {text}")
        print(f"Predicted intent: {predicted_intent}")
        if gt:
            print(f"Ground truth: {gt}")


if __name__ == "__main__":
    args = parse_args()
    classifier = IntentClassification(config_path=args.config)

    if args.text:
        predicted_intent = classifier(args.text)
        print(f"Input: {args.text}")
        print(f"Predicted intent: {predicted_intent}")
        if args.label:
            print(f"Ground truth: {args.label}")
    elif args.interactive:
        run_interactive(classifier)
    elif not sys.stdin.isatty():
        run_from_stdin(classifier)
    else:
        run_interactive(classifier)
