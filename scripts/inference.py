from __future__ import annotations

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


if __name__ == "__main__":
    classifier = IntentClassification(config_path="configs/inference.yaml")
    sample_message = "Toi muon khoa the ngay vi nghi bi lo thong tin."
    predicted_intent = classifier(sample_message)
    print(f"Message: {sample_message}")
    print(f"Predicted intent: {predicted_intent}")
