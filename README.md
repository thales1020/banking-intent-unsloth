# BANKING77 Intent Classification with Unsloth QLoRA

Du an nay huan luyen mo hinh nhe `unsloth/Qwen2.5-0.5B-Instruct` cho bai toan phan loai y dinh tren BANKING77 bang QLoRA, sau do suy luan nhan y dinh tu cau nhap vao.

## 1) Cai dat moi truong

### Yeu cau
- Python 3.10+ (khuyen nghi 3.10 hoac 3.11)
- GPU CUDA de train nhanh hon (co the chay CPU nhung rat cham)

### Cai dat
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Tren Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Chay huan luyen

Lenh truc tiep:
```bash
python scripts/train.py --config configs/train.yaml
```

Hoac dung script:
```bash
bash train.sh
```

Ket qua sau huan luyen:
- LoRA adapter va tokenizer duoc luu trong thu muc `saved_model/`

## 3) Chay suy luan

Lenh truc tiep:
```bash
python scripts/inference.py
```

Hoac dung script:
```bash
bash inference.sh
```

Script se tao doi tuong `IntentClassification`, truyen vao mot cau tin nhan ngan hang gia dinh va in ra nhan y dinh du doan.

## 4) Danh sach sieu tham so da su dung

### Mo hinh va QLoRA (configs/train.yaml)
- model.name: `unsloth/Qwen2.5-0.5B-Instruct`
- model.max_seq_length: `512`
- model.load_in_4bit: `true`
- lora.r: `16`
- lora.lora_alpha: `16`
- lora.lora_dropout: `0`
- lora.bias: `none`
- lora.target_modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`

### Du lieu train
- data.train_csv: `sample_data/train.csv`
- data.text_column: `text`
- data.label_column: `label` (fallback `label_text` neu can)
- data.output_text_column: `prompt`
- data.prompt_template: `Tin nhan: {text} - Y dinh: {label}`

### Huan luyen voi SFTTrainer
- per_device_train_batch_size: `2`
- gradient_accumulation_steps: `4`
- learning_rate: `2e-4`
- max_steps: `60`
- warmup_steps: `5`
- logging_steps: `5`
- seed: `42`
- optimizer: `adamw_8bit`

### Cau hinh suy luan (configs/inference.yaml)
- checkpoint.model_dir: `saved_model`
- inference.max_seq_length: `512`
- inference.load_in_4bit: `true`
- inference.max_new_tokens: `24`
- inference.temperature: `0.0`
- inference.do_sample: `false`
- inference.prompt_template: `Tin nhan: {text} - Y dinh:`
