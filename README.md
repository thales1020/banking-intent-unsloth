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

## 5) Demo Inference (CLI)

File `scripts/inference.py` bây giờ hỗ trợ chạy demo qua CLI theo nhiều chế độ:

- Single text (truyền 1 câu):

```bash
python scripts/inference.py --text "Toi muon khoa the ngay vi nghi bi lo thong tin."
```

- Single text với ground-truth (tùy chọn `--label`):

```bash
python scripts/inference.py --text "Toi muon khoa the ngay vi nghi bi lo thong tin." --label 5
```

- Interactive mode (gõ nhiều câu, hỗ trợ nhập kèm ground-truth bằng dấu tab):

```bash
python scripts/inference.py --interactive
# Trong interactive: nhập dạng: <text>\t<label>  hoặc chỉ <text>
```

- Pipe từ `stdin` (hỗ trợ mỗi dòng là `text` hoặc `text<TAB>label`):

Linux / macOS / WSL:

```bash
printf "Toi muon khoa the ngay vi nghi bi lo thong tin.\n" | python scripts/inference.py
printf "Toi muon khoa the ngay vi nghi bi lo thong tin.\t5\n" | python scripts/inference.py
```

PowerShell (Windows):

```powershell
"Toi muon khoa the ngay vi nghi bi lo thong tin." | python scripts/inference.py
"Toi muon khoa the ngay vi nghi bi lo thong tin.`t5" | python scripts/inference.py
```

Lưu ý:
- Phải có thư mục `saved_model/` chứa LoRA adapter và tokenizer trước khi chạy.
- `--config` cho phép truyền file YAML khác nếu cần: `--config configs/inference.yaml`.

## 6) Test Results & Analysis

### Cách chạy test

Sau khi hoàn tất training, chạy:

```bash
python scripts/test.py
```

Script sẽ:
- Load LoRA adapter từ `saved_model/`
- Đọc test set từ `sample_data/test.csv`
- Sinh dự đoán (generation)
- Chuẩn hóa nhãn (normalize) và cố gắng force dự đoán về nhãn đã biết
- In ra **Raw Accuracy** (trước khi force-to-known) và **Accuracy** (sau force-to-known)
- In ra **Classification Report** với precision, recall, F1 cho từng class

### Tại sao mô hình test tệ?

Nếu accuracy thấp (< 70%), các nguyên nhân tiềm ẩn:

1. **Model quá nhỏ (0.5B parameters):**
   - Mô hình Qwen2.5-0.5B có khả năng học hạn chế cho task phân loại 77 intent (80 classes).
   - So sánh: các mô hình lớn hơn (3B, 7B+) thường đạt kết quả tốt hơn trên task phức tạp.

2. **Training data quá ít:**
   - Nếu dùng `--sample-fraction=0.1` hoặc thấp hơn, mỗi class chỉ có ~5-6 mẫu huấn luyện.
   - Overfitting trên train set nhưng generalize tệ lên test set.
   - **Giải pháp:** Tăng `--sample-fraction` lên 0.5 hoặc 1.0 (full dataset) để có đủ dữ liệu huấn luyện.

3. **Hyperparameters không tối ưu:**
   - `num_train_epochs=3` quá ít → model chưa hội tụ.
   - `learning_rate=2e-4` có thể quá cao (dễ bị thrashing) hoặc quá thấp (hội tụ chậm).
   - `batch_size=2` quá nhỏ → gradient noise cao → training không ổn định.
   - **Giải pháp:** Thử tăng epochs lên 5-10, giảm learning rate xuống 1e-4, tăng batch size lên 4-8 (nếu VRAM cho phép).

4. **Prompt template không tốt:**
   - Prompt hiện tại: `### Instructions: Classify the banking message into one unique intent code from 0 to 76. Do not output anything else.\n### Message: {text}\n### Intent Code:`
   - Có thể model gặp khó khăn trong việc sinh đúng format (chỉ số 0-76).
   - **Giải pháp:** Thử prompt rõ ràng hơn hoặc thêm few-shot examples trong prompt.

5. **BANKING77 dataset có overlap intent:**
   - Một số intent trong BANKING77 có ý nghĩa tương tự (ví dụ: "activate card", "apply for card") → khó phân biệt.
   - Ngay cả con người cũng có thể nhầm lẫn → accuracy cao không phải lúc nào cũng khả thi.

6. **Model underfitting hoặc overfitting:**
   - Underfitting: loss train còn cao → cần training lâu hơn, learning rate cao hơn.
   - Overfitting: accuracy train cao nhưng test thấp → cần regularization (LoRA dropout), early stopping, hoặc thêm dữ liệu.

### Kết quả test mẫu

*Sẽ cập nhật sau khi chạy training + test trên full dataset.*

```
Raw Accuracy (no label forcing): 0.7234
Accuracy: 0.7568

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.92      0.88        13
           1       0.80      0.75      0.77        12
           ...
    weighted avg   0.76      0.76      0.76       120
```

### Cách cải thiện

1. **Tăng dữ liệu:** Sử dụng full BANKING77 dataset (`--sample-fraction=1.0`).
2. **Tăng model size:** Thử Qwen2.5-1.5B hoặc 3B thay vì 0.5B.
3. **Fine-tune hyperparameters:** Tăng epochs, điều chỉnh learning rate, tăng batch size.
4. **Cải thiện prompt:** Thêm instructions chi tiết hoặc few-shot examples.
5. **Regularization:** Tăng LoRA dropout hoặc weight decay để giảm overfitting.

