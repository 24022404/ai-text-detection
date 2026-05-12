# AI-Generated Text Detection

Phat hien van ban do AI sinh ra (AI-generated text detection) su dung mo hinh phan loai nhi phan: Human-written vs AI-generated.

## Bai toan

Xay dung mo hinh phan loai van ban thanh "Human-written" hoac "AI-generated" tren 2 ngon ngu: tieng Anh va tieng Viet.

## Dataset

**FAIDSet** (`ngocminhta/FAIDSet` tren HuggingFace)

- 85,683 records goc, lay 39,809 records (bo qua phan human-LLM collaborative)
- Nguon LLM: GPT, Gemini, Llama, DeepSeek
- Nguon human: luan van sinh vien, abstract bai bao khoa hoc
- Phan bo:

| Ngon ngu | Human | AI |
|---|---|---|
| Tieng Anh | 7,163 | 10,423 |
| Tieng Viet | 13,066 | 9,161 |

## Mo hinh

### Model 1 — XLM-RoBERTa (Song ngu)
Fine-tune `xlm-roberta-base` tren ca tieng Anh lan tieng Viet cung luc.

### Model 2 — Hybrid (RoBERTa + PhoBERT)
Ket hop 2 model chuyen biet theo routing logic:
- `P(ngon ngu chinh) >= 0.8` : Hard routing — tieng Anh di RoBERTa, tieng Viet di PhoBERT
- `P(ngon ngu chinh) < 0.8` : Soft voting — trung binh xac suat ca 2 model

## Ket qua

| Model | Language | Precision | Recall | F1 |
|---|---|---|---|---|
| XLM-RoBERTa | Overall | 0.98 | 0.98 | 0.98 |
| XLM-RoBERTa | EN | 0.98 | 0.97 | 0.98 |
| XLM-RoBERTa | VI | 0.99 | 0.99 | 0.99 |
| Hybrid | Overall | 0.99 | 0.99 | 0.99 |
| Hybrid | EN | 0.99 | 0.98 | 0.98 |
| Hybrid | VI | 0.99 | 0.99 | 0.99 |

## Cau truc thu muc

```
project/
|
|-- 00_download.ipynb           # Download FAIDSet tu HuggingFace
|-- 01_preprocessing.ipynb      # ELT pipeline: Extract, Load, Transform
|-- 02_training_xlmroberta.ipynb # Train Model 1: XLM-RoBERTa song ngu
|-- 03_training_hybrid.ipynb    # Train Model 2: RoBERTa + PhoBERT hybrid
|-- 04_feature_analysis.ipynb   # Phan tich feature: burstiness, entropy, TTR, avg sentence length
|-- 05_comparison.ipynb         # So sanh 2 model, confusion matrix
|
|-- raw_data/
|   |-- eng/
|   |   |-- AI/
|   |   |-- human/
|   |-- vi/
|       |-- AI/
|       |-- human/
|
|-- processed_data/
|   |-- full.csv
|   |-- train.csv
|   |-- val.csv
|   |-- test.csv
|
|-- models/
|   |-- xlmroberta/
|   |-- roberta/
|   |-- phobert/
|
|-- README.md
```

## Huong dan chay

**Buoc 1** — Download data
```
00_download.ipynb
```

**Buoc 2** — Tien xu ly
```
01_preprocessing.ipynb
```

**Buoc 3** — Train model (chay doc lap, khong phu thuoc nhau)
```
02_training_xlmroberta.ipynb
03_training_hybrid.ipynb
```

**Buoc 4** — Phan tich va so sanh
```
04_feature_analysis.ipynb
05_comparison.ipynb
```

## Yeu cau

```
torch
transformers
scikit-learn
pandas
langdetect
underthesea
sentencepiece
matplotlib
seaborn
scipy
huggingface_hub
```

Cai dat:
```
pip install torch transformers scikit-learn pandas langdetect underthesea sentencepiece matplotlib seaborn scipy huggingface_hub
```

## Thanh vien nhom

| Thanh vien | Phan viec |
|---|---|
| [Ten] | Download data, Preprocessing |
| [Ten] | Training Model 1 (XLM-RoBERTa) |
| [Ten] | Training Model 2 (Hybrid), Feature Analysis |
