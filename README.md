# AI-Generated Text Detection

Phát hiện văn bản do AI sinh ra (AI-generated text detection) sử dụng mô hình phân loại nhị phân: Human-written vs AI-generated, trên dữ liệu song ngữ Anh–Việt.

## Bài toán

Xây dựng mô hình phân loại văn bản thành "Human-written" hoặc "AI-generated" trên 2 ngôn ngữ: tiếng Anh và tiếng Việt.

## Dataset

Dữ liệu được lấy từ **2 nguồn**, kết hợp lại thành 3 tập train / val / test.

**FAIDSet** (`ngocminhta/FAIDSet` trên HuggingFace)
- 2024, song ngữ Anh + Việt, văn bản AI sinh từ GPT, Gemini, LLaMA, DeepSeek
- 85,683 record gốc (train+valid+test); sau khi loại nhãn `collaborative` (con người chỉnh sửa văn bản AI) và lọc ngôn ngữ khác EN/VI, còn lại khoảng 39,800 mẫu hợp lệ

**MAGE** (`yaful/MAGE` trên HuggingFace)
- Tiếng Anh, tổng hợp từ 27 LLM, đa lĩnh vực
- Lấy 15,000 mẫu cho train (7,500 Human + 7,500 AI, lấy mẫu ngẫu nhiên), và toàn bộ 60,279 mẫu test

**Phân chia train / val / test sau khi kết hợp:**

| Tập | Tổng | EN Human | EN AI | VI Human | VI AI |
|---|---|---|---|---|---|
| Train (gốc + augmented) | 48,257 | 14,227 | 14,989 | 11,984 | 7,057 |
| Validation | 4,920 | 1,359 | 1,666 | 1,111 | 784 |
| Test EN | 62,621 | 30,899 | 31,722 | — | — |
| Test VI | 3,278 | — | — | 1,958 | 1,320 |

Train gồm: FAIDSet train+valid (EN+VI) + 15,000 mẫu MAGE train + 3,984 mẫu Human được tăng cường bằng **back-translation** (EN→FR→EN và VI→EN→VI, dùng các model `Helsinki-NLP/opus-mt-*`), nhằm cân bằng lại tỉ lệ Human/AI ở tiếng Anh.

## Mô hình

Cả 2 mô hình đều được huấn luyện với kỹ thuật **R-Drop**: mỗi batch forward 2 lần qua cùng model (2 dropout mask khác nhau), thêm KL-divergence loss giữa 2 lần forward để ổn định huấn luyện — `total_loss = CE_loss + 0.5 × KL_loss`.

### Model 1 — XLM-RoBERTa (song ngữ)
Fine-tune `xlm-roberta-base` trên cả tiếng Anh lẫn tiếng Việt cùng lúc, dùng class weighting để bù lệch lớp ở tiếng Anh.

### Model 2 — Hybrid (RoBERTa + PhoBERT)
Huấn luyện riêng `roberta-base` (tiếng Anh) và `vinai/phobert-base-v2` (tiếng Việt, có word segmentation bằng `underthesea`), kết hợp qua routing logic dựa trên `langdetect`:
- `P(ngôn ngữ) ≥ 0.8` và là EN/VI: hard routing — đi thẳng vào RoBERTa hoặc PhoBERT
- `P(ngôn ngữ) < 0.8`, hoặc ngôn ngữ phát hiện không phải EN/VI: soft voting — trung bình xác suất cả 2 model

## Kết quả (Test set, 65,899 mẫu — chưa được model nhìn thấy khi huấn luyện)

| Model | Phạm vi | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| XLM-RoBERTa + R-Drop | Tổng | 0.8193 | 0.7949 | 0.7913 |
| XLM-RoBERTa + R-Drop | EN | 0.8113 | 0.7834 | 0.7799 |
| XLM-RoBERTa + R-Drop | VI | 0.9848 | 0.9890 | 0.9867 |
| Hybrid + R-Drop | Tổng | 0.8210 | 0.7807 | 0.7740 |
| Hybrid + R-Drop | EN | 0.8136 | 0.7679 | 0.7611 |
| Hybrid + R-Drop | VI | 0.9861 | 0.9902 | 0.9880 |

Cả hai model đạt F1 rất cao trên tiếng Việt (~0.99) nhưng còn thấp trên tiếng Anh (~0.76–0.78), chủ yếu do recall của lớp Human thấp (~0.58–0.64) — dễ nhận nhầm văn bản người viết thành AI-generated. XLM-RoBERTa nhích hơn Hybrid khoảng +0.017 F1 tổng thể trên test set, dù val F1 lúc huấn luyện của Hybrid (0.9266) gần với XLM-RoBERTa (0.9310).

## Cấu trúc thư mục

```
project/
|
|-- 01_download_and_preprocess.ipynb   # Download FAIDSet + MAGE, ELT pipeline, back-translation augmentation
|-- 02_training_xlmroberta.ipynb       # Train Model 1: XLM-RoBERTa song ngữ + R-Drop
|-- 03_training_hybrid.ipynb           # Train Model 2: RoBERTa (EN) + PhoBERT (VI) + R-Drop
|-- 04_test_set_evaluation.ipynb       # Đánh giá cả 2 model trên test set, routing distribution
|-- 05_feature_analysis.ipynb          # Feature thống kê (entropy, burstiness, độ dài câu) + attention visualization
|
|
|-- models/
|   |-- xlmroberta-rdrop/              # Kaggle Dataset: xlmroberta-rdrop
|   |-- hybrid-rdrop/
|       |-- roberta/                   # Kaggle Dataset: hybrid-rdrop
|       |-- phobert/
|
|-- README.md
```

## Hướng dẫn chạy

Các notebook được chạy trên Kaggle, đọc/ghi dữ liệu qua Kaggle Dataset (không chạy local trực tiếp nếu chưa chỉnh lại đường dẫn `/kaggle/input/...`).

**Bước 1** — Tải dữ liệu và tiền xử lý (tạo Kaggle Dataset `faidset-processed`)
```
01_download_and_preprocess.ipynb
```

**Bước 2** — Huấn luyện 2 mô hình (chạy độc lập, không phụ thuộc nhau, đều cần `faidset-processed` ở bước 1)
```
02_training_xlmroberta.ipynb   # tạo Kaggle Dataset xlmroberta-rdrop
03_training_hybrid.ipynb       # tạo Kaggle Dataset hybrid-rdrop
```

**Bước 3** — Đánh giá trên test set (cần cả `xlmroberta-rdrop` và `hybrid-rdrop` từ bước 2)
```
04_test_set_evaluation.ipynb
```

**Bước 4** — Phân tích đặc trưng (cần `faidset-processed` và cả 2 model đã train)
```
05_feature_analysis.ipynb
```

## Yêu cầu

```
torch
transformers
scikit-learn
pandas
numpy
langdetect
underthesea
sentencepiece
matplotlib
seaborn
huggingface_hub
datasets
tqdm
```

Cài đặt:
```
pip install torch transformers scikit-learn pandas numpy langdetect underthesea sentencepiece matplotlib seaborn huggingface_hub datasets tqdm
```

