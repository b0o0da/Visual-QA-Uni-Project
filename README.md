# 🔍 Visual Question Answering (VQA)

> An end-to-end deep learning system that answers natural language questions about images — combining **EfficientNetB5** for visual understanding, a **Seq2Seq decoder** for answer generation, and **YOLOv8** for real-time object detection.

---

## 📌 Project Overview

This project was built as a university capstone to explore the intersection of **Computer Vision** and **Natural Language Processing**. Given an image and a question about it, the system generates a natural language answer.

**Example:**
| Input Image | Question | Answer |
|---|---|---|
| 🐘 A photo of elephants | `"What animal is in the image?"` | `"elephant"` |
| 🚌 A city street | `"What vehicle is visible?"` | `"bus"` |

---

## 🗂️ Project Structure

```
Visual-QA-Uni-Project/
│
├── 1-Data Collection & Preparation/     # Dataset sourcing & raw data prep
├── 2-Data Analysis & Preprocessing/     # EDA, cleaning, tokenization
├── 3-Model Architecture/                # VQA model design & training
├── 4-Model Evaluation/                  # BLEU, METEOR, Exact Match scoring
└── 5-Model Deployment/                  # Streamlit web app
```

---

## 🧠 Model Architecture

```
Image ──► EfficientNetB5 (Feature Extractor)
                    │
                    ▼
Question ──► TextVectorization ──► Embedding
                    │
                    ▼
            Seq2Seq Decoder
                    │
                    ▼
              Answer Token(s)
```

- **Visual Encoder:** EfficientNetB5 pretrained on ImageNet
- **Text Encoder:** TextVectorization + Embedding layer
- **Decoder:** Seq2Seq with autoregressive generation
- **Object Detection:** YOLOv8 (custom trained — `best.pt`)

---

## 📊 Evaluation Metrics

The model was evaluated using standard NLP metrics:

| Metric | Description |
|---|---|
| **Exact Match** | % of predictions exactly matching ground truth |
| **BLEU-1 to 4** | N-gram precision overlap |
| **METEOR** | Harmonic mean of precision & recall with stemming |

Evaluation was done using **Beam Search** (`beam_width=5`) for better answer quality.

---

## 🚀 Deployment

The app is built with **Streamlit** and supports:

- 📤 Image upload (JPG, PNG, WEBP)
- 💬 Free-text question input
- 🎯 Real-time YOLO object detection overlay
- 📊 Image metadata display (width, height, format)
- 💡 Example question buttons

### Run Locally

```bash
streamlit run "5-Model Deployment.py"
```

> ⚠️ Make sure `Model.keras`, `question_vocab_V4.pkl`, `answer_vocab_V4.pkl`, and `best.pt` are in the same directory.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Deep Learning | TensorFlow / Keras |
| Object Detection | Ultralytics YOLOv8 |
| Image Processing | OpenCV, Pillow |
| Web App | Streamlit |
| Evaluation | NLTK (BLEU, METEOR) |

---

## 📁 Key Files

| File | Description |
|---|---|
| `Model.keras` | Trained VQA model |
| `best.pt` | Trained YOLOv8 weights |
| `question_vocab_V4.pkl` | Question vocabulary |
| `answer_vocab_V4.pkl` | Answer vocabulary (3000 tokens) |
| `5-Model Deployment.py` | Streamlit app entrypoint |

---

## 👥 Team

Built as a university project.

---

## 📄 License

This project is for academic purposes only.
