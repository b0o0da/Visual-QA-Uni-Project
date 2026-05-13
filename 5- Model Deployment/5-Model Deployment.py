import os
import pickle
from typing import Tuple, List

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================
# CUSTOM CSS
# ==================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.main { background: #0d0f14; }
[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #2a2d3a;
}

.hero {
    background: linear-gradient(135deg, #1a1d28 0%, #1e2235 100%);
    border: 1px solid #2a2d3a;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
}
.hero h1 { color: #e2e8f0; font-size: 28px; margin-bottom: 8px; }
.hero p  { color: #6b7280; font-size: 14px; }

.answer-box {
    background: linear-gradient(135deg, #1a1d28 0%, #1e2235 100%);
    border: 2px solid #5b6af0;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-top: 16px;
}
.answer-label {
    color: #6b7280;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.answer-text {
    color: #e2e8f0;
    font-size: 22px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

.detection-card {
    background: #1a1d28;
    border-left: 3px solid #f0856b;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #a0aec0;
}

.section-header {
    background: linear-gradient(90deg, #5b6af0 0%, transparent 100%);
    padding: 10px 16px;
    border-radius: 8px;
    margin: 20px 0 12px 0;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #e2e8f0;
}

.metric-card {
    background: #1a1d28;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-label { color: #6b7280; font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
.metric-value { color: #e2e8f0; font-size: 22px; font-family: 'Space Mono', monospace; font-weight: 700; }

.stButton > button {
    background: linear-gradient(135deg, #5b6af0, #818cf8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 1px;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stFileUploader"] {
    background: #1a1d28;
    border: 2px dashed #2a2d3a;
    border-radius: 12px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==================================
# CONSTANTS
# ==================================
IMG_SIZE  = 224
Q_MAX_LEN = 32
A_MAX_LEN = 12

# ==================================
# PATHS
# ==================================
VQA_MODEL_PATH = r"Model.keras"
Q_VOCAB_PATH   = r"question_vocab_V4.pkl"
A_VOCAB_PATH   = r"answer_vocab_V4.pkl"
YOLO_MODEL_PATH = r"best.pt"

# ==================================
# CUSTOM LOSS / ACCURACY
# ==================================
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

def masked_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    y_pred_ids = tf.argmax(y_pred, axis=-1)
    y_true     = tf.cast(y_true, tf.int32)
    y_pred_ids = tf.cast(y_pred_ids, tf.int32)
    matches    = tf.cast(tf.equal(y_true, y_pred_ids), tf.float32)
    mask       = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches    = matches * mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)

# ==================================
# LOAD VQA MODEL
# ==================================
@st.cache_resource
def load_vqa_assets(model_path, q_vocab_path, a_vocab_path):
    model = load_model(
        model_path,
        custom_objects={
            "masked_loss"    : masked_loss,
            "masked_accuracy": masked_accuracy,
        }
    )
    with open(q_vocab_path, "rb") as f:
        Q_vocab = pickle.load(f)
    with open(a_vocab_path, "rb") as f:
        A_vocab = pickle.load(f)

    Q_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=len(Q_vocab),
        output_mode="int",
        output_sequence_length=Q_MAX_LEN
    )
    Q_vectorizer.set_vocabulary(Q_vocab)

    A_idx_to_word = np.array(A_vocab)
    A_word_to_idx = {w: i for i, w in enumerate(A_vocab)}

    return {
        "model"         : model,
        "Q_vectorizer"  : Q_vectorizer,
        "A_idx_to_word" : A_idx_to_word,
        "A_word_to_idx" : A_word_to_idx,
    }

# ==================================
# LOAD YOLO MODEL
# ==================================
@st.cache_resource
def load_yolo(model_path):
    from ultralytics import YOLO
    return YOLO(model_path)

# ==================================
# VQA INFERENCE
# ==================================
def generate_answer(image: Image.Image, question: str, vqa_assets) -> str:
    model         = vqa_assets["model"]
    Q_vectorizer  = vqa_assets["Q_vectorizer"]
    A_idx_to_word = vqa_assets["A_idx_to_word"]
    A_word_to_idx = vqa_assets["A_word_to_idx"]

    # Preprocess image
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Preprocess question
    q_seq    = Q_vectorizer(tf.constant([question]))
    q_seq    = q_seq[:, :Q_MAX_LEN - 1]
    pad_size = (Q_MAX_LEN - 1) - tf.shape(q_seq)[1]
    q_seq    = tf.pad(q_seq, [[0, 0], [0, pad_size]])

    # ✅ start مش موجود فنبدأ بـ index 2 (أول كلمة حقيقية بعد padding و UNK)
    end_id     = A_word_to_idx.get("end", 1521)  # ✅ end موجود في 1521
    answer_ids = [2]  # ✅ ابدأ بأول كلمة حقيقية

    for _ in range(A_MAX_LEN - 1):
        a_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [answer_ids], maxlen=A_MAX_LEN - 1, padding="post"
        )
        a_seq   = tf.cast(a_seq, tf.int32)
        preds   = model.predict([img, q_seq, a_seq], verbose=0)
        next_id = int(np.argmax(preds[0, len(answer_ids) - 1]))

        # ✅ وقف عند end token أو padding
        if next_id == end_id or next_id == 0:
            break

        answer_ids.append(next_id)

    # ✅ استثني الـ padding و UNK و end
    words = [
        A_idx_to_word[i]
        for i in answer_ids
        if i not in [0, 1, 2, end_id] and i < len(A_idx_to_word)
    ]

    return " ".join(words) if words else "No answer generated."

# ==================================
# YOLO INFERENCE
# ==================================
def run_yolo(image: Image.Image, model, conf: float):
    rgb         = np.array(image.convert("RGB"))
    bgr         = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results     = model.predict(bgr, conf=conf, verbose=False)
    plotted     = results[0].plot()
    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    detections = []
    boxes      = results[0].boxes
    names      = model.names

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_v = float(box.conf[0].item())
            detections.append({
                "class"     : names[cls_id],
                "confidence": round(conf_v, 4)
            })

    return plotted_rgb, detections

# ==================================
# SESSION STATE
# ==================================
if "selected_example" not in st.session_state:
    st.session_state["selected_example"] = ""

# ==================================
# SIDEBAR
# ==================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    use_yolo = st.toggle(
        "Enable YOLO Detection",
        value=True,
        key="yolo_toggle"
    )

    if use_yolo:
        conf_threshold = st.slider(
            "YOLO Confidence Threshold",
            min_value=0.05, max_value=1.0,
            value=0.25, step=0.05,
            key="yolo_conf_slider"
        )
    else:
        conf_threshold = 0.25

    st.markdown("---")
    st.markdown("""
    <div style="color:#4b5563;font-size:11px;line-height:1.8">
    🔍 Visual Question Answering<br>
    🎯 YOLO Object Detection<br>
    🧠 EfficientNetB5 + Seq2Seq
    </div>
    """, unsafe_allow_html=True)

# ==================================
# HERO
# ==================================
st.markdown("""
<div class="hero">
    <h1>🔍 Visual Question Answering</h1>
    <p>Upload an image, ask a question, and get an AI-powered answer</p>
</div>
""", unsafe_allow_html=True)

# ==================================
# LOAD MODELS
# ==================================
vqa_assets = None
yolo_model = None
vqa_error  = None
yolo_error = None

try:
    vqa_assets = load_vqa_assets(VQA_MODEL_PATH, Q_VOCAB_PATH, A_VOCAB_PATH)
except Exception as e:
    vqa_error = str(e)

if use_yolo:
    try:
        yolo_model = load_yolo(YOLO_MODEL_PATH)
    except Exception as e:
        yolo_error = str(e)

if vqa_error:
    st.error(f"VQA Model Error: {vqa_error}")
if yolo_error:
    st.warning(f"YOLO Model Error: {yolo_error}")

# ==================================
# MAIN LAYOUT
# ==================================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-header">📁 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
        key="file_uploader"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=200, caption="Uploaded Image")

        w, h = image.size
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Width</div><div class="metric-value">{w}px</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Height</div><div class="metric-value">{h}px</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Format</div><div class="metric-value">{image.format or "RGB"}</div></div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">💬 Ask a Question</div>', unsafe_allow_html=True)

    st.markdown("<p style='color:#6b7280;font-size:12px;margin-top:8px'>💡 Examples:</p>", unsafe_allow_html=True)
    examples = [
        "What is the main subject?",
        "What color is dominant?",
        "Is this indoors or outdoors?",
        "What is happening in the image?",
    ]

    ex_cols = st.columns(2)
    for i, ex in enumerate(examples):
        with ex_cols[i % 2]:
            if st.button(ex, key=f"ex_btn_{i}"):
                st.session_state["selected_example"] = ex
                st.rerun()

    question = st.text_input(
        "Your question:",
        value=st.session_state["selected_example"],
        placeholder="What is the main subject of the image?",
        label_visibility="collapsed",
        key="question_input"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "🔍 Predict",
        type="primary",
        key="predict_btn",
        disabled=(uploaded_file is None or not question.strip())
    )

    # ==================================
    # PREDICTION
    # ==================================
    if predict_btn and uploaded_file and question.strip():
        with st.spinner("Running models..."):

            if vqa_assets:
                answer = generate_answer(image, question, vqa_assets)
            else:
                answer = "VQA model not loaded."

            if use_yolo and yolo_model:
                detected_img, detections = run_yolo(image, yolo_model, conf_threshold)
            else:
                detected_img, detections = None, []

        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-label">Answer</div>
            <div class="answer-text">{answer}</div>
        </div>
        """, unsafe_allow_html=True)

        if use_yolo:
            st.markdown('<div class="section-header">🎯 YOLO Detection</div>', unsafe_allow_html=True)

            if detected_img is not None:
                st.image(detected_img, width=400, caption="Detected Objects")

                if detections:
                    st.markdown(f"**{len(detections)} object(s) detected:**")
                    for d in detections:
                        conf_pct = int(d['confidence'] * 100)
                        st.markdown(f"""
                        <div class="detection-card">
                            <span>🏷️ {d['class']}</span>
                            <span style="color:#5b6af0;float:right">{conf_pct}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No objects detected.", icon="ℹ️")
            else:
                st.warning("YOLO model not loaded.")

# ==================================
# FOOTER
# ==================================
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4b5563;font-size:11px;font-family:'Space Mono',monospace">
Visual Question Answering · EfficientNetB5 + Seq2Seq · YOLO Detection
</div>
""", unsafe_allow_html=True)