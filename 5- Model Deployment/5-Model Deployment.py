import os
import pickle
from typing import Tuple, List

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(page_title="Image Captioning + YOLO Detection", layout="wide")
st.title("Image Captioning + Object Detection")
st.write("Upload one image, then run both models together.")

# ==================================
# PATHS
# ==================================
YOLO_MODEL_PATH = r"C:\Users\Boda\Documents\GitHub\New folder\Yolo_dataset\best.pt"
CAPTION_MODEL_PATH = r"C:\Users\Boda\Documents\GitHub\New folder\Yolo_dataset\Last Model.keras"
VOCAB_PATH = r"C:\Users\Boda\Documents\GitHub\New folder\Yolo_dataset\vocab.pkl"

# ==================================
# CONFIG FROM YOUR NOTEBOOK
# ==================================
IMG_SIZE = 224
MAX_LEN = 35
START_TOKEN = "start"
END_TOKEN = "end"

# ==================================
# CUSTOM OBJECTS FROM NOTEBOOK
# ==================================
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction="none"
)

def masked_loss(y_true, y_pred):
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    y_pred_ids = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.int32)
    y_pred_ids = tf.cast(y_pred_ids, tf.int32)
    matches = tf.cast(tf.equal(y_true, y_pred_ids), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    matches = matches * mask
    return tf.reduce_sum(matches) / tf.reduce_sum(mask)

# ==================================
# HELPERS
# ==================================
def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))

def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = pil_to_rgb_array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_caption_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def build_word_mappings(vocab: List[str]):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    return word_to_idx, idx_to_word

# ==================================
# LOAD MODELS
# ==================================
@st.cache_resource
def load_yolo_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    return YOLO(model_path)

@st.cache_resource
def load_caption_assets(model_path: str, vocab_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Caption model not found: {model_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    model = load_model(
        model_path,
        custom_objects={
            "masked_loss": masked_loss,
            "masked_accuracy": masked_accuracy,
        },
    )

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    word_to_idx, idx_to_word = build_word_mappings(vocab)

    return {
        "model": model,
        "vocab": vocab,
        "word_to_idx": word_to_idx,
        "idx_to_word": idx_to_word,
    }

# ==================================
# CAPTION INFERENCE
# NOTE:
# This uses greedy decoding because the uploaded notebooks
# did not include a separate beam search inference function.
# ==================================
def generate_caption(image: Image.Image, caption_assets) -> str:
    model = caption_assets["model"]
    word_to_idx = caption_assets["word_to_idx"]
    idx_to_word = caption_assets["idx_to_word"]

    if START_TOKEN not in word_to_idx or END_TOKEN not in word_to_idx:
        return "Vocab is missing 'start' or 'end' token."

    image_tensor = preprocess_caption_image(image)

    start_id = word_to_idx[START_TOKEN]
    end_id = word_to_idx[END_TOKEN]

    decoder_input = np.zeros((1, MAX_LEN - 1), dtype=np.int32)
    decoder_input[0, 0] = start_id

    generated_ids = []

    for t in range(MAX_LEN - 2):
        preds = model.predict([image_tensor, decoder_input], verbose=0)

        # preds shape: (1, MAX_LEN-1, VOCAB_SIZE)
        next_id = int(np.argmax(preds[0, t]))

        if next_id == 0:
            break
        if next_id == end_id:
            break

        generated_ids.append(next_id)

        if t + 1 < MAX_LEN - 1:
            decoder_input[0, t + 1] = next_id

    words = []
    for idx in generated_ids:
        word = idx_to_word.get(idx, "")
        if not word or word in {START_TOKEN, END_TOKEN}:
            continue
        words.append(word)

    if not words:
        return "No caption generated."

    return " ".join(words)

# ==================================
# YOLO INFERENCE
# ==================================
def run_yolo_detection(
    image: Image.Image,
    model,
    conf_threshold: float = 0.25
) -> Tuple[np.ndarray, list]:
    bgr = pil_to_bgr(image)
    results = model.predict(bgr, conf=conf_threshold, verbose=False)

    plotted = results[0].plot()
    plotted_rgb = bgr_to_rgb(plotted)

    detections = []
    boxes = results[0].boxes
    names = model.names

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append({
                "class": names[cls_id],
                "confidence": round(conf, 4)
            })

    return plotted_rgb, detections

# ==================================
# SIDEBAR
# ==================================
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider(
    "YOLO confidence threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05
)

# ==================================
# LOAD RESOURCES
# ==================================
yolo_error = None
caption_error = None

try:
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
except Exception as e:
    yolo_model = None
    yolo_error = str(e)

try:
    caption_assets = load_caption_assets(CAPTION_MODEL_PATH, VOCAB_PATH)
except Exception as e:
    caption_assets = None
    caption_error = str(e)

if yolo_error:
    st.error(yolo_error)

if caption_error:
    st.error(caption_error)

# ==================================
# FILE UPLOAD
# ==================================
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=False)

    if st.button("Predict", type="primary"):
        col1, col2 = st.columns(2)

        with st.spinner("Running both models..."):
            if caption_assets is not None:
                caption_text = generate_caption(image, caption_assets)
            else:
                caption_text = "Caption model not loaded."

            if yolo_model is not None:
                detected_image, detections = run_yolo_detection(
                    image=image,
                    model=yolo_model,
                    conf_threshold=conf_threshold,
                )
            else:
                detected_image, detections = None, []

        with col1:
            st.subheader("Image Captioning Output")
            st.success(caption_text)

        with col2:
            st.subheader("YOLO Detection Output")
            if detected_image is not None:
                st.image(detected_image, use_container_width=True)
            else:
                st.warning("YOLO model not loaded.")

        st.subheader("Detected Objects")
        if detections:
            st.dataframe(detections, use_container_width=True)
        else:
            st.write("No detections found.")

st.markdown("---")