import streamlit as st
import pandas as pd
from PIL import Image
from os import path
import torch
from core import (
    DEVICE,
    load_model_and_processor,
    check_required_files,
    load_labels,
    embed_text,
    classify,
    MODEL_NAME,
)
from utils import profile_resources, show_system_info
st.set_page_config(page_title="TrashSnap", layout="centered")
st.title("🧠 TrashSnap")
st.markdown("Drop or upload a picture of waste to get the correct Tonne")

show_system_info()

# ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_model_and_processor():
    return load_model_and_processor()

st.info("⏳ Loading model...")
try:
    model, processor = get_model_and_processor()
    st.success(f"✅ Model {MODEL_NAME} and processor loaded!")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────
# st.info("📄 Checking label files...")
required_files = ["data/labels_EN.txt", "data/labels_DE.txt", "data/answers_DE.txt"]
missing = check_required_files(required_files)
if missing:
    for f in missing:
        st.error(f"❌ Missing file: `{f}`")
    st.stop()

# ──────────────────────────────────────────────────────────────
try:
    labels_en_raw, labels_en, labels_de, answers_de = load_labels()
except Exception as e:
    st.error(f"Error loading label files: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def get_embeddings():
    if not path.isfile("text_embeddings.pt"):
        st.warning("Generating text embeddings...")
        text_feats = embed_text(labels_en, model, processor)
        torch.save(text_feats, "text_embeddings.pt")
    else:
        st.warning("Found cached embeddings!")
        text_feats = torch.load("text_embeddings.pt").to(DEVICE)
    return text_feats

text_feats = get_embeddings()
st.success("✅ Text embeddings ready!")

# ──────────────────────────────────────────────────────────────

photo = st.camera_input("Take a photo of your trash")

if photo:
    image = Image.open(photo)
    profiled_classify = profile_resources(classify)
    sim = profiled_classify(image, model, processor, text_feats)
    idx = sim.argmax().item()

    st.success(f"🗑️ {labels_en_raw[idx]} ➜ {labels_de[idx]} ➜ **{answers_de[idx]}**")

    # Информация о модели для результата
    st.markdown(f"*🤖 Classified using: {MODEL_NAME}*")

    st.markdown("🏷️ Top-10:")
    topk = sim.topk(10)
    
    # Create DataFrame for top-10 results
    top_indices = [topk.indices[i].item() for i in range(10)]
    df = pd.DataFrame({
        "Rank": range(1, 11),
        "Label (EN)": [labels_en_raw[k] for k in top_indices],
        "Label (DE)": [labels_de[k] for k in top_indices],
        "Disposal Method": [answers_de[k] for k in top_indices],
        "Similarity": [f"{topk.values[i].item():.4f}" for i in range(10)]
    })
    
    st.dataframe(df, use_container_width=True)