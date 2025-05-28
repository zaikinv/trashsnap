import streamlit as st
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
)
from utils import profile_resources
st.set_page_config(page_title="TrashSnap", layout="centered")
st.title("🧠 TrashSnap")
st.markdown("Drop or upload a picture of waste to get the correct Tonne")

# ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_model_and_processor():
    return load_model_and_processor()

st.info("⏳ Loading model...")
try:
    model, processor = get_model_and_processor()
    st.success("✅ Model and processor loaded!")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ──────────────────────────────────────────────────────────────
st.info("📄 Checking label files...")
required_files = ["labels_EN.txt", "labels_DE.txt", "answers_DE.txt"]
missing = check_required_files(required_files)
if missing:
    for f in missing:
        st.error(f"❌ Missing file: `{f}`")
    st.stop()
else:
    for f in required_files:
        st.write(f"✅ Found `{f}`")

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
    st.image(image, caption="Сделанное фото", width=336)
    st.info("🧠 Classifying...")
    profiled_classify = profile_resources(classify)
    sim = profiled_classify(image, model, processor, text_feats)
    idx = sim.argmax().item()

    st.success(f"🗑️ {labels_en_raw[idx]} ➜ {labels_de[idx]} ➜ **{answers_de[idx]}**")

    st.markdown("🏷️ Top-10:")
    topk = sim.topk(10)
    for i in range(10):
        k = topk.indices[i].item()
        st.markdown(f"`{labels_en_raw[k]}` ➜ `{labels_de[k]}` ➜ **{answers_de[k]}** — `{topk.values[i].item():.4f}`")

    st.success(f"🗑️ {labels_en_raw[idx]} ➜ {labels_de[idx]} ➜ **{answers_de[idx]}**")