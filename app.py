import os
import torch
import streamlit as st
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="TrashSnap", layout="centered")
st.title("🧠 TrashSnap")
st.markdown("Drop or upload a picture of waste to get the correct German bin 🚮")

# ──────────────────────────────────────────────────────────────
# Init
st.info("⏳ Loading model and label files...")
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: `{device}`")

model_name = "openai/clip-vit-large-patch14"

try:
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    st.success("Model and processor loaded ✅")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

required_files = ["labels_EN.txt", "labels_DE.txt", "answers_DE.txt"]
for file in required_files:
    if not os.path.isfile(file):
        st.error(f"❌ Missing file: `{file}`")
        st.stop()
    else:
        st.write(f"✅ Found `{file}`")

# ──────────────────────────────────────────────────────────────
# Load label data
try:
    with open("labels_EN.txt", encoding="utf-8") as f:
        labels_en_raw = f.read().splitlines()
    with open("labels_DE.txt", encoding="utf-8") as f:
        labels_de = f.read().splitlines()
    with open("answers_DE.txt", encoding="utf-8") as f:
        answers_de = f.read().splitlines()
    st.success(f"Loaded {len(labels_en_raw)} labels")
except Exception as e:
    st.error(f"Error loading label files: {e}")
    st.stop()

labels_en = [f"An image containing {x}" for x in labels_en_raw]

# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=True)
def embed_text(texts, chunk=128):
    st.info("🔠 Embedding label texts...")
    parts, out = [texts[i:i+chunk] for i in range(0, len(texts), chunk)], []
    with torch.no_grad():
        for i, p in enumerate(parts):
            st.write(f"Chunk {i+1}/{len(parts)}")
            t = processor(text=p, return_tensors="pt", padding=True, truncation=True).to(device)
            e = model.get_text_features(**t)
            e /= e.norm(dim=-1, keepdim=True)
            out.append(e)
    feats = torch.cat(out)
    feats /= feats.norm(dim=-1, keepdim=True)
    st.success("✅ Text embeddings ready")
    return feats

text_feats = embed_text(labels_en)

# ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload or drag a photo", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        image = ImageOps.exif_transpose(image)
        st.image(image, caption="Uploaded image", width=336)

        st.info("🧠 Classifying...")
        inp = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            v = model.get_image_features(**inp)
            v /= v.norm(dim=-1, keepdim=True)
            sim = (v @ text_feats.T)[0]
            idx = sim.argmax().item()

        st.success(f"🗑️ {labels_en_raw[idx]} ➜ {labels_de[idx]} ➜ **{answers_de[idx]}**")

        st.markdown("🏷️ Top-10:")
        topk = sim.topk(10)
        for i in range(10):
            k = topk.indices[i].item()
            st.markdown(f"`{labels_en_raw[k]}` ➜ `{labels_de[k]}` ➜ **{answers_de[k]}** — `{topk.values[i].item():.4f}`")

    except Exception as e:
        st.error(f"❌ Error: {e}")
