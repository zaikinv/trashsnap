import os
import torch
from PIL import Image, ImageOps
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_processor():
    model = CLIPModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False)
    model = model.to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor

def check_required_files(filenames):
    missing = [f for f in filenames if not os.path.isfile(f)]
    return missing

def load_labels():
    with open("labels_EN.txt", encoding="utf-8") as f:
        labels_en_raw = f.read().splitlines()
    with open("labels_DE.txt", encoding="utf-8") as f:
        labels_de = f.read().splitlines()
    with open("answers_DE.txt", encoding="utf-8") as f:
        answers_de = f.read().splitlines()
    labels_en = [f"An image containing {x}" for x in labels_en_raw]
    return labels_en_raw, labels_en, labels_de, answers_de

def embed_text(texts, model, processor, chunk=128):
    parts, out = [texts[i:i+chunk] for i in range(0, len(texts), chunk)], []
    with torch.no_grad():
        for text_chunk in parts:
            text_inputs = processor(text=text_chunk, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}
            embeddings = model.get_text_features(**text_inputs)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            out.append(embeddings)
    feats = torch.cat(out)
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats

def classify(image, model, processor, text_feats):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt")
    image_inputs = {k: v.to(DEVICE) for k, v in image_inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity_scores = (image_features @ text_feats.T)[0]
    return similarity_scores