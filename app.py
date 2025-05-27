import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-large-patch14"
model      = CLIPModel.from_pretrained(model_name).to(device)
processor  = CLIPProcessor.from_pretrained(model_name)

# load label files
with open("labels_EN.txt", encoding="utf-8") as f: labels_en_raw = f.read().splitlines()
with open("labels_DE.txt", encoding="utf-8") as f: labels_de     = f.read().splitlines()
with open("answers_DE.txt", encoding="utf-8") as f: answers_de   = f.read().splitlines()

labels_en = [f"An image containing {x}" for x in labels_en_raw]

# one-time text-embedding cache
def embed_text(texts, chunk=128):
    parts, out = [texts[i:i+chunk] for i in range(0, len(texts), chunk)], []
    with torch.no_grad():
        for p in parts:
            t = processor(text=p, return_tensors="pt", padding=True, truncation=True).to(device)
            e = model.get_text_features(**t)
            e /= e.norm(dim=-1, keepdim=True)
            out.append(e)
    feats = torch.cat(out)
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats

text_feats = embed_text(labels_en)

def classify(img: Image.Image) -> str:
    inp = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        v = model.get_image_features(**inp)
        v /= v.norm(dim=-1, keepdim=True)
        idx = (v @ text_feats.T)[0].argmax().item()
    return f"{labels_en_raw[idx]} ➜ {labels_de[idx]} ➜ {answers_de[idx]}"

gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🧠 TrashSnap",
    description="Drop or snap a picture of waste and get the right German bin."
).launch()
