from core import load_model_and_processor, load_labels, embed_text
import torch

print("Loading Xenova CLIP ONNX model...")
model, processor = load_model_and_processor()
print("Loading labels...")
_, labels_en, _, _ = load_labels()

print(f"Generating embeddings for {len(labels_en)} labels...")
text_feats = embed_text(labels_en, model, processor)
torch.save(text_feats, "text_embeddings.pt")
print("✅ Text embeddings saved to text_embeddings.pt")