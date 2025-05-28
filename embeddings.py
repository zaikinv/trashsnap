from core import load_model_and_processor, load_labels, embed_text
import torch

model, processor = load_model_and_processor()
_, labels_en, _, _ = load_labels()

text_feats = embed_text(labels_en, model, processor)
torch.save(text_feats, "text_embeddings.pt")