import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load model safely
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

# Load image and text
image = Image.open("datasets_folder/image-caption/Images/667626_18933d713e.jpg")
caption = "A young girl is lying in the sand , while ocean water is surrounding her ."  # example caption

# Process inputs
inputs = processor(text=[caption], images=[image], return_tensors="pt", padding=True)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

# Normalize embeddings (very important!)
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Compute cosine similarity
similarity = (text_embeds @ image_embeds.T).item()

print(f"Caption–Image Similarity: {similarity:.4f}")

# Decision logic
if similarity > 0.3:  # threshold can be tuned (0.25–0.35 is typical)
    print(" Caption matches the image (consistent)")
else:
    print(" Caption does NOT match the image (inconsistent)")
