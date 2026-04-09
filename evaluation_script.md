import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import cv2
import torchvision.transforms as transforms
import warnings

# Install lpips if not already installed
!pip install lpips
import lpips

# Ignore warnings to keep the output clean
warnings.filterwarnings("ignore")

print("Loading models, please wait...")

# ==========================================
# 1. Load OpenAI's CLIP model (using the base ViT-B/32 version)
# ==========================================
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id)

# ==========================================
# 2. Load the LPIPS perceptual distance model
# (The current academic standard for image difference evaluation)
# ==========================================
loss_fn_alex = lpips.LPIPS(net='alex')

# ==========================================
# 3. Load your images
# ==========================================
style_path = "style.jpg"         # Your style exemplar (Baseline)
generated_path = "generated.jpg"   # Your AI-generated iteration

img_style = Image.open(style_path).convert('RGB')
img_gen = Image.open(generated_path).convert('RGB')

# ==========================================
# Compute CLIP Image-to-Image Similarity
# Purpose: Evaluates how similar the two images are in terms of semantics and overall style.
# ==========================================
inputs = processor(images=[img_style, img_gen], return_tensors="pt")
with torch.no_grad():
    # Use the vision model directly to get features
    # This avoids the error where the main model expects text input_ids
    vision_outputs = clip_model.vision_model(pixel_values=inputs['pixel_values'])
    image_features = clip_model.visual_projection(vision_outputs[1])

# Calculate Cosine Similarity
features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
clip_similarity = torch.nn.functional.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
clip_score = clip_similarity.item()

# ==========================================
# Compute SSIM (Structural Similarity Index)
# Purpose: Evaluates the physical differences in pixel structure and luminance.
# ==========================================
cv_style = cv2.imread(style_path, cv2.IMREAD_GRAYSCALE)
cv_gen = cv2.imread(generated_path, cv2.IMREAD_GRAYSCALE)

# Force resize the generated image to match the reference image size for comparison
cv_gen_resized = cv2.resize(cv_gen, (cv_style.shape[1], cv_style.shape[0]))
ssim_score, _ = ssim(cv_style, cv_gen_resized, full=True)

# ==========================================
# Compute LPIPS (Learned Perceptual Image Patch Similarity)
# Purpose: Simulates the perceptual difference as seen by the human eye.
# (Note: Lower value is better/closer)
# ==========================================
# Convert images to the tensor format required by LPIPS
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

tensor_style = transform(img_style).unsqueeze(0)
tensor_gen = transform(img_gen).unsqueeze(0)

with torch.no_grad():
    lpips_score = loss_fn_alex(tensor_style, tensor_gen).item()

# ==========================================
# Print Final Report
# ==========================================
print("\n" + "="*50)
print("🎯 SADM Framework Visual Similarity Evaluation Report")
print("="*50)
print(f"1. CLIP Similarity Score (ViT-B/32): {clip_score:.4f}")
print("   (Note: Range 0~1. Higher means closer semantic style.")
print("          Typically >0.75 indicates high similarity.)")
print("-" * 50)
print(f"2. SSIM Structural Similarity: {ssim_score:.4f}")
print("   (Note: Range 0~1. Higher means closer pixel-level structure.")
print("          Expected to be relatively low for dynamic posters.)")
print("-" * 50)
print(f"3. LPIPS Perceptual Distance (AlexNet): {lpips_score:.4f}")
print("   (Note: Range 0~1. *LOWER* is better, indicating less perceptual")
print("          dissonance to the human eye. Typically <0.4 is good.)")
print("="*50)