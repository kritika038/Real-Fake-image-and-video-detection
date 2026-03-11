import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline

# ==============================
# CONFIG
# ==============================

TOTAL_IMAGES = 1200
OUTPUT_DIR = Path("data/train/fake_diffusion")
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Diverse prompts for robustness
prompts = [
    "Ultra realistic DSLR portrait of a 35 year old man, studio lighting",
    "Candid street photography of a woman walking in Tokyo at night",
    "Wildlife photo of a tiger in jungle, National Geographic style",
    "Professional food photography of gourmet burger",
    "Architectural photography of modern skyscraper",
    "Close up macro photography of human eye",
    "Wedding photography of bride and groom",
    "Travel photography of mountains in Switzerland",
    "Photojournalism style war scene",
    "Fashion magazine shoot, high detail skin texture",
    "Realistic selfie taken from iPhone front camera",
    "Group photo at birthday party indoor lighting",
    "Police bodycam footage style photo",
    "Sports photography of football match action shot",
    "Old vintage family photograph from 1990s"
]

# ==============================
# DEVICE SETUP
# ==============================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ==============================
# LOAD PIPELINE
# ==============================

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "mps" else torch.float32
)

pipe = pipe.to(device)

# Disable safety checker for dataset creation
pipe.safety_checker = None

# ==============================
# OUTPUT FOLDER
# ==============================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

existing_images = len(list(OUTPUT_DIR.glob("*.jpg")))
print(f"Resuming from image index: {existing_images}")

# ==============================
# GENERATION LOOP
# ==============================

try:
    for i in range(existing_images, TOTAL_IMAGES):

        prompt = prompts[i % len(prompts)]

        image = pipe(
            prompt,
            num_inference_steps=20,   # balance speed & quality
            guidance_scale=7.5
        ).images[0]

        save_path = OUTPUT_DIR / f"diff_{i}.jpg"
        image.save(save_path)

        if i % 50 == 0:
            print(f"Generated {i} / {TOTAL_IMAGES}")

    print("\nGeneration complete.")

except KeyboardInterrupt:
    print("\nStopped safely. You can resume later.")