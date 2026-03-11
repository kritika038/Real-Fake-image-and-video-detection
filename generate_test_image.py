from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A realistic portrait photo of a young woman"

image = pipe(prompt).images[0]
image.save("generated_ai_image.jpg")

print("Saved generated_ai_image.jpg")
