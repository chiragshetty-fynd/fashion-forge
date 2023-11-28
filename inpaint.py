import torch
from diffusers import StableDiffusionXLInpaintPipeline

pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    # "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "stabilityai/stable-diffusion-xl-base-1.0"
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Extra params
pipeline.enable_vae_slicing()
pipeline.enable_freeu(s1=0.65, s2=0.45, b1=1.16, b2=1.26)

generator = torch.Generator(device="cuda").manual_seed(0)
