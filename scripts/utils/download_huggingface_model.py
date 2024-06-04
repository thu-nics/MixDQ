# the code for generating the `local` diffusers pipeline
# 1st load the pipeline using automatic download, then use `save_pretrained` to generate local file
from diffusers import StableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"


cache_dir = "/share/public/diffusion_quant/huggingface/hub/"


save_path = "/share/public/diffusion_quant/huggingface/sdxl-base-1.0"  # change accordingly

pipe = StableDiffusionXLPipeline.from_pretrained(model_id, cache_dir = cache_dir)
pipe.save_pretrained(save_path)

