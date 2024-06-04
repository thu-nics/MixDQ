from diffusers import StableDiffusionXLPipeline, LCMScheduler
from pytorch_lightning import seed_everything
import torch

# pipeline_text2image = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# pipeline_text2image = pipeline_text2image.to("cuda")
seed_everything(42)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

pipe = StableDiffusionXLPipeline.from_pretrained(model_id, cache_dir="/share/public/diffusion_quant/huggingface/hub/")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# load and fuse lcm lora
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

model = pipe.unet
# 获取模型结构的字符串表示
model_str = ""
for name, module in model.named_modules():
    model_str += f"{name}: {type(module)}\n"

# 将模型结构写入到 txt 文件中
with open('/home/fangtongcheng/diffuser-dev/analysis_tools/model_arch/LCM_LoRA_SDXL.txt', 'w') as f:
    f.write(model_str)


# 你的代码是在创建一个UNet2DModel的实例，并通过named_modules()方法获取模型中所有模块的名称和类型。named_modules()方法会返回一个迭代器，
# 包含模型中所有模块的名称（一个字符串）和模块本身（一个nn.Module实例）。

