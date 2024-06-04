from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLPipeline, StableDiffusionPipeline
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image, UNet2DConditionModel, StableDiffusionPipeline, DDIMScheduler
from pytorch_lightning import seed_everything
from sdxl_pipeline import StableDiffusionXLPipeline
import time
import json

def prepare_coco_text_and_image(json_file):
    info = json.load(open(json_file, 'r'))
    annotation_list = info["annotations"]
    image_caption_dict = {}
    for annotation_dict in annotation_list:
        if annotation_dict["image_id"] in image_caption_dict.keys():
            image_caption_dict[annotation_dict["image_id"]].append(annotation_dict["caption"])
        else:
            image_caption_dict[annotation_dict["image_id"]] = [annotation_dict["caption"]]
    captions = list(image_caption_dict.values())
    image_ids = list(image_caption_dict.keys())

    active_captions = []
    for texts in captions:
        active_captions.append(texts[0])

    image_paths = []
    for image_id in image_ids:
        image_paths.append("/share/public/diffusion_quant/coco/coco/val2014/"+f"COCO_val2014_{image_id:012}.jpg")
    return active_captions, image_paths


if __name__ =="__main__":
    seed_everything(42)
    pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo", resume_download = True, cache_dir="/share/public/diffusion_quant/huggingface/hub")
    pipeline = pipeline.to("cuda")

    json_file = "/share/public/diffusion_quant/coco/coco/annotations/captions_val2014.json"
    prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
    input_list = []
    output_fp32_list = []
    total_num = 32
    for i in range(int(total_num//8)):
        prompts = prompt_list[i*8:(i+1)*8]
        image_container, input_unet, output_fp32_unet = pipeline(prompt=prompts, guidance_scale=0, num_inference_steps=1)
        input_list.append(input_unet)
        output_fp32_list.append(output_fp32_unet)
        image = image_container.images[0]
    image.save("sdxl_text.png")
    torch.save(input_list, '../error_func/unet_in_out/bs32_input_sdxl_turbo')
    torch.save(output_fp32_list, '../error_func/unet_in_out/bs32_output_sdxl_turbo')
