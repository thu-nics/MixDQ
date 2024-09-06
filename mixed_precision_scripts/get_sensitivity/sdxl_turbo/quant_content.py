import argparse, os
import logging
import cv2
import math
from skimage import io
from skimage import metrics
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch.cuda import amp
import json
import sys
import yaml

from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params

from PIL import Image
from tqdm.auto import tqdm

import sys


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     nargs="?",
    #     default="a painting of a virus monster playing guitar",
    #     help="the prompt to render"
    # )
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    # parser.add_argument(
    #     "--n_samples",
    #     type=int,
    #     default=3,
    #     help="how many samples to produce for each given prompt. A.k.a. batch size",
    # )
    # parser.add_argument(
    #     "--scale",
    #     type=float,
    #     default=7.5,
    #     help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    # )
    parser.add_argument(
        "--config",
        type=str,
        # default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        # default="/root/qdiffusion/q-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="path for generated images",
    )
    parser.add_argument(
        "--sensitivity_type",
        type=str,
        help="weight/act",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--skip_quant_act",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_weight",
        action='store_true',
    )
    # parser.add_argument(
    #     "--template_config", required=True,
    #     type=str,
    #     help="a template to init a sensitivity config",
    # )
    parser.add_argument(
        "--reference_img", required=True,
        type=str,
        help="the path to store reference imgs",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    opt.outdir = os.path.join(opt.base_path,'generated_images')
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # load the config from the log path
    if opt.config is None:
        opt.config = os.path.join(opt.base_path,'config.yaml')
    if opt.ckpt is None:
        opt.ckpt = os.path.join(opt.base_path,'ckpt.pth')
    if opt.image_folder is None:
        opt.image_folder = os.path.join(opt.base_path,'generated_images')
    config = OmegaConf.load(f"{opt.config}")
    # model = load_model_from_config(config, ckpt=None, cfg_type='diffusers')
    model, pipe = get_model(config.model, fp16=False, return_pipe=True, convert_model_for_quant=True)

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is False else True
    # use_act_quant = False if aq_params is False else True
    use_weight_quant = not opt.skip_quant_weight
    use_act_quant = not opt.skip_quant_act


    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        # act_quant_mode="qdiff",\
        # sm_abit=config.quant.softmax.n_bits,\
    )
    qnn.cuda()
    qnn.eval()

    qnn.set_quant_state(False, False)
    calib_added_cond = {}
    calib_added_cond["text_embeds"] = torch.randn(1, 1280).cuda()
    calib_added_cond["time_ids"] = torch.randn(1, 6).cuda()
    # calib_data_placeholder = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 2048), calib_added_cond_kwargs)  # assign empty calib_data placeholder
    with torch.no_grad():
        _ = qnn(torch.randn(1, 4, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 77, 2048).cuda(), added_cond_kwargs=calib_added_cond)
    
    json_file = "/share/public/diffusion_quant/coco/coco/annotations/captions_val2014.json"
    prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
    prompts = prompt_list[0:32] # use the first 32 prompts for coco

    # generate the reference FP16 images
    os.makedirs(opt.reference_img, exist_ok=True)
    with torch.no_grad():
        sample_fid(prompts, qnn, pipe, opt, batch_size=32, quant_inference = False, is_fp16= True)

    # set the init flag True, otherwise will recalculate params
    qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')
    load_quant_params(qnn, opt.ckpt)

    # content_layers = []
    # for n,m in model.named_modules():
    #     if 'attn2' in n or 'ff' in n:
    #         if isinstance(m, QuantLayer):
    #             content_layers.append('model.'+n)
    # config_ssim_layer = {}
    # for layer_name in content_layers:
    #     config_ssim_layer[layer_name] = []

    config_ssim_layer = {}  # leave empty

    if opt.sensitivity_type == 'act':

        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width} \n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

            qnn.set_quant_state(False, False)
            logger.info("------- start to compute the SSIM related to quant_layers -----")
            SSIM_Layer(model=qnn, qnn=qnn, pipe=pipe, opt=opt, prompts=prompts, weight_quant=False, act_quant=True, weight_only=False, config_ssim_layer=config_ssim_layer, cur_bit=bit_width)


    if opt.sensitivity_type == 'weight':
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')
            qnn.set_quant_state(False, False)
            logger.info("-------- start to compute the SSIM related to quant_layers -------")
            SSIM_Layer(model=qnn, qnn=qnn, pipe=pipe, opt=opt, prompts=prompts, weight_quant=True, act_quant=False, weight_only=True, config_ssim_layer=config_ssim_layer, cur_bit=bit_width)


    save_path_config = opt.base_path+'/sensitivity_{}_content.yaml'.format(opt.sensitivity_type)
    with open(save_path_config, 'w') as file:
        yaml.dump(config_ssim_layer, file)


def SSIM_Layer(model, qnn, pipe, opt, prompts, weight_quant=True, act_quant=False, weight_only=True, progressivly=False, config_ssim_layer={}, cur_bit=0, prefix=""):
    # Forward
    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {)}")
        if isinstance(module, QuantLayer):
            if 'ff' in full_name or 'attn2' in full_name:
            # if 'model.down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v' in full_name:
                # set_quant_state(module, weight_quant=weight_quant, act_quant=act_quant)
                module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                with torch.no_grad():
                    sample_fid(prompts, qnn, pipe, opt, batch_size=32, quant_inference = True, is_fp16= False)

                ssim = SSIM(img_path1=opt.image_folder, 
                            img_path2=opt.reference_img, 
                            bs=32)

                logger.info('SSIM:{:.5f}\n'.format(float(ssim)))

                # config_ssim_layer[full_name][int(math.log2(cur_bit)-1)] = float(ssim)
                if not full_name in config_ssim_layer.keys():
                    config_ssim_layer[full_name] = []
                config_ssim_layer[full_name].append(float(ssim))

                if weight_only:
                    if not progressivly:
                        qnn.set_quant_state(False, False)
                else:
                    if not progressivly:
                        qnn.set_quant_state(False, False)

        else:
            SSIM_Layer(model=module, qnn=qnn, pipe=pipe, opt=opt, prompts=prompts, weight_quant=weight_quant, act_quant=act_quant, weight_only=weight_only, progressivly=False, config_ssim_layer=config_ssim_layer, cur_bit=cur_bit, prefix=full_name+".")


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


def inference_sdxl_turbo(prompt, pipe):
    # prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0).images
    return image


def sample_fid(prompt, unet, pipe, opt, batch_size, quant_inference = False, is_fp16 = False):

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(opt.seed)  # Seed generator to create the initial latent noise
    total = len(prompt)
    # n = 16  # 按批量推理
    num = total // batch_size
    # num = opt.n_samples # DEBUG_ONLY: generate batch_size*n_sample images only
    img_id = 0
    # logger.info(f"starting from image {img_id}")
    

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    if quant_inference and is_fp16:
        pipe.unet = unet.half()  # 暂时debug for wxa16

    pipe.to("cuda")
    with torch.no_grad():
        for i in tqdm(
            range(num), desc="Generating image samples for FID evaluation."
        ):
            with amp.autocast(enabled=False):
                image = inference_sdxl_turbo(prompt[batch_size*i:batch_size*(i+1)], pipe)

            for j in range(batch_size):
                if quant_inference:
                    path_ = opt.image_folder
                else: # generate for FP16 reference
                    path_ = opt.reference_img
                image[j].save(f"{path_}/{img_id}.png")
                img_id += 1


def SSIM(img_path1, img_path2, bs=32):
    ssim_index_mean = 0
    for i in range(bs):
        img1 = (cv2.imread(img_path1+f'/{i}.png'))
        img2 = (cv2.imread(img_path2+f'/{i}.png'))
        ssim_index = metrics.structural_similarity(img1, img2, multichannel=True, channel_axis=2, win_size=511)
        ssim_index_mean = ssim_index_mean + ssim_index
        # print(ssim_index)
    ssim_index_mean = ssim_index_mean / bs
    return ssim_index_mean


def MSE_pixel(img_path1, img_path2, bs=32):
    mse_mean = 0
    for i in range(bs):
        img1 = (cv2.imread(img_path1+f'/{i}.png'))
        img2 = (cv2.imread(img_path2+f'/{i}.png'))
        # 计算像素域MSE
        mse = np.mean((img1-img2)**2)
        mse_mean = mse_mean + mse
    mse_mean = mse_mean / bs
    return mse_mean


if __name__ == "__main__":
    main()


