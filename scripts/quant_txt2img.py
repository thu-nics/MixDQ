import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch.cuda import amp
from contextlib import nullcontext
import json
import sys

from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params, prepare_coco_text_and_image

from tqdm.auto import tqdm
import sys


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="how many batches to produce for each given prompt. A.k.a. batch size",
    )

    parser.add_argument(
        "--cfg",
        type=float,
        default=None,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="path to config which constructs model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="path for generated images, leave empty to automatically read from base_path",
    )
    parser.add_argument(
        "--num_imgs",
        type=int,
        default=32,
        help="the number of the output images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--fp16",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_act",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_weight",
        action='store_true',
    )
    parser.add_argument(
        "--config_weight_mp",
        type=str,
        help="path for weight configs",
    )
    parser.add_argument(
        "--config_act_mp",
        type=str,
        help="path for act configs",
    )
    parser.add_argument(
        "--act_protect",
        type=str,
        help="the path for extremely sensitive acts",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    opt.outdir = os.path.join(opt.base_path,'generated_images')
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    log_path = os.path.join(opt.base_path, "run.log")
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

    if opt.cfg is None:
        opt.cfg = config.calib_data.scale_value

    # adapter_id = getattr(config.model, "adapter_id", None)
    # adapter_cache_dir = getattr(config.model, "adapter_cache_dir", None)
    model, pipe = get_model(config.model, fp16=opt.fp16, return_pipe=True)
    num_timesteps = config.calib_data.n_steps

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is False else True
    # use_act_quant = False if aq_params is False else True
    use_weight_quant = not opt.skip_quant_weight
    use_act_quant = not opt.skip_quant_act

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        aq_params['mixed_precision'] = config.mixed_precision

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    dtype = torch.float32 if not opt.fp16 else torch.float16
    if opt.fp16:
        qnn = qnn.half()  # make some newly genrated quant-related modules FP16

    qnn.set_quant_state(False, False)
    calib_added_cond = {}
    calib_added_cond["text_embeds"] = torch.randn(1, 1280, dtype=dtype).cuda().to(dtype)
    calib_added_cond["time_ids"] = torch.randn(1, 6, dtype=dtype).cuda().to(dtype)

    with torch.no_grad():
        if config.model.model_type == "sdxl":
            _ = qnn(torch.randn(1, 4, 64, 64).cuda().to(dtype), \
                    torch.randint(0, 1000, (1,)).cuda().to(dtype), \
                    torch.randn(1, 77, 2048).cuda().to(dtype), \
                    added_cond_kwargs=calib_added_cond)
        elif config.model.model_type == "sd":
            _ = qnn(torch.randn(1, 4, 64, 64).cuda().to(dtype), \
                    torch.randint(0, 1000, (1,)).cuda().to(dtype), \
                    torch.randn(1, 77, 768).cuda().to(dtype))

    # set the init flag True, otherwise will recalculate params
    qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    load_quant_params(qnn, opt.ckpt, dtype=dtype)

    # Forward
    if opt.prompt is None:
        json_file = "./scripts/utils/captions_val2014.json"
        prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
        prompts = prompt_list[0:opt.num_imgs]
    else:
        prompts = [opt.prompt]*opt.num_imgs


    use_weight_mp = opt.config_weight_mp is not None
    use_act_mp = opt.config_act_mp is not None

    # inference with the quantized model with 
    if use_weight_mp:
        with open(opt.config_weight_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("---------------- load the bitwidth config for weight! -------------------")
        logger.info(f"------------------ config: {opt.config_weight_mp} ---------------------")

        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

        if use_weight_mp and not use_act_mp:
            logger.info("-------- Inference with weight-only quantized -----------")
            gen_image(prompts, qnn, pipe, num_timesteps, opt)
            return None
    if use_act_mp:
        # protect the extremely sensitive layer
        acts_protected = torch.load(opt.act_protect)
        qnn.set_layer_quant(model=qnn, module_name_list=acts_protected, quant_level='per_layer', weight_quant=True, act_quant=False)

        with open(opt.config_weight_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("load the bitwidth config for weight!")
        logger.info(f"the config for weight is {opt.config_weight_mp}")
        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

        with open(opt.config_act_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("------------- load the bitwidth config for act! \nInference with w&a quantized! ------------")
        logger.info(f"------------ config: {opt.config_act_mp} -----------------")

        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='act')
        gen_image(prompts, qnn, pipe, num_timesteps, opt)
        return None
    else:
        logger.info("Inference without mixed precision!")
        gen_image(prompts, qnn, pipe, num_timesteps, opt)
        return None


def gen_image(prompt, unet, pipe, num_timesteps, opt):
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(opt.seed)  # Seed generator to create the initial latent noise
    total = len(prompt)
    batch_size = opt.batch_size
    assert(total >= batch_size), "the length of prompts should larger than batch_size"
    num = total // batch_size

    img_id = 0
    logger.info(f"starting from image {img_id}")

    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    pipe.unet = unet.half() if opt.fp16 else unet
    pipe.to("cuda")
    with torch.no_grad():
        for i in tqdm(
            range(num), desc="Generating image samples for FID evaluation."
        ):
            with amp.autocast(enabled=False):

                image = pipe(prompt=prompt[img_id:img_id+batch_size], num_inference_steps=num_timesteps, guidance_scale=opt.cfg).images

            for j in range(batch_size):
                image[j].save(f"{opt.image_folder}/{img_id}.png")
                img_id += 1

if __name__ == "__main__":
    main()


