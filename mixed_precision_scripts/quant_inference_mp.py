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
from qdiff.utils import get_model, load_quant_params

from PIL import Image
from tqdm.auto import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, UniPCMultistepScheduler

# setup debug ipdb hook when exception occurs
# from qdiff.utils import custom_excepthook
import sys
# sys.excepthook = custom_excepthook


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--config_weight_mp",
        type=str,
        nargs="?",
        help="dir to load the mixed precision config of the wieght",
    )
    parser.add_argument(
        "--config_act_mp",
        type=str,
        nargs="?",
        help="dir to load the mixed precision config of the act",
    )
    parser.add_argument(
        "--dir_weight_mp",
        type=str,
        nargs="?",
        help="dir to load the mixed precision config of the wieght",
    )
    parser.add_argument(
        "--dir_act_mp",
        type=str,
        nargs="?",
        help="dir to load the mixed precision config of the act",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
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
        "--reference_img",
        type=str,
        help="path for reference FP16 generated images",
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
    parser.add_argument(
        "--use_weight_mp",
        action='store_true',
    )
    parser.add_argument(
        "--use_act_mp",
        action='store_true',
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
    model, pipe = get_model(config.model, fp16=False, return_pipe=True, convert_model_for_quant=True)

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is False else True
    # use_act_quant = False if aq_params is False else True
    use_weight_quant = not opt.skip_quant_weight
    use_act_quant = not opt.skip_quant_act

    wq_params['mixed_precision'] = config.mixed_precision
    aq_params['mixed_precision'] = config.mixed_precision

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


    # set the init flag True, otherwise will recalculate params
    qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    load_quant_params(qnn, opt.ckpt)


    json_file = "/share/public/diffusion_quant/coco/coco/annotations/captions_val2014.json"
    prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
    prompts = prompt_list[0:1]

    mse_min = float("inf")
    final_config = {}
    # ------------ weight only ------------------
    if opt.use_weight_mp:
        for filename, config in load_yaml_files(opt.dir_weight_mp):
            bit_config = config
            logger.info("-------- load the bitwidth config for weight! \nInference with weight quantized only! ------")
            logger.info(f"-------- config: {filename} -------")

            qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

            if opt.use_weight_mp and not opt.use_act_mp:
                # opt.image_folder = os.path.join(opt.base_path,f'images_{filename}')
                os.makedirs(opt.image_folder, exist_ok=True)
                # prompts = ["portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"]
                sample_fid(prompts, qnn, pipe, opt, batch_size=1, quant_inference = True, is_fp16 = False, save_path=filename)
                mse_tmp = MSE_pixel(
                        f"{opt.image_folder}/{filename}.png",
                        os.path.join(opt.reference_img,'0.png')
                    )
                if mse_tmp < mse_min:
                    mse_min = mse_tmp
                    final_config = bit_config
                    final_name = filename
        write_yaml_file(os.path.join(opt.base_path, "final_weight_mp.yaml"), final_config)
        qnn.load_bitwidth_config(model=qnn, bit_config=final_config, bit_type='weight')
        sample_fid(prompts, qnn, pipe, opt, batch_size=1, quant_inference = True, is_fp16 = False, save_path='final')


    #################################################### wxa8 only ####################################################
    if opt.use_act_mp:
        acts_protected = torch.load("./mixed_precision_scripts/mixed_percision_config/sdxl_turbo/final_config/act/act_sensitivie_a8_1%.pt")
        
        qnn.set_layer_quant(model=qnn, module_name_list=acts_protected, quant_level='per_layer', weight_quant=True, act_quant=False)

        with open(opt.config_weight_mp, 'r') as file:
            bit_config = yaml.safe_load(file)
        logger.info("load the bitwidth config for weight!")
        logger.info(f"the config for weight is {opt.config_weight_mp}")
        qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

        for filename, config in load_yaml_files(opt.dir_act_mp):
            bit_config = config
            logger.info("############################# load the bitwidth config for act! \nInference with w&a quantized! #############################")
            logger.info(f"############################### config: {filename} ###############################")

            qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='act')

            # opt.image_folder = os.path.join(opt.base_path,f'images_{filename}')
            os.makedirs(opt.image_folder, exist_ok=True)
            # prompts = ["portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"]
            sample_fid(prompts, qnn, pipe, opt, batch_size=1, quant_inference = True, is_fp16 = False, save_path=filename)
            mse_tmp = MSE_pixel(
                    f"{opt.image_folder}/{filename}.png",
                    os.path.join(opt.reference_img,'0.png')
                )
            if mse_tmp < mse_min:
                mse_min = mse_tmp
                final_config = bit_config
                final_name = filename
        write_yaml_file(os.path.join(opt.base_path, "final_act_mp.yaml"), final_config)
        qnn.load_bitwidth_config(model=qnn, bit_config=final_config, bit_type='act')
        sample_fid(prompts, qnn, pipe, opt, batch_size=1, quant_inference = True, is_fp16 = False, save_path='final')


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
    # print("#######################################################################")
    # disable guidance_scale by passing 0
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0).images
    return image


def sample_fid(prompt, unet, pipe, opt, batch_size, quant_inference = False, is_fp16 = False, save_path=None):

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(opt.seed)  # Seed generator to create the initial latent noise
    total = len(prompt)
    # n = 16  # 按批量推理
    num = total // batch_size
    # num = opt.n_samples # DEBUG_ONLY: generate batch_size*n_sample images only
    img_id = 0
    logger.info(f"starting from image {img_id}")
    # total_n_samples = max_images

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
                image[j].save(f"{opt.image_folder}/{save_path}.png")
                img_id += 1


def load_yaml_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            with open(os.path.join(directory, filename), 'r') as file:
                data = yaml.safe_load(file)
                yield filename, data


def MSE_pixel(img_path1, img_path2, bs=1):
    img1 = (cv2.imread(img_path1))
    img2 = (cv2.imread(img_path2))
    mse = np.mean((img1-img2)**2)
    return mse

def write_yaml_file(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


if __name__ == "__main__":
    main()


