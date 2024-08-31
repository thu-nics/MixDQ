import os
import yaml
import logging
import argparse, os
import logging
from typing import Union
from omegaconf import OmegaConf
import math

import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

import sys

from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer, lp_loss
from qdiff.utils import get_model, load_quant_params

from diffusers.models.unet_2d_blocks import UpBlock2D, AttnUpBlock2D, CrossAttnUpBlock2D, DownBlock2D, AttnDownBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn
from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)

def LossFunction(pred, tgt, grad=None):
        """
        Compute the quant error: MES and the SQNR
        """

        # MSE Loss
        mse = lp_loss(pred, tgt, p=2, reduction='all')
        
        # SQNR
        err = pred - tgt
        tgt = torch.sum(tgt**2)  #or tgt = torch.norm(tgt)**2
        err = torch.sum(err**2)  #or err = torch.norm(err)**2
        # 二者相除
        divided = tgt / err
        # 直接计算信噪比
        sqnr = 10*torch.log10(divided)

        return mse, sqnr


def set_quant_state(model=None, weight_quant: bool = False, act_quant: bool = False):
    for m in model.modules():
        if isinstance(m, (QuantLayer, BaseQuantBlock)):
            m.set_quant_state(weight_quant, act_quant)


def get_layer_sqnr(model=None, quantized_model: QuantModel=None, weight_quant=True, act_quant=False, saved_in_out=None, config_sqnr={}, cur_bit=0, prefix=""):
    '''
    compute the error of the output of the model with a certain layer quantized
    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {)}")
        if isinstance(module, QuantLayer):
            if not 'ff' in full_name and not 'attn2' in full_name:
                module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                torch.cuda.empty_cache()
                logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")

                mse_mean = 0
                sqnr_mean = 0
                with torch.no_grad():
                    output_quant = quantized_model(
                        saved_in_out['xs'].cuda(), 
                        saved_in_out['ts'].cuda(), 
                        saved_in_out['text_embs'].cuda(), 
                        added_cond_kwargs=saved_in_out['added_cond_kwargs'])[0]
                mse_mean, sqnr_mean = LossFunction(output_quant, saved_in_out['outputs'])
                logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))

                if not full_name in config_sqnr.keys():
                    config_sqnr[full_name] = []
                config_sqnr[full_name].append(float(sqnr_mean))

                quantized_model.set_quant_state(False, False)
        else:
            get_layer_sqnr(model=module, quantized_model=quantized_model, weight_quant=weight_quant, act_quant=act_quant, saved_in_out=saved_in_out, config_sqnr=config_sqnr, cur_bit=cur_bit, prefix=full_name+".")

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        nargs="?",
        help="dir to load the ckpt",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="path for generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    # parser.add_argument(
    #     "--model_id", type=str, required=True,
    #     default="stabilityai/sdxl-turbo",
    #     help="the model type: sdxl or sdxl-turbo"
    # )
    parser.add_argument(
        "--calib_data_path", type=str,
        help="read the input and output data from file"
    )
    # parser.add_argument(
    #     "--unet_input_path", type=str, required=True,
    #     help="the input of the unet"
    # )
    # parser.add_argument(
    #     "--unet_output_path", type=str, required=True,
    #     help="the output of the unet with the weight of fp32"
    # )
    parser.add_argument(
        "--is_fp16", action="store_true", 
        help="if to use fp16 weight to inference"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--ssim_bit_config", type=str,
        help="result of stage 1"
    )
    parser.add_argument(
        "--sensitivity_type", type=str, required=True,
        help="weight or act"
    )
    parser.add_argument(
        "--sensitivity_path", type=str,
        help="weight or act"
    )
    # quantization configs
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
        "--skip_quant_act",
        action='store_true',
    )
    parser.add_argument(
        "--skip_quant_weight",
        action='store_true',
    )
    # parser.add_argument(
    #     "--template_config",
    #     type=str,
    #     help="a template to init a sensitivity config",
    # )

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
    model, pipe = get_model(config.model, fp16=False, return_pipe=True, convert_model_for_quant=True)
    assert(config.conditional)
    if opt.calib_data_path is None:
        opt.calib_data_path = config.calib_data.path

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
    logger.info(qnn)

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
    model = qnn
    qnn.cuda()


    # compute the sensitivity
    logger.info("quant_error_unet_output...")

    # load the input and output
    saved_in_out = torch.load(opt.calib_data_path)
    for k_ in saved_in_out:
        if k_ == 'prompts':
            continue
        elif k_ != 'added_cond_kwargs':
            saved_in_out[k_] = saved_in_out[k_].squeeze(0)  # squeeze the first dim
        else:
            for k_2 in saved_in_out[k_]:
                saved_in_out[k_][k_2] = saved_in_out[k_][k_2].squeeze(0)

    # disable the quant mode
    model.set_quant_state(False, False)

    logger.info("Verify the correctness")
    mse_mean = 0
    sqnr_mean = 0
    with torch.no_grad():
        output_val = qnn(
            saved_in_out['xs'].cuda(), 
            saved_in_out['ts'].cuda(), 
            saved_in_out['text_embs'].cuda(), 
            added_cond_kwargs = saved_in_out['added_cond_kwargs'])[0]
        mse_mean, sqnr_mean = LossFunction(output_val, saved_in_out['outputs'])

    logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))


    ##################################################################################
    if opt.sensitivity_type == 'weight':
        config_sqnr = {}
        # with open(opt.template_config, 'r') as file:
        #     config_sqnr = yaml.safe_load(file)
        
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            get_layer_sqnr(model=qnn, quantized_model=qnn, weight_quant=True, act_quant=False, saved_in_out=saved_in_out, config_sqnr=config_sqnr, cur_bit=bit_width)
    
    elif opt.sensitivity_type == 'act':
        config_sqnr = {}
        # with open(opt.template_config, 'r') as file:
        #     config_sqnr = yaml.safe_load(file)
        
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            get_layer_sqnr(model=qnn, quantized_model=qnn, weight_quant=False, act_quant=True, saved_in_out=saved_in_out, config_sqnr=config_sqnr, cur_bit=bit_width)

    save_path_config = opt.base_path+'/sensitivity_{}_quality.yaml'.format(opt.sensitivity_type)
    with open(save_path_config, 'w') as file:
        yaml.dump(config_sqnr, file)


if __name__ == "__main__":
    main()



                                            
