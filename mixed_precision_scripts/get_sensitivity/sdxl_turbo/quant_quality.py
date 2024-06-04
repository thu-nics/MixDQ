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
from qdiff.utils import get_model, load_model_from_config, load_quant_params

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


def layer_set_quant(model=None, quantized_model: QuantModel=None, weight_quant=True, act_quant=False, input_list=None, output_fp32_list=None, config_sqnr={}, cur_bit=0, prefix=""):
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
                for idx, input_data in enumerate(input_list):
                    with torch.no_grad():
                        output_quant = quantized_model(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs=input_data['added_cond_kwargs'])[0]
                    mse, sqnr = LossFunction(output_quant, output_fp32_list[idx])
                    mse_mean = mse_mean + mse
                    sqnr_mean = sqnr_mean + sqnr
                # 对于不同的输入取平均
                mse_mean = mse_mean / (idx+1)
                sqnr_mean = sqnr_mean / (idx+1)
                logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))
                config_sqnr[full_name][int(math.log2(cur_bit)-1)] = float(sqnr_mean)

                quantized_model.set_quant_state(False, False)
        else:
            layer_set_quant(model=module, quantized_model=quantized_model, weight_quant=weight_quant, act_quant=act_quant, input_list=input_list, output_fp32_list=output_fp32_list, config_sqnr=config_sqnr, cur_bit=cur_bit, prefix=full_name+".")


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
    parser.add_argument(
        "--model_id", type=str, required=True,
        default="stabilityai/sdxl-turbo",
        help="the model type: sdxl or sdxl-turbo"
    )
    parser.add_argument(
        "--unet_input_path", type=str, required=True,
        help="the input of the unet"
    )
    parser.add_argument(
        "--unet_output_path", type=str, required=True,
        help="the output of the unet with the weight of fp32"
    )
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
    parser.add_argument(
        "--template_config",
        type=str,
        help="a template to init a sensitivity config",
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
    model, pipe = get_model(model_id=opt.model_id, cache_dir="/share/public/diffusion_quant/huggingface/hub", quant_inference = True, is_fp16 = False, return_pipe=True, model_type=config.model.model_type)

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

    # TODO: load quant params
    load_quant_params(qnn, opt.ckpt)
    model = qnn
    qnn.cuda()


    ######################################################################
    # compute the sensitivity
    logger.info("quant_error_unet_output!")

    # use min-max based quantized model
    input_list = []
    input_list = torch.load(opt.unet_input_path)

    for input_data in input_list:
        input_data['added_cond_kwargs']['time_ids'] = input_data['added_cond_kwargs']['time_ids'].cuda()
        input_data['added_cond_kwargs']['text_embeds'] = input_data['added_cond_kwargs']['text_embeds'].cuda()

    # disable the quant mode
    model.set_quant_state(False, False)

    output_fp32_list = torch.load(opt.unet_output_path)
    # output_fp32_list = []
    logger.info("Verify the correctness")
    mse_mean = 0
    sqnr_mean = 0
    for idx, input_data in enumerate(input_list):
        # inference with the full precision model, get the output
        with torch.no_grad():
            output_val = qnn(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs = input_data['added_cond_kwargs'])[0]
        mse, sqnr = LossFunction(output_val, output_fp32_list[idx])
        # output_fp32_list.append(output_val)
        mse_mean = mse_mean + mse
        sqnr_mean = sqnr_mean + sqnr
    # 对于不同的输入取平均
    mse_mean = mse_mean / (idx+1)
    sqnr_mean = sqnr_mean / (idx+1)
    logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))


    ##################################################################################
    if opt.sensitivity_type == 'weight':
        config_sqnr = {}
        with open(opt.template_config, 'r') as file:
            config_sqnr = yaml.safe_load(file)
        
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            layer_set_quant(model=qnn, quantized_model=qnn, weight_quant=True, act_quant=False, input_list=input_list, output_fp32_list=output_fp32_list, config_sqnr=config_sqnr, cur_bit=bit_width)
    
    elif opt.sensitivity_type == 'act':
        config_sqnr = {}
        with open(opt.template_config, 'r') as file:
            config_sqnr = yaml.safe_load(file)
        
        for bit_width in [2,4,8]:
            logger.info(f"\nthe bit width is {bit_width}!\n")
            qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

            logger.info("################# Start to quantize the layers one by one #################")
            qnn.set_quant_state(False, False)

            layer_set_quant(model=qnn, quantized_model=qnn, weight_quant=False, act_quant=True, input_list=input_list, output_fp32_list=output_fp32_list, config_sqnr=config_sqnr, cur_bit=bit_width)
    save_path_config = opt.base_path+'/sensitivity.yaml'
    with open(save_path_config, 'w') as file:
        yaml.dump(config_sqnr, file)


if __name__ == "__main__":
    main()



                                            
