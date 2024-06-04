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
sys.path.insert(0, '/home/fangtongcheng/fast_ptq/diffuser-dev/q-diffusion')
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
        # tensor1 和 tensor2 的后三个维度的元素求平方和
        tgt = torch.sum(tgt**2)  #or tgt = torch.norm(tgt)**2
        err = torch.sum(err**2)  #or err = torch.norm(err)**2
        # 二者相除
        divided = tgt / err
        # 直接计算信噪比
        sqnr = 10*torch.log10(divided)

        # SQNR
        # another settings
        # err = pred - tgt
        # # tensor1 和 tensor2 的后三个维度的元素求平方和
        # tgt = torch.sum(tgt**2, dim=(1,2,3))  #or tgt = torch.norm(tgt)**2
        # err = torch.sum(err**2, dim=(1,2,3))  #or err = torch.norm(err)**2
        # # 二者相除
        # divided = tgt / err
        # # 最后对第一个维度（也就是batch）求平均
        # sqnr = 10*torch.log10(torch.mean(divided))
        return mse, sqnr


def set_quant_state(model=None, weight_quant: bool = False, act_quant: bool = False):
    for m in model.modules():
        if isinstance(m, (QuantLayer, BaseQuantBlock)):
            m.set_quant_state(weight_quant, act_quant)


##### the kurtosis #####
def kurtosis(x):
    n = x.numel()
    mean = torch.mean(x)
    std = torch.std(x)
    z = (x - mean) / std
    return torch.mean(z ** 4) - 3


##### mse between the quantized weight and the original weight #####
def weight_quant_error(model=None, quantized_model: QuantModel=None, prefix=""):
    '''
    the quant error of the quantized weight
    '''
    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # compute the quant error of the weight
        if isinstance(module, QuantLayer):
            # get the weight of fp32
            weight_fp32 = module.weight
            # get the weight quantized
            with torch.no_grad():
                if module.split != 0:
                    weight_0 = module.weight_quantizer(module.weight[:, :module.split, ...])
                    weight_1 = module.weight_quantizer_0(module.weight[:, module.split:, ...])
                    weight_quantized = torch.cat([weight_0, weight_1], dim=1)
                else:
                    weight_quantized = module.weight_quantizer(module.weight)
            torch.cuda.empty_cache()
            logger.info(f"{full_name}")
            mse, sqnr = LossFunction(weight_fp32, weight_quantized)
            logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse*1e5), float(sqnr)))
        else:
            weight_quant_error(model=module, quantized_model=quantized_model, prefix=full_name+".")


##### mse between the quantized output and the original output #####
# Block wise的敏感度分析，此时没有产生敏感度列表，应该遍历block
def top_block_set_quant(model=None, quantized_model: QuantModel=None, weight_quant=True, act_quant=False, input_list=None, output_fp32_list=None, prefix=""):
    '''
    compute the error of the output of the model with a certain top level block quantized
    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {)}")
        if isinstance(module, (UpBlock2D, AttnUpBlock2D, CrossAttnUpBlock2D, DownBlock2D, AttnDownBlock2D,
        CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn)):
            set_quant_state(module, weight_quant=weight_quant, act_quant=act_quant)
            torch.cuda.empty_cache()
            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")

            # TODO: 进行推理单个模块权重量化的推理
            mse_mean = 0
            sqnr_mean = 0
            for idx, input_data in enumerate(input_list):
                # 每次取一份输入(bs=8)，计算一次quant_error
                with torch.no_grad():
                    output_w8a32 = quantized_model(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs=input_data['added_cond_kwargs'])[0]
                mse, sqnr = LossFunction(output_w8a32, output_fp32_list[idx])
                mse_mean = mse_mean + mse
                sqnr_mean = sqnr_mean + sqnr
            # 对于不同的输入取平均
            mse_mean = mse_mean / (idx+1)
            sqnr_mean = sqnr_mean / (idx+1)
            logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))

            quantized_model.set_quant_state(False, False)
        else:
            top_block_set_quant(model=module, quantized_model=quantized_model, weight_quant=True, act_quant=False, input_list=input_list, output_fp32_list=output_fp32_list, prefix=full_name+".")


# Block wise的敏感度分析，此时没有产生敏感度列表，应该遍历block
def lower_block_set_quant(model=None, quantized_model: QuantModel=None, weight_quant=True, act_quant=False, input_list=None, output_fp32_list=None, config_sqnr={}, config_mse={}, prefix=""):
    '''
    compute the error of the output of the model with a certain lower level block quantized
    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {)}")
        if isinstance(module, BaseQuantBlock):
            # set_quant_state(module, weight_quant=weight_quant, act_quant=act_quant)
            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
            torch.cuda.empty_cache()
            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
            
            # TODO: 进行推理单个模块权重量化的推理
            mse_mean = 0
            sqnr_mean = 0
            for idx, input_data in enumerate(input_list):
                # 每次取一份输入(bs=8)，计算一次quant_error
                with torch.no_grad():
                    output_w8a32 = quantized_model(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs=input_data['added_cond_kwargs'])[0]
                mse, sqnr = LossFunction(output_w8a32, output_fp32_list[idx])
                mse_mean = mse_mean + mse
                sqnr_mean = sqnr_mean + sqnr
            # 对于不同的输入取平均
            mse_mean = mse_mean / (idx+1)
            sqnr_mean = sqnr_mean / (idx+1)
            logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))
            
            config_sqnr[full_name] = float(sqnr_mean)
            config_mse[full_name] = float(mse_mean)

            # 重新把模型置为非量化模式
            quantized_model.set_quant_state(False, False)

        elif isinstance(module, QuantLayer):
            # set_quant_state(module, weight_quant=weight_quant, act_quant=act_quant)
            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
            torch.cuda.empty_cache()
            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
            
            # TODO: 进行推理单个模块权重量化的推理
            mse_mean = 0
            sqnr_mean = 0
            for idx, input_data in enumerate(input_list):
                # 每次取一份输入(bs=8)，计算一次quant_error
                with torch.no_grad():
                    output_w8a32 = quantized_model(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs=input_data['added_cond_kwargs'])[0]
                mse, sqnr = LossFunction(output_w8a32, output_fp32_list[idx])
                mse_mean = mse_mean + mse
                sqnr_mean = sqnr_mean + sqnr
            # 对于不同的输入取平均
            mse_mean = mse_mean / (idx+1)
            sqnr_mean = sqnr_mean / (idx+1)
            logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))
            
            config_sqnr[full_name] = float(sqnr_mean)
            config_mse[full_name] = float(mse_mean)

            # 重新把模型置为非量化模式
            quantized_model.set_quant_state(False, False)

        else:
            lower_block_set_quant(model=module, quantized_model=quantized_model, weight_quant=True, act_quant=False, input_list=input_list, output_fp32_list=output_fp32_list, config_sqnr=config_sqnr, config_mse=config_mse, prefix=full_name+".")


# Layer wise的敏感度分析，此时没有产生敏感度列表，应该遍历layer
def layer_set_quant(model=None, quantized_model: QuantModel=None, weight_quant=True, act_quant=False, input_list=None, output_fp32_list=None, config_sqnr={}, cur_bit=0, prefix=""):
    '''
    compute the error of the output of the model with a certain layer quantized
    '''

    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {)}")
        if isinstance(module, QuantLayer):
            # if not 'ff' in full_name and not 'attn2' in full_name:
                # set_quant_state(module, weight_quant=weight_quant, act_quant=act_quant)
                module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                torch.cuda.empty_cache()
                logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                
                # TODO: 进行推理单个模块权重量化的推理
                mse_mean = 0
                sqnr_mean = 0
                for idx, input_data in enumerate(input_list):
                    # 每次取一份输入(bs=8)，计算一次quant_error
                    with torch.no_grad():
                        output_quant = quantized_model(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs=input_data['added_cond_kwargs'])[0]
                        # torch.save(output_quant, '/home/fangtongcheng/fast_ptq/diffuser-dev/analysis_tools/error_func/unet_in_out_w4a32')
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


# group wise的敏感度分析
def group_set_quant(model=None, quantized_model: QuantModel=None, module_class=[], class_ignore=[], weight_quant=True, act_quant=False, prefix=""):
        for name, module in model.named_children():
            full_name = prefix + name if prefix else name
            if isinstance(module, QuantLayer):
                    # 按照结构来配置量化与否
                    if module_class in full_name:
                        # if all(element not in full_name for element in class_ignore):
                            # set_quant_state(module, weight_quant=weight_quant, act_quant=act_quant)
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                            
            else:
                group_set_quant(model=module, quantized_model=quantized_model,
                                module_class=module_class, class_ignore=class_ignore, weight_quant=weight_quant, 
                                act_quant=act_quant, prefix=full_name+".")


##### mse between the quantized act and the original act #####
class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch 
        if self.store_output:
            self.output_store = output_batch 
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut_SDXL:
    def __init__(self, model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock, nn.Module],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        # self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)

    def __call__(self, x, timesteps, context=None, added_conds=None):
        self.model.eval()
        # self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)  
        with torch.no_grad():
            try:
                _ = self.model(x, timesteps, context, added_cond_kwargs=added_conds)
            except StopForwardException:
                pass

        handle.remove()

        if len(self.data_saver.input_store) > 1 and len(self.data_saver.input_store) < 7 and torch.is_tensor(self.data_saver.input_store[1]):
            return (self.data_saver.input_store[0].detach(),  
                self.data_saver.input_store[1].detach())
        elif len(self.data_saver.input_store) == 7:
            # 针对QuantTransformerBlock 有7个输入（待优化）
            input_tuple = []
            for input in self.data_saver.input_store:
                if input == None:
                    input_tuple.append(input)
                else:
                    input_tuple.append(input.detach())
            return tuple(input_tuple)  # difference
        else:
            return self.data_saver.input_store[0].detach()


def get_data(model, module, input_data):
    data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
    handle = module.register_forward_hook(data_saver) 

    with torch.no_grad():
        _ = model(input_data['xs'].cuda(), input_data['ts'].cuda(), input_data['text_embs'].cuda(), added_cond_kwargs=input_data['added_cond_kwargs'])[0]

    handle.remove()

    if len(data_saver.input_store) > 1 and len(data_saver.input_store) < 7 and torch.is_tensor(data_saver.input_store[1]):
        # the input of the ResnetBlock2D contains two tensors
        return (data_saver.input_store[0].detach(), 
        data_saver.input_store[1].detach()), data_saver.output_store.detach()
    elif len(data_saver.input_store) == 7:
        # 针对QuantTransformerBlock 有7个输入（待优化）
        input_tuple = []
        for input in data_saver.input_store:
            if input == None:
                input_tuple.append(input)
            else:
                input_tuple.append(input.detach())
        return tuple(input_tuple), data_saver.output_store.detach()
    else:
        return data_saver.input_store[0].detach(), data_saver.output_store.detach()


def act_quant_error(model=None, quantized_model: QuantModel=None, input_list=None, quant_weight=False, quant_act=False, prefix=""):
    '''
    the quant error of the quantized act
    '''
    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        # compute the quant error of the weight
        if isinstance(module, QuantLayer):
            quantized_model.set_quant_state(quant_weight, quant_act)  # 关闭量化

            mse_mean = 0
            sqnr_mean = 0
            kurt_mean = 0
            logger.info(f"{full_name}")
            for idx, input_data in enumerate(input_list):
                act_fp32, _ = get_data(quantized_model, module, input_data)  # 获取对应module的输入数据
                # get the act quantized
                with torch.no_grad():
                    if module.split != 0:
                        if module.act_quant_mode == 'qdiff':
                            act_quantized_0 = module.act_quantizer(act_fp32[:, :module.split, :, :])
                            act_quantized_1 = module.act_quantizer_0(act_fp32[:, module.split:, :, :])
                        act_quantized = torch.cat([act_quantized_0, act_quantized_1], dim=1)
                    else:
                        if module.act_quant_mode == 'qdiff':
                            act_quantized = module.act_quantizer(act_fp32)
                torch.cuda.empty_cache()
                mse, sqnr = LossFunction(act_fp32, act_quantized)
                kurt = kurtosis(act_fp32)
                mse_mean = mse_mean + mse
                sqnr_mean = sqnr_mean + sqnr 
                kurt_mean = kurt_mean + kurt

            mse_mean = mse_mean / (idx+1)
            sqnr_mean = sqnr_mean / (idx+1)
            kurt_mean = kurt_mean / (idx+1)

            logger.info('MSE:{:.5f}x10^(-5),\tSQNR:{:.5f}dB \n'.format(float(mse_mean*1e5), float(sqnr_mean)))

            logger.info('Kurt:{:.5f} \n'.format(float(kurt_mean)))

        else:
            act_quant_error(model=module, quantized_model=quantized_model, input_list=input_list, prefix=full_name+".")


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
        "--model_id", type=str, 
        # required=True,
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
    # parser.add_argument(
    #     "--sensitivity_path", type=str, required=True,
    #     help="weight or act"
    # )
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
    # analyse mode
    parser.add_argument(
        "--analysis_target", type=str, required=True,
        choices=["quant_error_unet_output", "quant_error_weight", "quant_error_act"], 
        help="what to compute"
    )
    
    opt = parser.parse_args()

    seed_everything(opt.seed)

    opt.outdir = os.path.join(opt.base_path,'generated_images') if opt.image_folder is None else opt.image_folder
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
    model, pipe = get_model(model_id=opt.model_id, cache_dir="/share/public/diffusion_quant/huggingface/sdxl-turbo", quant_inference = True, is_fp16 = False, return_pipe=True)

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
    # 计算敏感度
    if opt.analysis_target == 'quant_error_unet_output':
        logger.info("quant_error_unet_output!")
        # use min-max based quantized model
        # 加载UNet的输入（取了8个prompt，已经提前保存到磁盘）

        input_list = []
        input_list = torch.load(opt.unet_input_path) # torch.load('/home/fangtongcheng/diffuser-dev/unet_input_fp32')

        for input_data in input_list:
            # 把字典数据先存到CUDA上，其余数据可以在输入时直接.cuda，字典数据要直接传进去，所以先搬运
            input_data['added_cond_kwargs']['time_ids'] = input_data['added_cond_kwargs']['time_ids'].cuda()
            input_data['added_cond_kwargs']['text_embeds'] = input_data['added_cond_kwargs']['text_embeds'].cuda()

        # 关闭所有量化，进行全精度推理
        model.set_quant_state(False, False)

        # 加载全精度UNet的输出，校验是否全部关闭量化
        output_fp32_list = torch.load(opt.unet_output_path)  # torch.load('/home/fangtongcheng/diffuser-dev/unet_output_fp32')
        # output_fp32_list = []
        logger.info("Verify the correctness")
        mse_mean = 0
        sqnr_mean = 0
        for idx, input_data in enumerate(input_list):
            # 验证计算的准确性，全精度模型的输出应该和保存下来的输出相等，误差为0
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
        # 指定是分析权重敏感度还是激活值敏感度，命名时请注意区分
        if opt.sensitivity_type == 'weight':
            config_sqnr = {}
            # 创建一个新字典，遍历所有层，在这里先加载一个原有yaml文件，得到一个老的字典（键:module_name，key:a list of sensitivity），再根据新跑出来的敏感度更新这个老的字典的value
            # 激活值和权重均用这个字典即可，因为module_name是一致的，只需要修改不同的value即可
            with open('/share/public/diffusion_quant/mixed_percision_quant/sensitivity_log/weight/mse/mse_all_layer_weight.yaml', 'r') as file:
                config_sqnr = yaml.safe_load(file)
            
            for bit_width in [2,4,8]:
                # 全局比特位宽重置
                logger.info(f"\nthe bit width is {bit_width}!\n")
                # 每一轮都重置所有层的比特位宽
                qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')

                logger.info("################# Start to quantize the layers one by one #################")
                
                # 先把所有层的量化关闭
                qnn.set_quant_state(False, False)

                # 逐个量化权重，计算敏感度
                layer_set_quant(model=qnn, quantized_model=qnn, weight_quant=True, act_quant=False, input_list=input_list, output_fp32_list=output_fp32_list, config_sqnr=config_sqnr, cur_bit=bit_width)
        
        elif opt.sensitivity_type == 'act':
            config_sqnr = {}
            # 创建一个新字典，仅仅测算与mse有关的权重即cross attention 和 ff 以外的层
            with open('/share/public/diffusion_quant/mixed_percision_quant/sensitivity_log/weight/mse/mse_all_layer_weight.yaml', 'r') as file:
                config_sqnr = yaml.safe_load(file)
            
            for bit_width in [2,4,8]:
                # 全局比特位宽设置
                logger.info(f"\nthe bit width is {bit_width}!\n")
                qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

                logger.info("################# Start to quantize the layers one by one #################")
                qnn.set_quant_state(False, False)

                # SSIM混合精度配置
                # with open(opt.ssim_bit_config, 'r') as file:
                #     bit_config = yaml.safe_load(file)
                # logger.info("load the bitwidth config!")
                # qnn.load_bitwidth_config(model=qnn, bit_config=bit_config, bit_type='weight')

                # # 把ssim相关权重量化设置为True
                # qnn.set_layer_quant(model=qnn, group_list=['ff', 'attn2'], quant_level='per_group', weight_quant=True, act_quant=False)

                # 逐个量化剩下的权重，计算敏感度
                layer_set_quant(model=qnn, quantized_model=qnn, weight_quant=False, act_quant=True, input_list=input_list, output_fp32_list=output_fp32_list, config_sqnr=config_sqnr, cur_bit=bit_width)
        save_path_config = opt.base_path+'/sensitivity.yaml'
        with open(save_path_config, 'w') as file:
            yaml.dump(config_sqnr, file)




if __name__ == "__main__":
    main()