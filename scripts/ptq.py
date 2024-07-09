import argparse, os, datetime, gc, yaml
import logging
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from pytorch_lightning import seed_everything
import torch
import shutil
import sys

from qdiff.optimization.model_recon import model_reconstruction
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_model, load_quant_params, get_quant_calib_data

from qdiff.models.quant_block_forward_func import convert_model_split, convert_transformer_storable, set_shortcut_split


logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/sdxl.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INFO: add bakup file and bakup cfg into logpath for debug
    if os.path.exists(os.path.join(outpath,'config.yaml')):
        os.remove(os.path.join(outpath,'config.yaml'))
    shutil.copy(opt.config, os.path.join(outpath,'config.yaml'))
    if os.path.exists(os.path.join(outpath,'qdiff')): # if exist, overwrite
        shutil.rmtree(os.path.join(outpath,'qdiff'))
    shutil.copytree('./quant_utils/qdiff', os.path.join(outpath,'qdiff'))

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

    config = OmegaConf.load(f"{opt.config}")
    model = get_model(config.model, fp16=False, return_pipe=False, convert_model_for_quant=True)
    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = False if wq_params is None else True
    use_act_quant = False if aq_params is None else True

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

    if not config.quant.grad_checkpoint:
        logger.info('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)

    logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
    calib_data_ckpt = torch.load(config.calib_data.path, map_location='cpu')
    calib_data = get_quant_calib_data(config, calib_data_ckpt, config.calib_data.n_steps, model_type=config.model.model_type)
    del(calib_data_ckpt)
    gc.collect()

    # prepare data for init the model
    calib_batch_size = config.calib_data.batch_size  # DEBUG: actually for weight quant, only bs=1 is enough

    if config.model.model_type == 'sdxl':
        text_embeds = calib_data[3]["text_embeds"]
        time_ids = calib_data[3]["time_ids"]
        logger.info(f"Calibration data shape: {calib_data[0].shape} {calib_data[1].shape} {calib_data[2].shape} {text_embeds.shape} {time_ids.shape}")
        del(text_embeds)
        del(time_ids)
        calib_xs, calib_ts, calib_cs, calib_added_conds = calib_data
        calib_added_text_embeds = calib_added_conds["text_embeds"]
        calib_added_time_ids = calib_added_conds["time_ids"]
        calib_added_conds = {}
        calib_added_conds["text_embeds"] = calib_added_text_embeds[:calib_batch_size].cuda()
        calib_added_conds["time_ids"] = calib_added_time_ids[:calib_batch_size].cuda()
    elif config.model.model_type == 'sd':
        logger.info(f"Calibration data shape: {calib_data[0].shape} {calib_data[1].shape} {calib_data[2].shape}")
        calib_xs, calib_ts, calib_cs = calib_data
        calib_added_conds = {}

    # ----------------------- get the quant params (training-free), using the calibration data -------------------------------------
    with torch.no_grad():
        _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), added_cond_kwargs=calib_added_conds)
        qnn.set_module_name_for_quantizer(module=qnn.model)  # add the module name as attribute for each quantizer

        # --- the weight quantization -----
        qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
        _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), added_cond_kwargs=calib_added_conds)
        logger.info("weight initialization done!")
        qnn.set_quant_init_done('weight')
        torch.cuda.empty_cache()

        # --- the activation quantization -----
        # by default, use the running_mean of calibration data to determine activation quant params
        qnn.set_quant_state(True, True) # quantize activation with fixed quantized weight
        logger.info('Running stat for activation quantization')
        inds = np.arange(calib_xs.shape[0])
        np.random.shuffle(inds)
        rounds = int(calib_xs.size(0) / calib_batch_size)

        for i in trange(rounds):
            if len(calib_added_conds) != 0:
                calib_added_conds["text_embeds"] = calib_added_text_embeds[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda()
                calib_added_conds["time_ids"] = calib_added_time_ids[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda()
            _ = qnn(calib_xs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                calib_ts[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                calib_cs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                added_cond_kwargs=calib_added_conds)
        qnn.set_quant_init_done('activation')
        logger.info("activation initialization done!")
        torch.cuda.empty_cache()

    # ----------------------- get the quant params (training opt), using the calibration data -------------------------------------
    # INFO: LSQ-like quant params tuning, **not used in MxiDQ**
    # INFO: in configs, whether leave the whole optimization.weight being empty or just empty params
    weight_optimization = False
    if config.quant.weight.optimization is not None:
        if config.quant.weight.optimization.params is not None:
            weight_optimization = True
    act_optimization = False
    if config.quant.activation.optimization is not None:
        if config.quant.activation.optimization.params is not None:
            act_optimization = True
    use_optimization = any([weight_optimization, act_optimization])

    if not use_optimization:  # no need for optimization-based quantization
        pass
    else:
        # INFO: get the quant parameters
        qnn.train()  # setup the train_mode
        opt_d = {}
        if weight_optimization:
            opt_d['weight'] = getattr(config.quant,'weight').optimization.params.keys()
        else:
            opt_d['weight'] = None
        if act_optimization:
            opt_d['activation'] = getattr(config.quant,'activation').optimization.params.keys()
        else:
            opt_d['activation'] = None
        qnn.replace_quant_buffer_with_parameter(opt_d)

        if config.quant.weight.optimization.joint_weight_act_opt:  # INFO: optimize all quant params together
            assert weight_optimization and act_optimization
            qnn.set_quant_state(True, True)
            opt_target = 'weight_and_activation'
            param_types = {
                    'weight': list(config.quant.weight.optimization.params.keys()),
                    'activation': list(config.quant.activation.optimization.params.keys())
                    }
            if 'alpha' in param_types['weight']:
                assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
            if 'alpha' in param_types['activation']:
                assert config.quant.activation.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
            model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)
            logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types,opt_target))
            torch.save(qnn.get_quant_params_dict(), os.path.join(outpath, "ckpt.pth"))

        else:  # INFO: sequantially quantize weight and activation quant params
            # --- the weight quantization (with optimization) -----
            if not weight_optimization:
                logger.info("No quant parmas, skip optimizing weight quant parameters")
            else:
                qnn.set_quant_state(True, False)  # use FP activation
                opt_target = 'weight'
                # --- unpack the config ----
                param_types = list(config.quant.weight.optimization.params.keys())
                if 'alpha' in param_types:
                    assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
                # INFO: recursive iter through all quantizers, for weight/act quantizer, optimize the delta & alpha (if any) together
                model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)  # DEBUG_ONLY
                logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types, opt_target))
                torch.save(qnn.get_quant_params_dict(), os.path.join(outpath, "ckpt.pth"))

            # --- the activation quantization (with optimization) -----
            if not act_optimization:
                logger.info("No quant parmas, skip optimizing activation quant parameters")
            else:
                qnn.set_quant_state(True, True)  # use FP activation
                opt_target = 'activation'
                # --- unpack the config ----
                param_types = list(config.quant.activation.optimization.params.keys())
                if 'alpha' in param_types:
                    assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
                # INFO: recursive iter through all quantizers, for weight/act quantizer, optimize the delta & alpha (if any) together
                model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)  # DEBUG_ONLY
                logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types, opt_target))
                torch.save(qnn.get_quant_params_dict(), os.path.join(outpath, "ckpt.pth"))

        qnn.replace_quant_parameter_with_buffers(opt_d)  # replace back to buffer for saving

    # save the quant params
    logger.info("Saving calibrated quantized UNet model")
    quant_params_dict = qnn.get_quant_params_dict()
    torch.save(quant_params_dict, os.path.join(outpath, "ckpt.pth"))

if __name__ == "__main__":
    main()
