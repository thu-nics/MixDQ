from diffusers import LatentConsistencyModelPipeline

import argparse
import torch
import os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from qdiff.models.customized_pipeline import CustomizedStableDiffusionPipeline, CustomizedStableDiffusionXLPipeline
from qdiff.utils import get_model, prepare_coco_text_and_image

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--save_image_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")

    # set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set pipelin]
    if config.model.model_type == 'sdxl':
        custom_pipe_cls = CustomizedStableDiffusionXLPipeline
    elif config.model.model_type == 'sd':
        custom_pipe_cls = CustomizedStableDiffusionPipeline
    else:
        raise NotImplementedError

    model, pipe = get_model(config.model, fp16=False, return_pipe=True, custom_pipe_cls=custom_pipe_cls)
    pipe.to(torch_device="cuda", torch_dtype=torch.float32)

    # load sd_coco calibration data]
    json_file = "./scripts/utils/captions_val2014.json"
    prompt_list, image_path = prepare_coco_text_and_image(json_file=json_file)
    prompts = prompt_list[0:config.calib_data.n_samples]

    # get init latents
    # latents = torch.randn([int(config.calib_data.n_samples), 4, 64, 64])

    # begin data generation
    save_data = {"prompts": prompts}
    assert config.calib_data.n_samples % config.calib_data.batch_size == 0, "the n_samples should be divisible by batch_size"
    bs = config.calib_data.batch_size
    sample_id = 0
    for _ in range(config.calib_data.n_samples//config.calib_data.batch_size):
        print(f"Generated {sample_id} results")
        cur_save_data = {} # data of the current loop
        # set pipeline input
        # input_noise = latents[sample_id:sample_id + bs]
        prompt = prompts[sample_id:sample_id+bs]
        # input_cond_emb = prompt_embeds[sample_id : sample_id + bs] # use embedding as input

        # get guidance scale embedding
        # if config.calib_data.scale_type == "fix":
            # w = torch.tensor(config.calib_data.scale_value - 1).repeat(bs)
        # elif config.calib_data.scale_type == "range":
            # scale_range = config.calib_data.scale_range
            # left, right = scale_range[0], scale_range[1]
            # w = torch.rand(size=bs) * (right - left) + left - 1
        # else:
            # raise NotImplementedError(f"{opt.scale_type} is not supported!")
        # guidance_scale_embedding = pipe.get_guidance_scale_embedding(w)

        # run sampling
        # NOTE: LatentConsistencyModelPipeline.__call__ should be modified to support guidance_scale_embedding and return_trajectory input.
        #      StableDiffusionPipelineOutput should also be modified to support trajectory output.
        return_args = ['trajectory','text_emb','output']
        if config.model.model_type == 'sdxl':
            return_args.append('added_conds')
            cur_save_data["added_cond_kwargs"] = {}
        output = pipe(prompt=prompt, guidance_scale=config.calib_data.scale_value,
                    num_inference_steps=config.calib_data.n_steps, return_args=return_args)
        traj = output.return_args['trajectory']
        outputs = output.return_args['output']
        added_conds = output.return_args['added_conds']
        input_cond_emb = output.return_args['text_emb']

        # collect results
        cur_save_data["ts"] = torch.tensor(list(traj.keys())).unsqueeze(1).repeat(1, bs)
        cur_save_data["xs"] = torch.stack(list(traj.values()), dim=0)
        cur_save_data["outputs"] = torch.stack(list(outputs.values()), dim=0)
        cur_save_data["text_embs"] = input_cond_emb.unsqueeze(0).repeat(len(traj), 1, 1, 1)
        for k_ in added_conds.keys():
            cur_save_data["added_cond_kwargs"][k_] = torch.stack(added_conds[k_], dim=0)

        # cur_save_data["ws"] = w.unsqueeze(0).repeat(len(traj), 1)
        # cur_save_data["wcs"] = guidance_scale_embedding.unsqueeze(0).repeat(len(traj), 1, 1)
        for key in cur_save_data.keys():
            if not key in save_data.keys():
                save_data[key] = cur_save_data[key]
            else:
                if isinstance(save_data[key], torch.Tensor):
                    save_data[key] = torch.cat([save_data[key], cur_save_data[key]], dim=1)
                elif isinstance(save_data[key], dict):
                    # unpack the dict, and append each value for added_conds
                    for k_ in save_data[key]:
                        save_data[key][k_] = torch.cat([save_data[key][k_], cur_save_data[key][k_]], dim=1)
                else:
                    logger.info("Unexpcted type in save_data")
                    import ipdb; ipdb.set_trace()

        # save generated images (for debug)
        if opt.save_image_path is not None:
            os.makedirs(opt.save_image_path, exist_ok=True)
            for image in output.images:
                image.save(os.path.join(opt.save_image_path, f"{sample_id}.png"))
                sample_id += 1
        else:
            sample_id += bs
        # exit the loop

    # save calibration data
    torch.save(save_data, config.calib_data.path)

if __name__ == "__main__":
    main()
