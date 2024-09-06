import argparse
from diffusers import StableDiffusionXLPipeline
import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, PlaceholderObserver
from mixdq_extension.nn.Linear import QuantizedLinear
from mixdq_extension.nn.Conv2d import QuantizedConv2d
from pytorch_lightning import seed_everything
import logging
import functools
import threading
# import torch.cuda.nvtx as nvtx

def nvtx_decorator(forward_func, name=None):
    def wrapper(self, *args, **kwargs):

        if name is not None:
            name_ = name
        else:
            name_ = f"Forward {self.__class__.__name__}"
        
        print('logging nvtx for layer {}'.format(name_))
        # with torch.cuda.profiler.profile():
        torch.cuda.nvtx.range_push(name_)
        # with torch.autograd.profiler.emit_nvtx():    # annotate the pytorch function calls
        result = forward_func(self, *args, **kwargs)
        torch.cuda.nvtx.range_pop()
        return result
    return wrapper

def create_pipeline(args):
    """Load the model pipeline from huggingface"""
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16, variant="fp16"
    )
    return pipeline

def register_qconfig_from_input_files(unet, args, bos, bos_dict):
    """Load the quantization bitwidth configurations of each layer
    from the config files, and set the qconfig parameters of each 
    layer."""
    import yaml

    bw_to_dtype = {
        8: torch.qint8,
        4: torch.quint4x2,
        2: torch.quint4x2, # !!!TODO: 2 is not supported, treat as 4
    }

    # load weight bits
    with open(args.w_config, 'r') as input_file:
        mod_name_to_weight_width = yaml.safe_load(input_file)

    # filter 'model.' from all names 
    def filter_mod_name_prefix(mod_name):
        if 'model.' in mod_name:
            pos = mod_name.index('model.')
            mod_name = mod_name[pos + 6:]
        return mod_name
    
    mod_name_to_weight_width_copy = {}
    for mod_name, bit_width in mod_name_to_weight_width.items():
        new_name = filter_mod_name_prefix(mod_name)
        mod_name_to_weight_width_copy[new_name] = bit_width
    mod_name_to_weight_width = mod_name_to_weight_width_copy
    
    # add qconfig to all modules whose name are in the yaml
    mod_name_to_weight_width_copy = mod_name_to_weight_width
    for name, mod in unet.named_modules():
        if name in mod_name_to_weight_width:
            assert not hasattr(mod, 'qconfig')
            # get the corresponding bit-width of the layer
            w_bitwidth = mod_name_to_weight_width[name]  
            w_dtype = bw_to_dtype[w_bitwidth]
            # get the statistic info in the tensor
            act_preprocess = PlaceholderObserver.with_args(dtype=torch.float16)
            weight_process = PlaceholderObserver.with_args(dtype=w_dtype)
            mod.qconfig = \
                QConfig(activation=act_preprocess, weight=weight_process)
            
            # init some parameters for each unquantized module
            mod.module_name = name  # set module name for each module
            # record the bit_width of the weight
            mod.w_bit = mod_name_to_weight_width[name]  
            if 'attn2' in name:
                if 'to_k' in name or 'to_v' in name:
                    mod.bos = bos  # set bos for corss attn layers
                    mod.bos_pre_computed = bos_dict[name]

            del mod_name_to_weight_width_copy[name]
    # check if there is any module not in the unet
    if len(mod_name_to_weight_width_copy):
        for name in mod_name_to_weight_width_copy.keys():
            print(f"{name} not found in UNet!")
        raise RuntimeError("Not all keys in weight yaml map to a module in "\
                           "UNet.")
    
    # load activation bits
    if args.a_config is None:
        return
    
    with open(args.a_config, 'r') as input_file:
        mod_name_to_act_width = yaml.safe_load(input_file)
    # filter 'model.' from all names 
    mod_name_to_act_width_copy = {}
    for mod_name, bit_width in mod_name_to_act_width.items():
        new_name = filter_mod_name_prefix(mod_name)
        mod_name_to_act_width_copy[new_name] = bit_width
    mod_name_to_act_width = mod_name_to_act_width_copy
    
    # add qconfig to all modules whose name are in the yaml
    mod_name_to_act_width_copy = mod_name_to_act_width
    for name, mod in unet.named_modules():
        if name in mod_name_to_act_width:
            a_bitwidth = mod_name_to_act_width[name]
            a_dtype = bw_to_dtype[a_bitwidth]
            act_preprocess = PlaceholderObserver.with_args(dtype=a_dtype)
            if hasattr(mod, 'qconfig') and mod.qconfig:
                assert isinstance(mod.qconfig, QConfig)
                mod.qconfig = QConfig(weight=mod.qconfig.weight, 
                                      activation=act_preprocess)
            else:
                weight_process = PlaceholderObserver.with_args(
                    dtype=torch.float16)
                mod.qconfig = QConfig(activation=act_preprocess,
                                      weight=weight_process)
            
            # init some parameters for each unquantized module
            # record the bit_width of the act
            mod.a_bit = mod_name_to_act_width[name]  

            del mod_name_to_act_width_copy[name]
    # check if there is any module not in the unet
    if len(mod_name_to_act_width_copy):
        for name in mod_name_to_act_width_copy.keys():
            print(f"{name} not found in UNet!")
        raise RuntimeError("Not all keys in act yaml map to a module in "\
                           "UNet.")


def convert_to_quantized(unet, ckpt):
    from quantize import convert
    convert(unet,
            mapping={
                nn.Linear: QuantizedLinear,
                nn.Conv2d: QuantizedConv2d,
                },
            inplace=True,
            ckpt = ckpt)
    # print("unet after quantization")
    # print(unet)

def quantize_unet(unet, args, ckpt, bos, bos_dict):
    register_qconfig_from_input_files(unet, args, bos=bos, bos_dict=bos_dict)
    convert_to_quantized(unet, ckpt)

def compile_opt(pipeline, args):
    if args.run_pipeline:
        pipeline = pipeline.to('cuda')
    else:
        pipeline.unet = pipeline.unet.to("cuda")

    if args.compile_tool == 'pt2':
        pipeline.unet = torch.compile(pipeline.unet)
    elif args.compile_tool == 'onediff':
        from onediff.infer_compiler import oneflow_compile
        pipeline.unet = oneflow_compile(pipeline.unet)
    elif args.compile_tool == 'sfast':
        # apply stable-fast
        from sfast.compilers.diffusion_pipeline_compiler import compile_unet
        from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig

        sfast_config = CompilationConfig.Default()
        sfast_config.enable_triton = True
        sfast_config.enable_cuda_graph = False
        sfast_config.enable_jit = True

        pipeline.unet = compile_unet(pipeline.unet, sfast_config)
    else:
        print(f"Unknown compile tool {args.compile_tool}")
    return pipeline

def cuda_graph_opt(unet, args):

    def hash_arg(arg):
        if isinstance(arg, torch.Tensor):
            arg_device = arg.device
            arg_device_type = arg_device.type
            return (arg_device_type, arg_device.index, arg.dtype, arg.shape,
                    arg.item()
                    if arg_device_type == 'cpu' and arg.numel() == 1 else None)
        if isinstance(arg, (str, int, float, bytes, bool)):
            return arg
        if isinstance(arg, (tuple, list)):
            return tuple(map(hash_arg, arg))
        if isinstance(arg, dict):
            return tuple(
                sorted(((hash_arg(k), hash_arg(v)) for k, v in arg.items()),
                    key=lambda x: x[0]))
        return type(arg)
    
    def copy_args(arg):
        if isinstance(arg, tuple):
            return tuple(map(copy_args, arg))
        if isinstance(arg, list):
            return list(map(copy_args, arg))
        if isinstance(arg, dict):
            d_ = dict()
            for k, v in arg.items():
                d_[k] = copy_args(v)
            return d_
        if isinstance(arg, (str, int, float, bytes, bool)):
            return arg
        if isinstance(arg, torch.Tensor):
            return arg.detach().clone()
        if arg is None:
            return None
        raise ValueError(f"Unknown argument type {arg}")
    
    def copy_args_to_dest(dest_arg, src_arg):
        if isinstance(src_arg, (tuple, list)):
            for i, x in enumerate(src_arg):
                copy_args_to_dest(dest_arg[i], x)
        if isinstance(src_arg, dict):
            for k, v in src_arg.items():
                copy_args_to_dest(dest_arg[k], v)
        if isinstance(src_arg, (str, int, float, bytes, bool)) \
            or src_arg is None:
            pass # should be the same with dest_arg
        if isinstance(src_arg, torch.Tensor):
            dest_arg.copy_(src_arg)
    
    def create_forward_with_cuda_graph(net):
        lock = threading.Lock()
        cached_cuda_graphs = {}

        wrapped = net.forward

        @functools.wraps(wrapped)
        def forward_with_cuda_graph(*args, **kwargs):
            key = (hash_arg(args), hash_arg(kwargs))
            if not (key in cached_cuda_graphs):
                with lock:
                    if not (key in cached_cuda_graphs):
                        args_, kwargs_ = copy_args((args, kwargs))

                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())

                        with torch.no_grad():
                            with torch.cuda.stream(s):
                                for _ in range(3):
                                    static_output = wrapped(*args_, **kwargs_)

                        g = torch.cuda.CUDAGraph()
                        with torch.no_grad():
                            with torch.cuda.graph(g):
                                static_output = wrapped(*args_, **kwargs_)

                        cached_cuda_graphs[key] = (
                            (args_, kwargs_),
                            g,
                            static_output
                        )
            static_inputs, graph, static_output = cached_cuda_graphs[key]
            args_, kwargs_ = static_inputs

            copy_args_to_dest((args_, kwargs_), (args, kwargs))
            graph.replay()
            return static_output

        forward_with_cuda_graph.__self__ = net
        forward_with_cuda_graph._cached = cached_cuda_graphs
        return forward_with_cuda_graph

    unet.forward = create_forward_with_cuda_graph(unet)

    # # change static parts of unet to cuda graph
    # for mod in unet.down_blocks:
    #     mod.forward = create_forward_with_cuda_graph(mod)
    # if unet.mid_block is not None:
    #     unet.mid_block.forward = create_forward_with_cuda_graph(unet.mid_block)
    # for mod in unet.up_blocks:
    #     mod.forward = create_forward_with_cuda_graph(mod)
    return unet


def parse_args():

    parser = argparse.ArgumentParser(
        description="Script to run diffusers SDXL"
    )

    parser.add_argument("--prompt", "-p", type=str,
        default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.")
    parser.add_argument("--model", type=str, default="/home/zhaotianchen/local_ckpt/")
    parser.add_argument("--ckpt", type=str, default="./output/new_ckpt.pth")  # quant_para_fp16.pth
    parser.add_argument("--output_type", type=str, default="latent")
    parser.add_argument("--w_config", type=str, default="./cfgs/weight/weight_8.00.yaml")
    parser.add_argument("--a_config", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--memory_snapshot_name", type=str, default=None)
    parser.add_argument("--run_pipeline", action='store_true')
    parser.add_argument("--profile", action='store_true')
    parser.add_argument("--profile_tool", type=str, default="torch_profiler", choices=['torch_profiler', 'nsys'])
    parser.add_argument("--compile", action='store_true')
    parser.add_argument("--compile_tool", type=str, default="onediff", choices=['onediff', 'pt2', 'sfast'])
    parser.add_argument("--cuda_graph_only", action="store_true")
    parser.add_argument("--bos", action='store_true')

    args = parser.parse_args()
    return args


def make_memory_friendly(bytes):

    MBs = bytes / (1024*1024)

    B = bytes % 1024
    bytes = bytes // 1024
    kB = bytes % 1024
    bytes = bytes // 1024
    MB = bytes % 1024
    GB = bytes // 1024

    return f"{GB} G {MB} M {B} {kB} K {B} Bytes ({MBs} MBs)"


def run(pipeline, args):
    if args.run_pipeline:
        pipeline.to('cuda')
    else:
        pipeline.unet.to("cuda")

    model_memory = torch.cuda.memory_allocated()
    print("Static (weights) memory usage:", make_memory_friendly(model_memory))


    # start = time.time()
    if args.run_pipeline:
        def run_once():
            latents = pipeline(prompt=[args.prompt]*args.batch_size, 
                            guidance_scale=0.0, 
                            num_inference_steps=1, 
                            output_type=args.output_type).images[0]      
            return latents
    else:
        sample_shape = (
            args.batch_size * 1, 
            pipeline.unet.config.in_channels,
            pipeline.unet.config.sample_size,
            pipeline.unet.config.sample_size,
        )

        encoder_embedding_shape = (
            args.batch_size * 1,
            77, # just an example,
            2048,
        )

        device=torch.device('cuda')
        example_sample = torch.rand(*sample_shape, device=device, 
                                    dtype=torch.float16)
        example_embedding = torch.rand(*encoder_embedding_shape, 
                                    device=device, dtype=torch.float16)
        timestep = torch.tensor(999., device=device)
        text_embeds = torch.rand(args.batch_size, 1280, device=device, 
                                dtype=torch.float16)
        time_ids = torch.tensor([[512.,512.,0.,0.,512.,512.]], dtype=torch.float16,
                                device=device)
        time_ids = torch.concat([time_ids] * args.batch_size)

        def run_once():
            with torch.no_grad():
                latents = pipeline.unet(sample=example_sample,
                                    timestep=timestep,
                                    encoder_hidden_states=example_embedding,
                                    added_cond_kwargs={
                                        'time_ids': time_ids,
                                        'text_embeds': text_embeds
                                    },
                                    return_dict=False)[0]
                return latents

        def layers_nvtx_annotate():

            # ------------------------ annotate some layers ------------------------------
            modules_to_add_nvtx = {
                    'conv_320_320': pipeline.unet.down_blocks[0].resnets[0].conv1,    # input_shape: [1, 320, 64, 64]
                    'conv_1280_1280': pipeline.unet.down_blocks[2].resnets[0].conv2,  # input_shape: [1, 1280, 16, 16]
                    'conv_2560_1280': pipeline.unet.up_blocks[0].resnets[0].conv1,    # input_shape: [1, 2560, 16, 16]
                    'linear_640_640': pipeline.unet.down_blocks[1].attentions[0].transformer_blocks[0].attn1.to_q,   # input_shape: [1024, 640]
                    'linear_1280_1280': pipeline.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn1.to_q, # input_shape: [256, 1024]
                    'linear_2048_1280': pipeline.unet.down_blocks[2].attentions[0].transformer_blocks[0].attn2.to_k, # input_shape: [77, 2048]
                    }

            # ------------------------ annotate blocks ------------------------------
            for i_down_block, down_block_ in enumerate(pipeline.unet.down_blocks):
                for i_resnet, resnet_ in enumerate(down_block_.resnets):
                    modules_to_add_nvtx[f'down_block_{i_down_block}_resnet_{i_resnet}'] = resnet_
                if hasattr(down_block_, 'attentions'):
                    for i_transformer, transformer_ in enumerate(down_block_.attentions):
                        modules_to_add_nvtx[f'down_block_{i_down_block}_transformers_{i_transformer}'] = transformer_

            for i_up_block, up_block_ in enumerate(pipeline.unet.up_blocks):
                for i_resnet, resnet_ in enumerate(up_block_.resnets):
                    modules_to_add_nvtx[f'up_block_{i_down_block}_resnet_{i_resnet}'] = resnet_
                if hasattr(up_block_, 'attentions'):
                    for i_transformer, transformer_ in enumerate(up_block_.attentions):
                        modules_to_add_nvtx[f'mid_block_{i_up_block}_transformers_{i_transformer}'] = transformer_


            for i_resnet, resnet_ in enumerate(pipeline.unet.mid_block.resnets):
                modules_to_add_nvtx[f'mid_block_resnet_{i_resnet}'] = resnet_
            for i_transformer, transformer_ in enumerate(pipeline.unet.mid_block.attentions[0].transformer_blocks):
                modules_to_add_nvtx[f'mid_block_transformers_{i_transformer}'] = transformer_


            # ------------------------ annotate all modules ------------------------------
            # causes bug `wrapper() missing 1 required positional argument: 'self'`, some blocks donot have a forward
            # for name_, module_ in pipeline.unet.named_modules():
                # if name_ != '':
                    # modules_to_add_nvtx[name_] = module_

            # add the nvtx wrapper
            for name_, module_ in modules_to_add_nvtx.items():
                module_.forward = nvtx_decorator(module_.forward.__get__(module_, type(module_)), name=name_)

    latents = run_once()
    latents = run_once()

    if args.run_pipeline:
        if args.quantize:
            latents.save('result_00.png')
        else:
            latents.save('result.png')

    if args.cuda_graph_only:
        if args.compile:
            logging.warning("--compile and --cuda_graph_only should not be used"
                            " together, cuda_graph_only is ignored.")
        else:
            pipeline.unet = cuda_graph_opt(pipeline.unet, args)
            latents = run_once()
            if args.run_pipeline:
                latents.save('result_01.png')
            latents = run_once()
            if args.run_pipeline:
                latents.save('result_02.png')
            
    peak_memory = torch.cuda.max_memory_allocated()
    print("Dynamic (acts) memory usage:", 
          make_memory_friendly(peak_memory - model_memory))
    print("Peak (total) memory usage:", make_memory_friendly(peak_memory))
    
    if args.memory_snapshot_name is not None:
        torch.cuda.memory._dump_snapshot(args.memory_snapshot_name)
    
    if args.profile:
        if args.profile_tool == "nsys":
            torch.cuda.cudart().cudaProfilerStart()
            layers_nvtx_annotate() # add some nvtx range for specific layers
            for iter in range(3):
                torch.cuda.nvtx.range_push(f"iter_{iter}")
                run_once()
                torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

        elif args.profile_tool == "torch_profiler":
            with torch.profiler.profile(
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/sdxl'),
                with_stack=True
            ):
                for _ in range(3):
                    run_once()
        else:
            print(f"Unknown profile_tool {args.profile_tool}, use nsys or torch_profiler")

    # if args.output_type=="pil":
        # image = latents
        # image.save("result.png")
    return latents


if __name__ == "__main__":
    seed_everything(42)

    args = parse_args()
    pipeline = create_pipeline(args)
    
    if args.quantize:
        ckpt = torch.load(args.ckpt, map_location = 'cpu')

        # contain the tensor obtained from pre-computed first_token@weight
        bos_dict = torch.load("./bos_pre_computed.pt", map_location = 'cpu')  

        quantize_unet(pipeline.unet, args, ckpt, bos=args.bos, bos_dict=bos_dict)
    
    if args.compile:
        pipeline = compile_opt(pipeline, args)

    if args.memory_snapshot_name is not None:
        torch.cuda.memory._record_memory_history()


    print("start inference!")
    run(pipeline, args)


    # to pre-computed the bos
    # bos_dict = {}
    # for name, module in pipeline.unet.named_modules():
    #     if isinstance(module, QuantizedLinear):
    #         if hasattr(module, 'out_0'):
    #             print(module.module_name)
    #             bos_dict[module.module_name] = module.out_0

    # torch.save(bos_dict, 'bos_pre_computed.pt')

