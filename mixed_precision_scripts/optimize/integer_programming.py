import argparse, os, datetime, gc, yaml
from ortools.linear_solver import pywraplp
import yaml
import numpy as np
import yaml
import numpy as np


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def combine_dicts(dict1, dict2):
    combined_dict = dict1.copy()
    combined_dict.update(dict2)
    return combined_dict


def write_yaml_file(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def get_mean_bit(ssim2sqnr_ratio, k=1, average_bitwidth=8):
    '''
    Input the ratio of the average bitwidth of the two parts of the network and the average bit width of the entire network, 
    and output the average bit widths of the two parts of the network respectively.
    '''
    average_bitwidth_sqnr = average_bitwidth*(1+ssim2sqnr_ratio) / (k*ssim2sqnr_ratio+1)
    average_bitwidth_ssim = k*average_bitwidth_sqnr
    return average_bitwidth_sqnr, average_bitwidth_ssim


def get_mixed_precision_config_weight(stage, ratio_config, sensitivity_config, mean_bit):

    sensitivity_config_tmp = {}
    for key, value in sensitivity_config.items():
        if stage == 'ssim':
            if 'ff' in key or 'attn2' in key:
                sensitivity_config_tmp[key] = value
        elif stage == 'sqnr':
            if not 'ff' in key and not 'attn2' in key:
                sensitivity_config_tmp[key] = value
    
    sensitivity_config = sensitivity_config_tmp
    print(len(sensitivity_config))

    if stage == 'ssim':
        b_values = [2, 4, 8]    # bit width candidate 
    elif stage == 'sqnr':
        b_values = [4, 8]    # bit width candidate 


    # creat a solver
    mean_bit = float(mean_bit)
    solver = pywraplp.Solver.CreateSolver('SCIP')

    w = ratio_config # weight ratio dict
    s = sensitivity_config   # sensitivity dict
    c = {}

    # 计compute the para size
    import numpy as np
    intensity = 0
    intensity = sum(w[name] for name, ssim in s.items())


    # create the variable
    for name, ssim in s.items():
        for b in b_values:
            c[(name, b)] = solver.BoolVar('c_' + name + '_' + str(b))
    print("Number of variables =", solver.NumVariables())


    # create the constrains
    for name, ssim in s.items():
        solver.Add(sum(c[(name, b)] for b in b_values) == 1)
    print("Number of constraints =", solver.NumConstraints())

    solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,ssim in s.items()) for b in b_values) >= (mean_bit - 0.02) * intensity)
    solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,ssim in s.items()) for b in b_values) <= (mean_bit + 0.02) * intensity)
    print("Number of constraints =", solver.NumConstraints())


    import math
    objective = solver.Objective()
    for name, ssim in s.items():
        # print(name,ssim)
        for b in b_values:
            objective.SetCoefficient(c[(name, b)], ssim[int(math.log2(b)-1)])  # s_{i,b}是c_{i,b}的系数
    objective.SetMaximization()


    # solve the problem
    solver.Solve()

    # generate the bit width config for weight and act seperatly
    print('Solution:')
    solution_dict = {}
    for name, ssim in s.items():
        solution_dict[name] = {}
        solution_dict[name] = 0
        solution_dict[name] = 0
        for b in b_values:
            if c[(name, b)].solution_value() > 0:
                solution_dict[name] = b

    return solution_dict


def get_mixed_precision_config_act(stage, ratio_config, sensitivity_config, mean_bit, act_sensitivity_1):

    layer_filtered_ratio = {}
    for key, value in ratio_config.items():
        if key in act_sensitivity_1:
            layer_filtered_ratio[key] = value


    layer_filtered_para = sum(para for name, para in layer_filtered_ratio.items())
    print(layer_filtered_para)


    sensitivity_config_tmp = {}
    for key, value in sensitivity_config.items():
        if stage == 'ssim':
            if 'ff' in key or 'attn2' in key:
                if not key in act_sensitivity_1:
                    sensitivity_config_tmp[key] = value
        elif stage == 'sqnr':
            if not 'ff' in key and not 'attn2' in key:
                if not key in act_sensitivity_sqnr_1:
                    sensitivity_config_tmp[key] = value


    sensitivity_config = sensitivity_config_tmp
    print(len(sensitivity_config))

    # create a solver
    mean_bit = float(mean_bit)
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # define the variable
    w = ratio_config # weight ratio dict
    s = sensitivity_config   # sensitivity dict
    b_values = [4, 8]    # bit width candidate 
    c = {}

    intensity = sum(w[name] for name, ssim in s.items())
    print(intensity)


    for name, ssim in s.items():
        for b in b_values:
            c[(name, b)] = solver.BoolVar('c_' + name + '_' + str(b))
    print("Number of variables =", solver.NumVariables())


    for name, ssim in s.items():
        solver.Add(sum(c[(name, b)] for b in b_values) == 1)
    print("Number of constraints =", solver.NumConstraints())

    solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,ssim in s.items()) for b in b_values) >= (mean_bit - 0.02) * intensity)
    solver.Add(sum(sum(c[(i, b)] * b * w[i] for i,ssim in s.items()) for b in b_values) <= (mean_bit + 0.02) * intensity)
    print("Number of constraints =", solver.NumConstraints())

    import math
    objective = solver.Objective()
    for name, ssim in s.items():
        # print(name,ssim)
        for b in b_values:
            objective.SetCoefficient(c[(name, b)], ssim[int(math.log2(b)-1)]) 
    objective.SetMaximization()


    solver.Solve()

    # generate the bit width config for weight and act seperatly
    print('Solution:')
    solution_dict = {}
    for name, ssim in s.items():
        solution_dict[name] = {}
        solution_dict[name] = 0
        solution_dict[name] = 0
        for b in b_values:
            if c[(name, b)].solution_value() > 0:
                solution_dict[name] = b

    return solution_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_precision_type", required=True,
        type=str,
        help="perform integer programming for weight or act"
    )
    parser.add_argument(
        "--sensitivity_ssim", required=True,
        type=str,
        help="path for sensitivity list based on ssim",
    )
    parser.add_argument(
        "--sensitivity_sqnr", required=True,
        type=str,
        help="path for sensitivity list based on sqnr"
    )
    parser.add_argument(
        "--para_size_config", required=True,
        type=str,
        help="path for the config of parameter size"
    )
    parser.add_argument(
        "--mixed_precision_config", required=True,
        type=str,
        help="path for the output config"
    )
    parser.add_argument(
        "--target_bitwidth", required=True,
        type=float,
        help="the average bitwidth of weight or act"
    )
    opt = parser.parse_args()


    if opt.mixed_precision_type == 'weight':

        os.makedirs(opt.mixed_precision_config, exist_ok=True)

        # config_path_weight_ssim ='../sensitivity_log/sdxl_turbo/weight/ssim/bs32_split_ssim_weight/sensitivity.yaml'
        with open(opt.sensitivity_ssim, 'r') as file:
            layer_config_ssim = yaml.safe_load(file)

        # config_path_weight_sqnr ='../sensitivity_log/sdxl_turbo/weight/sqnr/bs32_split_sqnr_weight/sensitivity.yaml'
        with open(opt.sensitivity_sqnr, 'r') as file:
            layer_config_sqnr = yaml.safe_load(file)


        # config_path = f'./tensor_ratio/sdxl_turbo/weight_ratio_config.yaml'
        with open(opt.para_size_config, 'r') as file:
            ratio_config = yaml.safe_load(file)

        num_para_ff_attn2 = 0
        num_para_non_ff_attn2 = 0 
        for name, value in ratio_config.items(): 
            if 'ff' in name or 'attn2' in name:
                num_para_ff_attn2 = num_para_ff_attn2 + value
            else:
                num_para_non_ff_attn2 = num_para_non_ff_attn2 + value

        ssim2sqnr_ratio = num_para_ff_attn2 / num_para_non_ff_attn2

        ssim_weight = ssim2sqnr_ratio
        sqnr_weight = 1

        # choose a average bitwidth we want
        # target_bitwidth = 5

        layer_ratios = np.linspace(0.46,1.36,10)
        # the ratio of the average bitwidth between content-related layers and quality-related layers 

        bitwidth_list = np.linspace(opt.target_bitwidth-0.3, opt.target_bitwidth, 10)

        for average_bitwidth in bitwidth_list:
            for k in layer_ratios:
                average_bitwidth_sqnr, average_bitwidth_ssim = get_mean_bit(ssim2sqnr_ratio, k, average_bitwidth=average_bitwidth)
                if average_bitwidth_sqnr <4 or average_bitwidth_sqnr>8 or average_bitwidth_ssim < 2 or average_bitwidth_ssim>8:
                    continue
                print(average_bitwidth_sqnr, average_bitwidth_ssim)
                
                mixed_precision_ssim = get_mixed_precision_config_weight(stage='ssim', ratio_config=ratio_config, sensitivity_config=layer_config_ssim, mean_bit=average_bitwidth_ssim)
                mixed_precision_sqnr = get_mixed_precision_config_weight(stage='sqnr', ratio_config=ratio_config, sensitivity_config=layer_config_sqnr, mean_bit=average_bitwidth_sqnr)
                
                weight_config = combine_dicts(mixed_precision_sqnr, mixed_precision_ssim)
                # print(len(weight_config))
                
                k_value = format(k,'.2f')
                bit_value = format(average_bitwidth,'.2f')
                
                # please replace <your_path> with a existing path 
                # file_path_weight = f"<your_path>/weight_{(bit_value)}_{(k_value)}.yaml"
                write_yaml_file(file_path=os.path.join(opt.mixed_precision_config, f"weight_{(bit_value)}_{(k_value)}.yaml"), data=weight_config)


    elif opt.mixed_precision_type == 'act':

        os.makedirs(opt.mixed_precision_config, exist_ok=True)
        
        act_sensitivity_ssim_5 = ['model.down_blocks.2.attentions.1.transformer_blocks.6.ff.net.2', 'model.up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2', 
        'model.down_blocks.2.attentions.1.transformer_blocks.8.ff.net.2', 'model.down_blocks.2.attentions.1.transformer_blocks.5.ff.net.2', 
        'model.down_blocks.2.attentions.1.transformer_blocks.7.ff.net.2', 'model.down_blocks.2.attentions.1.transformer_blocks.4.ff.net.2', 
        'model.up_blocks.0.attentions.0.transformer_blocks.4.ff.net.2', 'model.up_blocks.0.attentions.0.transformer_blocks.5.ff.net.2', 
        'model.up_blocks.0.attentions.0.transformer_blocks.2.ff.net.2', 'model.up_blocks.0.attentions.0.transformer_blocks.6.ff.net.2', 
        'model.up_blocks.0.attentions.0.transformer_blocks.3.ff.net.2', 'model.down_blocks.2.attentions.1.transformer_blocks.9.ff.net.2', 
        'model.down_blocks.2.attentions.1.transformer_blocks.3.ff.net.2', 'model.down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_v', 
        'model.up_blocks.0.attentions.0.transformer_blocks.7.ff.net.2', 'model.up_blocks.0.attentions.1.transformer_blocks.3.ff.net.2', 
        'model.down_blocks.2.attentions.1.transformer_blocks.1.ff.net.2', 'model.up_blocks.0.attentions.0.transformer_blocks.1.ff.net.2', 
        'model.down_blocks.2.attentions.1.transformer_blocks.2.ff.net.2', 
        'model.down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_k', 'model.up_blocks.0.attentions.1.transformer_blocks.4.ff.net.2']
        act_sensitivity_ssim_1 = act_sensitivity_ssim_5[0:5]

        act_sensitivity_sqnr_5 = ['model.conv_in', 'model.conv_out', 'model.down_blocks.0.resnets.0.conv2', 
        'model.up_blocks.2.resnets.2.conv_shortcut', 'model.down_blocks.2.attentions.1.proj_in', 
        'model.up_blocks.2.resnets.2.conv2', 'model.down_blocks.0.resnets.1.conv2', 'model.up_blocks.2.resnets.1.conv_shortcut', 
        'model.down_blocks.1.downsamplers.0.conv', 'model.down_blocks.0.downsamplers.0.conv', 'model.down_blocks.2.resnets.1.time_emb_proj', 
        'model.down_blocks.2.resnets.0.conv_shortcut', 'model.down_blocks.2.resnets.0.conv2', 'model.down_blocks.2.attentions.0.proj_out', 
        'model.down_blocks.2.resnets.0.time_emb_proj', 'model.up_blocks.0.attentions.0.proj_out', 'model.up_blocks.0.attentions.0.proj_in', 
        'model.up_blocks.0.attentions.2.proj_out', 'model.down_blocks.1.resnets.0.conv_shortcut', 'model.down_blocks.1.attentions.1.proj_out', 
        'model.mid_block.attentions.0.proj_in']
        act_sensitivity_sqnr_1 = act_sensitivity_sqnr_5[0:4]


        # config_path_weight_ssim ='../sensitivity_log/sdxl_turbo/act/ssim/bs32_split_ssim_act/sensitivity.yaml'
        with open(opt.sensitivity_ssim, 'r') as file:
            layer_config_ssim = yaml.safe_load(file)

        # config_path_weight_ssim ='../sensitivity_log/sdxl_turbo/act/sqnr/bs32_split_sqnr_act/sensitivity.yaml'
        with open(opt.sensitivity_sqnr, 'r') as file:
            layer_config_sqnr = yaml.safe_load(file)


        # config_path = f'./tensor_ratio/sdxl_turbo/act_ratio_config.yaml'

        with open(opt.para_size_config, 'r') as file:
            ratio_config = yaml.safe_load(file)

        num_para_ff_attn2 = 0
        num_para_non_ff_attn2 = 0 
        for name, value in ratio_config.items(): 
            if 'ff' in name or 'attn2' in name:
                num_para_ff_attn2 = num_para_ff_attn2 + value
            else:
                num_para_non_ff_attn2 = num_para_non_ff_attn2 + value

        ssim2sqnr_ratio = num_para_ff_attn2 / num_para_non_ff_attn2

        ssim_weight = ssim2sqnr_ratio
        sqnr_weight = 1

        layer_ratios = np.linspace(0.94,1.09,20)
        # the ratio of the average bitwidth between content related layers and quality related layers

        bitwidth_list = np.linspace(opt.target_bitwidth-0.3, opt.target_bitwidth, 10)

        for average_bitwidth in bitwidth_list:
            for k in layer_ratios:
                average_bitwidth_sqnr, average_bitwidth_ssim = get_mean_bit(ssim2sqnr_ratio, k, average_bitwidth=average_bitwidth)
                if average_bitwidth_sqnr <4 or average_bitwidth_sqnr>8 or average_bitwidth_ssim < 2 or average_bitwidth_ssim>8:
                    continue
                print(average_bitwidth_sqnr, average_bitwidth_ssim)
                
                mixed_precision_ssim = get_mixed_precision_config_act(stage='ssim', ratio_config=ratio_config, sensitivity_config=layer_config_ssim, mean_bit=average_bitwidth_ssim, act_sensitivity_1=act_sensitivity_ssim_1)
                mixed_precision_sqnr = get_mixed_precision_config_act(stage='sqnr', ratio_config=ratio_config, sensitivity_config=layer_config_sqnr, mean_bit=average_bitwidth_sqnr, act_sensitivity_1=act_sensitivity_sqnr_1)
                
                weight_config = combine_dicts(mixed_precision_sqnr, mixed_precision_ssim)
                # print(len(weight_config))
                
                k_value = format(k,'.2f')
                bit_value = format(average_bitwidth,'.2f')
                
                # please replace <your_path> with a existing path 
                # file_path_weight = f"<your_path>/act_{(bit_value)}_{(k_value)}.yaml"
                # write_yaml_file(file_path=file_path_weight, data=weight_config)
                write_yaml_file(file_path=os.path.join(opt.mixed_precision_config, f"act_{(bit_value)}_{(k_value)}.yaml"), data=weight_config)
