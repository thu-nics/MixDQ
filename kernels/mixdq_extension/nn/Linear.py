from typing import Dict
import torch.nn as nn
import torch
from torch.ao.quantization import QConfig
import torch.nn.functional as F
from .utils import (dtype_to_bw, create_qparams_from_dtype)
from mixdq_extension.op.quant import quantize_per_tensor, quantize_per_tensor_vectorized
from mixdq_extension.op.qlinear import qlinear
import logging

all = [
    'QuantizedLinear'
]

quant_op = quantize_per_tensor_vectorized

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        device=None, w_qparams=None, a_qparams=None, module_name=None) -> None:
        
        super().__init__()
        self.module_name = module_name
        # print(module_name)
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.valid_for_acceleration = (
            w_qparams is not None and \
            a_qparams is not None and \
            w_qparams.dtype in [torch.qint8, torch.quint8] and \
            a_qparams.dtype in [torch.qint8, torch.quint8] and \
            w_qparams.qscheme == torch.per_channel_affine and \
            a_qparams.qscheme == torch.per_tensor_affine and \
            torch.all(w_qparams.zero_points == 0.0).item()
        )
        if self.valid_for_acceleration and (
            self.in_features % 4 != 0 or self.out_features % 4 != 0):
            logging.warning("Linear layer with in_features = "
                    f"{self.in_features} and out_features = "
                    f"{self.out_features} cannot use quantized kernel due to "
                    "misalignment. Falling back to FP kernels")
            self.valid_for_acceleration = False
        
        if self.valid_for_acceleration:
            self.register_buffer("weight_scales", 
                                 w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", 
                                 w_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales", 
                                 a_qparams.scales.to(device).float())
            self.register_buffer("act_zero_points", 
                                 a_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales_inv", 1 / self.act_scales)
        
    
    @classmethod
    def from_float(cls, float_mod, split=0, ckpt=None):
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, 
                                                            QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device=float_mod.weight.device

        w_qparams, w_qparams_0 = create_qparams_from_dtype(dtype=w_dtype, 
                                                device=device,
                                                is_channel_wise=True,
                                                num_kernels=num_kernels,
                                                ckpt=ckpt,
                                                module_name=\
                                                    float_mod.module_name,
                                                quant_type='weight',
                                                bit_width=float_mod.w_bit,
                                                split=split)
                                              

        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype

        if hasattr(float_mod, 'a_bit'):
            a_qparams, a_qparams_0 = create_qparams_from_dtype(dtype=act_dtype, 
                                                    device=device,
                                                    is_channel_wise=False,
                                                    num_kernels=num_kernels,
                                                    ckpt=ckpt,
                                                    module_name=\
                                                        float_mod.module_name,
                                                    quant_type='act',
                                                    bit_width=float_mod.a_bit,
                                                    split=split)
        else:
            a_qparams = None
            a_qparams_0 = None

        new_mod = cls(float_mod.in_features,
                      float_mod.out_features,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,
                      w_qparams=w_qparams,
                      a_qparams=a_qparams,
                      module_name = float_mod.module_name,
                      )

        weight = float_mod.weight.detach()

        if 'attn2' in float_mod.module_name:
            if 'to_k' in float_mod.module_name or \
                'to_v' in float_mod.module_name:
                new_mod.bos = float_mod.bos
                new_mod.register_buffer("bos_pre_computed", float_mod.bos_pre_computed)
                # the input of the org_weight is key_first_token
                # new_mod.register_buffer("org_weight", weight)

        if new_mod.valid_for_acceleration:
            weight_int = torch.quantize_per_channel(
                weight.float(), 
                new_mod.weight_scales, 
                new_mod.weight_zero_points,
                axis=w_qparams.axis, 
                dtype=w_qparams.dtype).int_repr()

            new_mod.register_buffer("weight_int", weight_int)

            # auxiliary structure, used to quickly compute act_zp @ weight
            weight_sum_by_input_channels = weight_int.float().sum(dim=1)
            new_mod.register_buffer("weight_sum_by_input_channels", 
                                    weight_sum_by_input_channels)
            new_mod.register_buffer("scale", 
                                    new_mod.weight_scales*new_mod.act_scales)
            new_mod.register_buffer("bias0", 
                weight_sum_by_input_channels * new_mod.act_zero_points)
        else:
            new_mod.register_buffer("weight", weight)
        if float_mod.bias is not None:
            bias = float_mod.bias.detach()
            new_mod.register_buffer("bias", bias)
        else:
            new_mod.bias = None
        return new_mod
    
    def _get_name(self):
        if self.valid_for_acceleration:
            return "QuantizedLinearW8A8"
        return "QuantizedLinearFPFallback"
    
    def forward_fallback(self, x):
        weight_recovered = self.weight_int.float()* self.weight_scales[:, None]
        weight_recovered = weight_recovered.to(x.dtype)
        return F.linear(x, 
                        weight_recovered, 
                        self.bias.to(x.dtype) if self.bias is not None else None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.valid_for_acceleration:
            return F.linear(x, self.weight, self.bias)

        if not x.dtype == torch.float16:
            return self.forward_fallback(x)

        if not hasattr(self, 'bos') or not self.bos:
            x_int = quant_op(x, 
                                        self.act_scales_inv, 
                                        self.act_zero_points)
            output = qlinear(
                x_int,                              # input_int
                self.weight_int,                    # weight_int
                self.weight_scales,                 # weight_scale
                self.act_scales,                    # input_scale
                self.act_zero_points,               # input_zp
                self.weight_sum_by_input_channels,  
                                    # weight_sum_by_input_channels
                self.scale,
                self.bias0,
                self.bias                           # bias (None or tensor)
            )
            return output
        else:
            # use bos and quantize the activation
            x_except_first_token = quant_op(x[:,1:,:], 
                                                       self.act_scales_inv, 
                                                       self.act_zero_points)
            out_except_first_token = qlinear(x_except_first_token, 
                                             self.weight_int,
                                             self.weight_scales,
                                             self.act_scales,
                                             self.act_zero_points,
                                             self.weight_sum_by_input_channels,
                                             self.scale,
                                             self.bias0,
                                             self.bias)
            out_first_token = self.bos_pre_computed.expand(x.shape[0], -1, -1)
            output =torch.cat([out_first_token, out_except_first_token], dim=1)
            return output
