import logging
import yaml
import torch.nn as nn
from typing import Union, Optional, Dict, Any, Tuple
from qdiff.models.quant_block import get_specials, BaseQuantBlock
from qdiff.models.quant_block import QuantTransformerBlock, QuantResnetBlock2D
from qdiff.models.quant_layer import QuantLayer
from qdiff.quantizer.base_quantizer import StraightThrough, BaseQuantizer, WeightQuantizer, ActQuantizer

from diffusers.models.transformer_2d import BasicTransformerBlock as TransformerBlock
import torch
logger = logging.getLogger(__name__)


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.dtype = torch.float32

        self.weight_quant = False if weight_quant_params is None else True
        self.act_quant = False if act_quant_params is None else True

        self.model = model
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials()  # some nn.Modules require special process
        logger.info(f"\n --------------- refactoring quant layers --------------- \n")
        self.quant_layer_refactor(self.model, weight_quant_params, act_quant_params)
        logger.info(f"\n --------------- refactoring quant blocks --------------- \n")
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        # self.set_module_name_for_quantizer(module=self.model)  # add the module name as attribute for each quantizer
        self.quant_params_dict = {}  # init the quant_params_dict as empty


    def quant_layer_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantLayer
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                tmp_module = child_module
                setattr(module, name, QuantLayer(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
                # logger.info(f"\n origional module: {name}:{tmp_module}, \n new module {prev_quantmodule}")
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_layer_refactor(child_module, weight_quant_params, act_quant_params)  # recursive call1


    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                tmp_module = child_module
                if self.specials[type(child_module)] in [QuantTransformerBlock]:
                    setattr(module, name, self.specials[type(child_module)](child_module, act_quant_params))
                else:
                    tmp_module = child_module
                    setattr(module, name, self.specials[type(child_module)](child_module, act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # update the QuantModel quant_state
        self.weight_quant = weight_quant
        self.act_quant = act_quant

        for m in self.model.modules():
            if isinstance(m, (QuantLayer, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)


    def get_quant_state(self):
        return self.weight_quant, self.act_quant


    def set_module_name_for_quantizer(self, module, prefix=""):
        '''set the nn.Module name for each quantizer'''
        for name_, module_ in module.named_children():
            full_name = prefix + name_ if prefix else name_
            torch.cuda.empty_cache()
            if isinstance(module_, BaseQuantizer):  # end with quantizer module
                setattr(module_,'module_name',full_name)
                # print(module_, full_name)
            else:
                self.set_module_name_for_quantizer(module=module_, prefix=full_name+'.')


    def set_quant_init_done(self, quantizer_type_name, module=None):
        if module is None:
            module = self.model  # use full model when empty module input
        '''set init_done name for each quantizer'''
        if quantizer_type_name == "weight":
            quantizer_type = WeightQuantizer
        elif quantizer_type_name == "activation":
            quantizer_type = ActQuantizer
        else:
            raise NotImplementedError

        for name_, module_ in module.named_children():
            torch.cuda.empty_cache()
            if isinstance(module_, quantizer_type):  # end with quantizer module
                module_.init_done = True
            else:
                self.set_quant_init_done(quantizer_type_name, module_)


    def get_quant_params_dict(self, module=None, prefix="", dtype=torch.float32):
        # iter through all quantizers, get the buffers
        if module is None:
            module = self.model
            self.quant_params_dict = {}
        quantizer_type = BaseQuantizer
        # recursively iter through all quantizers
        for name, module_ in module.named_children():
            full_name = prefix + name if prefix else name
            torch.cuda.empty_cache()
            if isinstance(module_, quantizer_type):
                # pack the dict into the 'module_name'
                # [buffers_(OrderdDict), parameters(OrderedDict)]
                self.quant_params_dict[module_.module_name] = []
                self.quant_params_dict[module_.module_name].append(module_._buffers)
                self.quant_params_dict[module_.module_name].append(module_._parameters)
            else:
                self.get_quant_params_dict(module=module_, prefix=full_name+'.')

        return self.quant_params_dict


    def set_quant_params_dict(self, quant_params_dict, module=None, load_buffer_only=True, dtype=torch.float32):
        # iter through all quantizers, set the buffers with self.quant_param_dict
        # quant_parma_dict: ['conv_in.weight_quantizer'] is a tuple, 1st is _bufferes, 2nd is _params()]
        # load_buffer_only: when `quantized_inference`, should only load the buffers (the saved ckpt should be all buffers)
        # when resuming quantization, load both the buffers and the parameters
        if module is None:
            module = self.model

        quantizer_type = BaseQuantizer

        # recursively iter through all quantizers
        for name, module_ in module.named_children():
            torch.cuda.empty_cache()
            if isinstance(module_, quantizer_type):
               # unpack the dict
                if load_buffer_only:
                    assert len(quant_params_dict[module_.module_name][1]) == 0  # parameters() has no element
                    for name, quant_params in quant_params_dict[module_.module_name][0].items():  # use module_name to index the dict
                        setattr(module_, name, quant_params.to(dtype) if quant_params is not None else None)
                else:
                    # set buffer
                    for name, quant_params in quant_params_dict[module_.module_name][0].items():
                        setattr(module_, name, quant_params.to(dtype) if quant_params is not None else None)
                    # set parameter
                    for name, quant_params in quant_params_dict[module_.module_name][0].items():
                        setattr(module_, name, quant_params.to(dtype) if quant_params is not None else None)
            else:
                self.set_quant_params_dict(quant_params_dict=quant_params_dict, module=module_)


    def replace_quant_buffer_with_parameter(self, opt_d, module=None):
        if module is None:
            module = self.model

        for opt_target in opt_d.keys():

            if opt_target == 'weight':
                quantizer_type = WeightQuantizer
            elif opt_target == 'activation':
                quantizer_type = ActQuantizer

            for name, module_ in module.named_children():
                # print(module_)
                torch.cuda.empty_cache()
                if isinstance(module_, quantizer_type):
                    if opt_d[opt_target] is not None:
                        for param_type in opt_d[opt_target]:
                            buffer_ = getattr(module_, param_type)
                            assert isinstance(buffer_, torch.Tensor)
                            delattr(module_, param_type)
                            module_.register_parameter(param_type, nn.Parameter(buffer_))
                else:
                    self.replace_quant_buffer_with_parameter(opt_d, module=module_)


    def replace_quant_parameter_with_buffers(self, opt_d, module=None):
        if module is None:
            module = self.model

        quantizer_type = BaseQuantizer

        for opt_target in opt_d.keys():

            if opt_target == 'weight':
                quantizer_type = WeightQuantizer
            elif opt_target == 'activation':
                quantizer_type = ActQuantizer

            for name, module_ in module.named_children():
                torch.cuda.empty_cache()
                if isinstance(module_, quantizer_type):
                    # if opt_d[opt_target] is not None:
                    if opt_d[opt_target] is not None:
                        for param_type in opt_d[opt_target]:
                            buffer_ = getattr(module_, param_type).data
                            assert isinstance(buffer_, torch.Tensor)
                            delattr(module_, param_type)
                            module_.register_buffer(param_type, buffer_)
                else:
                    self.replace_quant_parameter_with_buffers(opt_d, module=module_)


    def forward(self,
                sample: torch.FloatTensor,
                timestep: Union[torch.Tensor, float, int],
                encoder_hidden_states: torch.Tensor,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True,):
        # compatible with diffusers UNetCondition2D forward function
        return self.model(sample, timestep, encoder_hidden_states, class_labels, timestep_cond, attention_mask, cross_attention_kwargs,
                        added_cond_kwargs, down_block_additional_residuals, mid_block_additional_residual, down_block_additional_residuals, encoder_attention_mask,
                        return_dict)

    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt
            if isinstance(m, (QuantTransformerBlock, TransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt


    def set_layer_quant(self, model=None, module_name_list=[], group_list=[], group_ignore=[],  quant_level='per_layer', weight_quant=True, act_quant=False, prefix=""):
        '''
        progressively quantize the groups or layers, which is different from the func in the quant_error.py
        quantify all layers in the module_list or group_list at once
        group_ignore: if quant_level is 'per_group', selectively ignore the quantization of certain groups
        '''
        if quant_level=='per_group':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if isinstance(module, QuantLayer):
                    for module_class in group_list:
                        if module_class in full_name:
                            if all(element not in full_name for element in group_ignore):
                                module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                else:
                    self.set_layer_quant(model=module, module_name_list=module_name_list, group_list=group_list, group_ignore=group_ignore, quant_level='per_group', weight_quant=weight_quant, act_quant=act_quant, prefix=full_name+".")
        
        if quant_level=='per_layer':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if isinstance(module, QuantLayer):
                    for module_name in module_name_list:
                        if module_name == full_name or ('model.'+module_name)==full_name:
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            torch.cuda.empty_cache()
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                else:
                    self.set_layer_quant(model=module, module_name_list=module_name_list, group_list=group_list, group_ignore=group_ignore, quant_level='per_layer', weight_quant=weight_quant, act_quant=act_quant, prefix=full_name+".")

        if quant_level=='per_block':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if isinstance(module, BaseQuantBlock):
                    for module_name in module_name_list:
                        module_name = 'model.'+module_name
                        if module_name == full_name:
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            torch.cuda.empty_cache()
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                elif isinstance(module, QuantLayer):
                    for module_name in module_name_list:
                        module_name = 'model.'+module_name
                        if module_name == full_name:
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            torch.cuda.empty_cache()
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                else:
                    self.set_layer_quant(model=module, module_name_list=module_name_list, group_list=group_list, group_ignore=group_ignore, quant_level='per_block', weight_quant=weight_quant, act_quant=act_quant, prefix=full_name+".")


    def set_layer_bit(self, model=None, n_bit=None, module_name_list=[], group_list=[], quant_level='per_layer', bit_type='weight', prefix=""):
        '''
        Progressively set bit of the the groups or layers, which is different from the func in the quant_error.py.
        Selectivly quantize some layers of groups to low bit
        bit_type: 'weight' of 'act', we can only quantize weight or act at once
        '''
        if quant_level=='per_layer':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if bit_type == 'weight':
                    if isinstance(module, WeightQuantizer):
                        for module_name in module_name_list:
                            if  module_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: weight_bit={n_bit}")
                    else:
                        self.set_layer_weight_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_layer', bit_type=bit_type, prefix=full_name+".")
                elif bit_type == 'act':
                    if isinstance(module, ActQuantizer):
                        for module_name in module_name_list:
                            if  module_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: act_bit={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_layer', bit_type=bit_type, prefix=full_name+".")

        elif quant_level=='per_group':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if bit_type == 'weight':
                    if isinstance(module, WeightQuantizer):
                        for group_name in group_list:
                            if group_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: weight_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_group', bit_type=bit_type, prefix=full_name+".")
                if bit_type == 'act':
                    if isinstance(module, ActQuantizer):
                        for group_name in group_list:
                            if group_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: act_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_group', bit_type=bit_type, prefix=full_name+".")
        
        elif quant_level=='reset':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if bit_type == 'weight':
                    if isinstance(module, WeightQuantizer):
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: weight_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='reset', bit_type=bit_type, prefix=full_name+".")
                if bit_type == 'act':
                    if isinstance(module, ActQuantizer):
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: act_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='reset', bit_type=bit_type, prefix=full_name+".")


    def load_bitwidth_config(self, model, bit_config, bit_type, prefix=''):
        '''
        please pass the bit_config of weight and act seperatly
        '''
        for name, module in model.named_children():
            full_name = prefix + name if prefix else name

            if isinstance(module, QuantLayer):
                if full_name in bit_config.keys():
                    if bit_type == 'weight':
                        module.weight_quantizer.bitwidth_refactor(bit_config[full_name])
                        if hasattr(module, 'weight_quantizer_0'):
                            module.weight_quantizer_0.bitwidth_refactor(bit_config[full_name])
                        logger.info(f"{full_name}: the w_bit = {bit_config[full_name]}")

                    elif bit_type == 'act':
                        module.act_quantizer.bitwidth_refactor(bit_config[full_name])
                        if hasattr(module, 'act_quantizer_0'):
                            module.act_quantizer_0.bitwidth_refactor(bit_config[full_name])      
                        logger.info(f"{full_name}: the a_bit = {bit_config[full_name]}")

                    torch.cuda.empty_cache()

            else:
                self.load_bitwidth_config(model=module, bit_config=bit_config, bit_type=bit_type, prefix=full_name+".")


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
