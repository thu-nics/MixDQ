#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>

#include "qlinear/qlinear.h"
#include "qconv2d/qconv2d.h"
#include "quant_dequant/quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    mixdq::initQuantizedLinearBindings(m);
    mixdq::initQuantizedConv2dBindings(m);
    mixdq::initQuantizationBindings(m);
}