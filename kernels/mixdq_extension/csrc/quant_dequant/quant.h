#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>

namespace mixdq {

void initQuantizationBindings(py::module m);

}