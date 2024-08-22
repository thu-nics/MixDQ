#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <pybind11/stl.h>

namespace mixdq {
void initQuantizedLinearBindings(py::module m);
}