#pragma once

#ifndef UTILITIES_H
#define UTILITIES_H

#include <torch/torch.h>

torch::Tensor remove_diagonal(torch::Tensor tensor);

#endif  // UTILITIES_H
