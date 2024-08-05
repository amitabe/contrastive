#pragma once

#ifndef DISTANCES_H
#define DISTANCES_H

#include <torch/torch.h>

torch::Tensor cosine_distances(torch::Tensor tensor);
torch::Tensor l2_distances(torch::Tensor tensor);

#endif  // DISTANCES_H
