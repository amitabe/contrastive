#pragma once

#ifndef SNNL_H
#define SNNL_H

#include <torch/torch.h>
#include <string>

// MACRO to dispatch based on the tensor type
#define DISPATCH_FLOATING_TYPES_AND_BFLOAT16(TYPE, NAME, ...)                \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    switch (the_type) {                                                      \
      case torch::kFloat32: {                                                \
        using scalar_t = float;                                              \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case torch::kFloat64: {                                                \
        using scalar_t = double;                                             \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case torch::kFloat16: {                                                \
        using scalar_t = at::Half;                                           \
        return __VA_ARGS__();                                                \
      }                                                                      \
      case torch::kBFloat16: {                                               \
        using scalar_t = at::BFloat16;                                       \
        return __VA_ARGS__();                                                \
      }                                                                      \
      default:                                                               \
        throw std::runtime_error("Unsupported tensor type in " #NAME);       \
    }                                                                        \
  }()


torch::Tensor snnl(torch::Tensor tensor, torch::Tensor labels, double temperature = 1,
    std::string reduce = "mean", bool use_cosine = false);

#endif  // SNNL_H
