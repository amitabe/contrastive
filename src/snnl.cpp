#include "snnl.h"
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <string>
#include <iostream>
#include "utilities.h"
#include "distances.h"


torch::Tensor snnl(torch::Tensor tensor, torch::Tensor labels, double temperature,
    std::string reduce, bool use_cosine) {
    // Step 1: Argument Checks
    if (reduce != "mean" && reduce != "none") {
        throw std::invalid_argument("reduce must be 'mean' or 'none'");
    }

    if (tensor.dim() < 2) {
        throw std::invalid_argument("tensor must have at least 2 dimensions");
    }

    // Step 2: Compute the similarity matrix based on use_cosine
    torch::Tensor sim;
    if (use_cosine) {
        sim = cosine_distances(tensor);
    } else {
        sim = -l2_distances(tensor);
    }
    sim = sim / temperature;

    // Step 3: Create Positive Mask
    auto positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0);

    // Step 4: Remove Diagonal from sim and positive_mask
    sim = remove_diagonal(sim);
    positive_mask = remove_diagonal(positive_mask);

    // Step 5: Numerical Stability Adjustment
    auto max_val = std::get<0>(sim.max(1, true)).detach();
    sim = sim - max_val;

    // Step 6: Exponentiate the similarity matrix
    auto sim_exp = sim.exp();

    // Step 7: Compute Numerator and Denominator
    auto numerator = (sim_exp * positive_mask).sum(-1);
    auto denominator = (sim_exp * (~positive_mask)).sum(-1);

    // Step 8: Compute log argument with numerical stability
    auto log_arg = numerator / (denominator + numerator);

    // Add a tiny value to log_arg to avoid log(0)
    DISPATCH_FLOATING_TYPES_AND_BFLOAT16(log_arg.scalar_type(), "get_tiny_value", [&]() {
        scalar_t tiny = std::numeric_limits<scalar_t>::min();
        log_arg = log_arg + tiny;
        return log_arg;
    });

    // Step 9: Compute Negative Log
    auto log_res = -log_arg.log();

    // Step 10: Apply reduction based on the reduce argument
    if (reduce == "mean") {
        return log_res.mean();
    } else if (reduce == "none") {
        return log_res;
    } else {
        throw std::invalid_argument("reduce must be 'mean' or 'none'");
    }
}


// Binding the functions to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("snnl", &snnl,
        py::arg("tensor"), py::arg("labels"), py::arg("temperature") = 1, py::arg("reduce") = "mean",
        py::arg("use_cosine") = false,
        "Compute the Soft-Nearest Neighbors Loss");
}
