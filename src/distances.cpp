#include "distances.h"
#include <torch/torch.h>

// Function to compute cosine distances on the last 2 dimensions
torch::Tensor cosine_distances(torch::Tensor tensor) {
    // Check if the tensor has fewer than 2 dimensions
    if (tensor.dim() < 2) {
        throw std::invalid_argument("Tensor must have at least 2 dimensions.");
    }

    // Check if the tensor is empty
    if (tensor.numel() == 0) {
        return tensor;  // Return as-is if the tensor is empty
    }

    // Normalize along the last dimension
    namespace F = torch::nn::functional;
    auto options = F::NormalizeFuncOptions().p(2).dim(-1);
    auto normalized_tensor = F::normalize(tensor, options);

    // Compute the cosine similarity matrix for the last two dimensions
    auto cosine_similarity = torch::matmul(normalized_tensor, normalized_tensor.transpose(-2, -1));

    return cosine_similarity;
}



// Function to compute negative L2 distances on the last dimension
torch::Tensor l2_distances(torch::Tensor tensor) {
    // If the tensor is one-dimensional, expand it to two dimensions
    if (tensor.dim() == 1) {
        tensor = tensor.unsqueeze(-1);
    }

    // Check if the tensor has fewer than 2 dimensions
    if (tensor.dim() < 2) {
        throw std::invalid_argument("Tensor must have at least 2 dimensions.");
    }

    // Check if the tensor is empty
    if (tensor.numel() == 0) {
        return tensor;  // Return as-is if the tensor is empty
    }

    // cdist is not supported for float16, so we will compute the L2 distances manually
    if (tensor.scalar_type() != torch::kFloat16) {
        return torch::cdist(tensor, tensor, 2);
    }

    // Compute the pairwise L2 distances
    auto diff = tensor.unsqueeze(-2) - tensor.unsqueeze(-3);  // Expand dimensions for pairwise differences
    auto squared_diff = diff.pow(2).sum(-1);  // Sum of squares along the last dimension
    auto l2 = squared_diff.sqrt();  // Take square root and negate

    return l2;
}
