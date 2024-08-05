
#include "utilities.h"
#include <torch/torch.h>
#include <vector>

// Function to remove the diagonal from athe first 2 dimensions of a tensor
torch::Tensor remove_diagonal(torch::Tensor tensor) {
    // Check if the tensor is empty or has fewer than 2 dimensions
    if (tensor.numel() == 0 || tensor.dim() < 2) {
        return tensor;  // Return as-is if fewer than 2 dimensions
    }

    // Get the shape of the input tensor
    auto N = tensor.size(0);
    auto remaining_dims = tensor.sizes().slice(2).vec();

    // Flatten the tensor up to the first dimension
    auto flattened = tensor.flatten(0, 1);

    // Remove the first element (corresponding to the first diagonal element in the original matrix)
    auto flattened_no_diag = flattened.narrow(0, 1, flattened.size(0) - 1);

    // Reshape to remove the diagonal
    std::vector<int64_t> new_shape = {N - 1, N + 1};
    auto reshaped = flattened_no_diag.unflatten(0, new_shape);

    // Remove the last column (the original diagonal elements)
    auto tensor_without_diagonal = reshaped.narrow(1, 0, N);

    // Reshape to the final desired shape
    std::vector<int64_t> final_shape = {N, N - 1};
    final_shape.insert(final_shape.end(), remaining_dims.begin(), remaining_dims.end());

    tensor_without_diagonal = tensor_without_diagonal.reshape(final_shape);

    return tensor_without_diagonal;
}
