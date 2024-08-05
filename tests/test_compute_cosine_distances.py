import torch
from contrastive import cosine_distances  # Import the custom module

def test_cosine_similarity_basic():
    # Basic test with 2D tensor
    z = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)

    expected = torch.tensor([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float32)

    result = cosine_distances(z)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"

def test_cosine_similarity_all_ones():
    # Test with all ones tensor
    z = torch.tensor([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]], dtype=torch.float32)

    expected = torch.tensor([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]], dtype=torch.float32)

    result = cosine_distances(z)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"

def test_cosine_similarity_random():
    # Test with random tensor
    z = torch.randn(3, 4)
    normalized_z = torch.nn.functional.normalize(z, dim=-1)
    expected = normalized_z @ normalized_z.T

    result = cosine_distances(z)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"

def test_cosine_similarity_empty():
    # Test with an empty tensor
    z = torch.tensor([[]], dtype=torch.float32)
    expected = torch.tensor([[]], dtype=torch.float32)

    result = cosine_distances(z)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_cosine_similarity_single_element():
    # Test with a single element tensor
    z = torch.tensor([[1.0]], dtype=torch.float32)
    expected = torch.tensor([[1.0]], dtype=torch.float32)

    result = cosine_distances(z)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_cosine_similarity_more_than_2d():
    # Test with a tensor with more than 2 dimensions
    z = torch.randn(2, 3, 4)
    normalized_z = torch.nn.functional.normalize(z, dim=-1)

    # Compute pairwise cosine similarity for each 3x4 matrix in the batch
    expected = normalized_z @ normalized_z.transpose(-1, -2)

    result = cosine_distances(z)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


def test_cosine_similarity_dtype64_preservation():
    # Test with random tensor
    z = torch.randn(3, 4, dtype=torch.float64)

    result = cosine_distances(z)
    assert result.dtype == torch.float64, f"Expected dtype torch.float64, but got {result.dtype}"


def test_cosine_similarity_dtype32_preservation():
    # Test with random tensor
    z = torch.randn(3, 4, dtype=torch.float32)

    result = cosine_distances(z)
    assert result.dtype == torch.float32, f"Expected dtype torch.float32, but got {result.dtype}"


def test_cosine_similarity_dtype64_preservation():
    # Test with random tensor
    z = torch.randn(3, 4, dtype=torch.float16)

    result = cosine_distances(z)
    assert result.dtype == torch.float16, f"Expected dtype torch.float16, but got {result.dtype}"



if __name__ == "__main__":
    import pytest
    pytest.main()
