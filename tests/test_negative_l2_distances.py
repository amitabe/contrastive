import torch
from contrastive import l2_distances # Import the custom module

def test_l2_distances_basic():
    # Basic test with 2D tensor
    z = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0]], dtype=torch.float32)

    # Expected negative L2 distances manually calculated
    expected = torch.tensor([[-0.0, -1.4142, -1.0],
                             [-1.4142, -0.0, -1.0],
                             [-1.0, -1.0, -0.0]])

    result = l2_distances(z)
    assert torch.allclose(result, expected, atol=1e-4), f"Expected {expected}, but got {result}"

def test_l2_distances_all_ones():
    # Test with all ones tensor
    z = torch.tensor([[1.0, 1.0],
                      [1.0, 1.0],
                      [1.0, 1.0]], dtype=torch.float32).unsqueeze(0)

    # Expected negative L2 distances should all be 0
    expected = torch.tensor([[-0.0, -0.0, -0.0],
                             [-0.0, -0.0, -0.0],
                             [-0.0, -0.0, -0.0]]).unsqueeze(0)

    result = l2_distances(z)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"

def test_l2_distances_random():
    # Test with random tensor
    z = torch.randn(3, 4)
    l2_distances = -(z.unsqueeze(1) - z.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()

    result = l2_distances(z.unsqueeze(0))
    assert torch.allclose(result, l2_distances, atol=1e-6), f"Expected {l2_distances}, but got {result}"

def test_l2_distances_empty():
    # Test with an empty tensor
    z = torch.tensor([[]], dtype=torch.float32).unsqueeze(0)
    expected = torch.tensor([[]], dtype=torch.float32).unsqueeze(0)

    result = l2_distances(z)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_l2_distances_single_element():
    # Test with a single element tensor
    z = torch.tensor([[1.0]], dtype=torch.float32).unsqueeze(0)
    expected = torch.tensor([[-0.0]], dtype=torch.float32).unsqueeze(0)

    result = l2_distances(z)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_l2_distances_more_than_2d():
    # Test with a tensor with more than 2 dimensions
    z = torch.randn(2, 3, 4)
    l2_distances = torch.stack([
        -((z[i].unsqueeze(1) - z[i].unsqueeze(0)).pow(2).sum(dim=-1).sqrt())
        for i in range(z.size(0))
    ])

    result = l2_distances(z)
    assert torch.allclose(result, l2_distances, atol=1e-6), f"Expected {l2_distances}, but got {result}"

def test_l2_distances_1d_tensor():
    # Test with a 1D tensor
    z = torch.tensor([1, 2, 3])

    # Expected negative L2 distances should all be 0
    expected = torch.tensor([[-0, -1, -2],
                             [-1, -0, -1],
                             [-2, -1, -0]], dtype=torch.float32)

    result = l2_distances(z)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


def test_l2_distances_dtype64_preservation():
    # Test with random tensor
    z = torch.randn(3, 4, dtype=torch.float64)

    result = l2_distances(z)
    assert result.dtype == torch.float64, f"Expected dtype torch.float64, but got {result.dtype}"

def test_l2_distances_dtype32_preservation():
    # Test with random tensor
    z = torch.randn(3, 4, dtype=torch.float32)

    result = l2_distances(z)
    assert result.dtype == torch.float32, f"Expected dtype torch.float32, but got {result.dtype}"

def test_l2_distances_dtype16_preservation():
    # Test with random tensor
    z = torch.randn(3, 4, dtype=torch.float16)

    result = l2_distances(z)
    assert result.dtype == torch.float16, f"Expected dtype torch.float16, but got {result.dtype}"


if __name__ == "__main__":
    import pytest
    pytest.main()
