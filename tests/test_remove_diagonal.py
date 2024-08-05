import torch
from contrastive import remove_diagonal

def test_remove_simple_diagonal_2d():
    # Test with a 2D tensor
    x = torch.tensor([[1, 2],
                      [3, 4]], dtype=torch.float32)
    expected = torch.tensor([[2], [3]], dtype=torch.float32)

    result = remove_diagonal(x)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_remove_diagonal_2d():
    # Test with a 2D tensor
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=torch.float32)
    expected = torch.tensor([[2, 3],
                             [4, 6],
                             [7, 8]], dtype=torch.float32)

    result = remove_diagonal(x)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_remove_diagonal_3d():
    # Test with a 3D tensor
    x = torch.tensor([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]]], dtype=torch.float32) # shape [2, 2, 2]

    # Since this is a 3D tensor, and remove_diagonal operates on 2D (removing the diagonal along the last two dimensions)
    expected = torch.tensor([[[3, 4]], [[5, 6]]], dtype=torch.float32)

    result = remove_diagonal(x)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_remove_diagonal_empty():
    # Test with an empty tensor
    x = torch.tensor([[]], dtype=torch.float32)
    expected = torch.tensor([[]], dtype=torch.float32)

    result = remove_diagonal(x)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_remove_diagonal_single_element():
    # Test with a single element tensor
    x = torch.tensor([[42]], dtype=torch.float32)
    expected = torch.tensor([], dtype=torch.float32).reshape(1, 0)

    result = remove_diagonal(x)
    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_remove_diagonal_dtype64_preservation():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    result = remove_diagonal(x)
    assert result.dtype == torch.float64, f"Expected dtype torch.float64, but got {result.dtype}"

def test_remove_diagonal_dtype32_preservation():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = remove_diagonal(x)
    assert result.dtype == torch.float32, f"Expected dtype torch.float32, but got {result.dtype}"

def test_remove_diagonal_dtype16_preservation():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
    result = remove_diagonal(x)
    assert result.dtype == torch.float16, f"Expected dtype torch.float16, but got {result.dtype}"


if __name__ == "__main__":
    import pytest
    pytest.main()
