"""
Verify all code snippets from Part 1 slides are runnable.

Run with: python code-verification.py
"""

import torch
import numpy as np


def test_tensor_creation():
    """From part-01-tensors-basics.md: Creating Tensors."""
    # From a Python list
    t1 = torch.tensor([1, 2, 3])
    assert t1.shape == torch.Size([3])

    # From nested lists (matrix)
    t2 = torch.tensor([[1, 2], [3, 4]])
    assert t2.shape == torch.Size([2, 2])

    # From NumPy array
    arr = np.array([1, 2, 3])
    t3 = torch.from_numpy(arr)
    assert t3.shape == torch.Size([3])

    # Factory functions
    zeros = torch.zeros(3, 4)
    assert zeros.shape == torch.Size([3, 4])

    ones = torch.ones(2, 3)
    assert ones.shape == torch.Size([2, 3])

    rand = torch.rand(2, 2)
    assert rand.shape == torch.Size([2, 2])

    randn = torch.randn(2, 2)
    assert randn.shape == torch.Size([2, 2])

    seq = torch.arange(0, 10, 2)
    assert torch.equal(seq, torch.tensor([0, 2, 4, 6, 8]))

    lin = torch.linspace(0, 1, 5)
    assert lin.shape == torch.Size([5])

    like = torch.zeros_like(rand)
    assert like.shape == rand.shape

    print("  [PASS] tensor_creation")


def test_tensor_attributes():
    """From part-01-tensors-basics.md: Tensor Attributes."""
    t = torch.rand(3, 4)

    assert t.shape == torch.Size([3, 4])
    assert t.dtype == torch.float32
    assert str(t.device) == "cpu"

    # Shape methods
    assert t.size() == torch.Size([3, 4])
    assert t.ndim == 2
    assert t.numel() == 12

    print("  [PASS] tensor_attributes")


def test_dtype_casting():
    """From part-01-tensors-basics.md: Data Types and Casting."""
    # Explicit dtype
    t_float = torch.tensor([1, 2, 3], dtype=torch.float32)
    assert t_float.dtype == torch.float32

    t_int = torch.tensor([1.5, 2.5], dtype=torch.int64)
    assert t_int.dtype == torch.int64

    # Casting
    t = torch.tensor([1, 2, 3])
    t_float = t.to(torch.float32)
    assert t_float.dtype == torch.float32

    assert t.float().dtype == torch.float32
    assert t.double().dtype == torch.float64
    assert t.int().dtype == torch.int32
    assert t.long().dtype == torch.int64
    assert t.bool().dtype == torch.bool

    # Truncation warning
    result = torch.tensor([1.7, 2.3]).int()
    assert torch.equal(result, torch.tensor([1, 2]))

    print("  [PASS] dtype_casting")


def test_elementwise_operations():
    """From part-01-operations.md: Element-wise Operations."""
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])

    assert torch.equal(a + b, torch.tensor([5, 7, 9]))
    assert torch.equal(a - b, torch.tensor([-3, -3, -3]))
    assert torch.equal(a * b, torch.tensor([4, 10, 18]))
    assert torch.equal(a**2, torch.tensor([1, 4, 9]))

    # With scalars
    assert torch.equal(a + 10, torch.tensor([11, 12, 13]))
    assert torch.equal(a * 2, torch.tensor([2, 4, 6]))

    # Math functions
    t = torch.tensor([1.0, 4.0, 9.0])
    assert torch.equal(torch.sqrt(t), torch.tensor([1.0, 2.0, 3.0]))

    print("  [PASS] elementwise_operations")


def test_reduction_operations():
    """From part-01-operations.md: Reduction Operations."""
    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    assert t.sum() == 21
    assert t.mean() == 3.5
    assert t.max() == 6
    assert t.min() == 1

    # Along axis
    assert torch.equal(t.sum(dim=0), torch.tensor([5.0, 7.0, 9.0]))
    assert torch.equal(t.sum(dim=1), torch.tensor([6.0, 15.0]))

    # argmax/argmin
    t2 = torch.tensor([3, 1, 4, 1, 5])
    assert t2.argmax() == 4
    assert t2.argmin() == 1

    print("  [PASS] reduction_operations")


def test_matrix_operations():
    """From part-01-operations.md: Matrix Operations."""
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[5, 6], [7, 8]])

    expected = torch.tensor([[19, 22], [43, 50]])
    assert torch.equal(a @ b, expected)
    assert torch.equal(torch.matmul(a, b), expected)
    assert torch.equal(torch.mm(a, b), expected)

    # Transpose
    m = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    assert torch.equal(m.T, torch.tensor([[1, 3], [2, 4]], dtype=torch.float32))

    # Dot product
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    assert torch.dot(a, b) == 32

    print("  [PASS] matrix_operations")


def test_comparison_operations():
    """From part-01-operations.md: Comparison Operations."""
    a = torch.tensor([1, 2, 3, 4])
    b = torch.tensor([2, 2, 2, 2])

    assert torch.equal(a > b, torch.tensor([False, False, True, True]))
    assert torch.equal(a >= b, torch.tensor([False, True, True, True]))
    assert torch.equal(a == b, torch.tensor([False, True, False, False]))

    # torch.where
    result = torch.where(a > 2, a, torch.zeros_like(a))
    assert torch.equal(result, torch.tensor([0, 0, 3, 4]))

    print("  [PASS] comparison_operations")


def test_broadcasting():
    """From part-01-broadcasting.md: Broadcasting."""
    # Scalar broadcasting
    a = torch.tensor([1, 2, 3])
    assert torch.equal(a + 10, torch.tensor([11, 12, 13]))

    # (2, 3) + (3,) -> (2, 3)
    a = torch.ones(2, 3)
    b = torch.tensor([1, 2, 3])
    expected = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    assert torch.equal(a + b, expected)

    # (2, 1) + (1, 3) -> (2, 3)
    a = torch.tensor([[1], [2]])
    b = torch.tensor([[10, 20, 30]])
    expected = torch.tensor([[11, 21, 31], [12, 22, 32]])
    assert torch.equal(a + b, expected)

    print("  [PASS] broadcasting")


def test_reshaping():
    """From part-01-reshaping.md: Reshaping Tensors."""
    t = torch.arange(12)

    reshaped = t.reshape(3, 4)
    assert reshaped.shape == torch.Size([3, 4])

    # -1 inference
    assert t.reshape(3, -1).shape == torch.Size([3, 4])
    assert t.reshape(-1, 6).shape == torch.Size([2, 6])

    # squeeze/unsqueeze
    t = torch.zeros(1, 3, 1, 4)
    assert t.squeeze().shape == torch.Size([3, 4])
    assert t.squeeze(0).shape == torch.Size([3, 1, 4])

    t = torch.zeros(3, 4)
    assert t.unsqueeze(0).shape == torch.Size([1, 3, 4])
    assert t.unsqueeze(-1).shape == torch.Size([3, 4, 1])

    # flatten
    t = torch.zeros(2, 3, 4)
    assert t.flatten().shape == torch.Size([24])
    assert t.flatten(1).shape == torch.Size([2, 12])

    print("  [PASS] reshaping")


def test_transpose_permute():
    """From part-01-reshaping.md: Transpose and Permute."""
    m = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert m.T.shape == torch.Size([3, 2])
    assert m.t().shape == torch.Size([3, 2])
    assert m.transpose(0, 1).shape == torch.Size([3, 2])

    # Permute
    t = torch.zeros(2, 3, 4)
    assert t.permute(0, 2, 1).shape == torch.Size([2, 4, 3])
    assert t.permute(2, 1, 0).shape == torch.Size([4, 3, 2])

    print("  [PASS] transpose_permute")


def test_views_copies():
    """From part-01-reshaping.md: Views vs Copies."""
    # View shares memory
    a = torch.tensor([1, 2, 3, 4])
    b = a.view(2, 2)
    assert a.data_ptr() == b.data_ptr()

    # Clone creates copy
    a = torch.tensor([1, 2, 3, 4])
    b = a.clone()
    b[0] = 99
    assert a[0] == 1  # unchanged

    print("  [PASS] views_copies")


def test_contiguous():
    """From part-01-reshaping.md: Contiguous Memory."""
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert a.is_contiguous()

    b = a.T
    assert not b.is_contiguous()

    # contiguous() makes it contiguous
    c = b.contiguous()
    assert c.is_contiguous()

    print("  [PASS] contiguous")


def test_basic_indexing():
    """From part-01-indexing.md: Basic Indexing."""
    t = torch.tensor([10, 20, 30, 40, 50])
    assert t[0] == 10
    assert t[-1] == 50
    assert t[2] == 30

    m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert m[0, 0] == 1
    assert m[1, 2] == 6
    assert m[-1, -1] == 9
    assert torch.equal(m[0], torch.tensor([1, 2, 3]))
    assert torch.equal(m[:, 0], torch.tensor([1, 4, 7]))

    print("  [PASS] basic_indexing")


def test_slicing():
    """From part-01-indexing.md: Slicing."""
    t = torch.arange(10)

    assert torch.equal(t[2:5], torch.tensor([2, 3, 4]))
    assert torch.equal(t[:3], torch.tensor([0, 1, 2]))
    assert torch.equal(t[7:], torch.tensor([7, 8, 9]))
    assert torch.equal(t[::2], torch.tensor([0, 2, 4, 6, 8]))

    print("  [PASS] slicing")


def test_boolean_indexing():
    """From part-01-indexing.md: Boolean Indexing."""
    t = torch.tensor([1, 5, 3, 8, 2, 9])

    mask = t > 4
    assert torch.equal(t[mask], torch.tensor([5, 8, 9]))
    assert torch.equal(t[t > 4], torch.tensor([5, 8, 9]))

    # Multiple conditions
    assert torch.equal(t[(t > 2) & (t < 8)], torch.tensor([5, 3]))

    print("  [PASS] boolean_indexing")


def test_advanced_indexing():
    """From part-01-indexing.md: Advanced Indexing."""
    t = torch.tensor([10, 20, 30, 40, 50])
    indices = torch.tensor([0, 2, 4])
    assert torch.equal(t[indices], torch.tensor([10, 30, 50]))

    # torch.where
    a = torch.tensor([1, 2, 3, 4, 5])
    b = torch.tensor([10, 20, 30, 40, 50])
    result = torch.where(a > 3, a, b)
    assert torch.equal(result, torch.tensor([10, 20, 30, 4, 5]))

    print("  [PASS] advanced_indexing")


def test_numpy_conversion():
    """From part-01-numpy-comparison.md: NumPy Conversion."""
    # NumPy to PyTorch
    arr = np.array([1, 2, 3])
    t1 = torch.from_numpy(arr)
    t2 = torch.tensor(arr)

    # Shared memory check
    arr[0] = 99
    assert t1[0].item() == 99  # changed
    assert t2[0].item() == 1  # unchanged (copy)

    # PyTorch to NumPy
    t = torch.tensor([1, 2, 3])
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)

    # Detach for grad tensors
    t_grad = torch.tensor([1.0], requires_grad=True)
    arr = t_grad.detach().numpy()
    assert isinstance(arr, np.ndarray)

    print("  [PASS] numpy_conversion")


def test_autograd_basic():
    """From part-01-autograd.md: Basic Autograd."""
    # requires_grad
    a = torch.tensor([1.0, 2.0, 3.0])
    assert not a.requires_grad

    b = torch.tensor([1.0, 2.0], requires_grad=True)
    assert b.requires_grad

    print("  [PASS] autograd_basic")


def test_backward():
    """From part-01-autograd.md: Computing Gradients."""
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2
    y.backward()
    assert x.grad.item() == 4.0  # dy/dx = 2x = 4

    # Multiple variables
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([3.0], requires_grad=True)
    y = x * w
    y.backward()
    assert x.grad.item() == 3.0  # dy/dx = w = 3
    assert w.grad.item() == 2.0  # dy/dw = x = 2

    # Chain rule
    x = torch.tensor([2.0], requires_grad=True)
    y = (x**2 + 1) ** 3
    y.backward()
    assert x.grad.item() == 300.0

    print("  [PASS] backward")


def test_gradient_accumulation():
    """From part-01-autograd.md: Gradient Accumulation."""
    x = torch.tensor([2.0], requires_grad=True)

    y1 = x**2
    y1.backward()
    assert x.grad.item() == 4.0

    # Gradients accumulate
    y2 = x**3
    y2.backward()
    assert x.grad.item() == 16.0  # 4 + 12

    # zero_grad
    x.grad.zero_()
    y3 = x**3
    y3.backward()
    assert x.grad.item() == 12.0

    print("  [PASS] gradient_accumulation")


def test_no_grad():
    """From part-01-autograd.md: Disabling Gradients."""
    x = torch.tensor([2.0], requires_grad=True)

    with torch.no_grad():
        y = x**2
        assert not y.requires_grad

    z = x**2
    assert z.requires_grad

    # detach
    y = x**2
    y_detached = y.detach()
    assert not y_detached.requires_grad

    print("  [PASS] no_grad")


def test_autograd_example():
    """From part-01-autograd.md: Complete Example."""
    w = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([3.0, 5.0, 7.0])

    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    assert w.grad is not None
    assert b.grad is not None

    print("  [PASS] autograd_example")


def main():
    """Run all verification tests."""
    print("Verifying Part 1 slide code snippets...\n")

    print("Tensor Basics:")
    test_tensor_creation()
    test_tensor_attributes()
    test_dtype_casting()

    print("\nOperations:")
    test_elementwise_operations()
    test_reduction_operations()
    test_matrix_operations()
    test_comparison_operations()

    print("\nBroadcasting:")
    test_broadcasting()

    print("\nReshaping:")
    test_reshaping()
    test_transpose_permute()
    test_views_copies()
    test_contiguous()

    print("\nIndexing:")
    test_basic_indexing()
    test_slicing()
    test_boolean_indexing()
    test_advanced_indexing()

    print("\nNumPy Comparison:")
    test_numpy_conversion()

    print("\nAutograd:")
    test_autograd_basic()
    test_backward()
    test_gradient_accumulation()
    test_no_grad()
    test_autograd_example()

    print("\n" + "=" * 50)
    print("All code snippets verified successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
