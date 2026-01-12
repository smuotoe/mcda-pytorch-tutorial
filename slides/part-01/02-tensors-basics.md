---
layout: image-right
image: /images/02-tensors.jpg
backgroundSize: cover
backgroundPosition: center
class: flex flex-col justify-center
---

# Tensors: The Core Data Structure

---
layout: default
---

# What is a Tensor?

A **tensor** is an n-dimensional array - the fundamental data structure in PyTorch.

<div class="grid grid-cols-4 gap-4 mt-8 text-center">

<div class="p-4 border rounded">

### Scalar
0-dimensional
```python
torch.tensor(5)
```
Shape: <code>()</code>

</div>

<div class="p-4 border rounded">

### Vector
1-dimensional
```python
torch.tensor([1, 2, 3])
```
Shape: <code>(3,)</code>

</div>

<div class="p-4 border rounded">

### Matrix
2-dimensional
```python
torch.tensor([
    [1, 2],
    [3, 4]
])
```
Shape: <code>(2, 2)</code>

</div>

<div class="p-4 border rounded">

### 3D Tensor
3-dimensional
```python
torch.rand(2, 3, 4)
```
Shape: <code>(2, 3, 4)</code>

</div>

</div>

<div class="mt-8 text-sm">

**Real-world examples**: Images are 3D tensors (height x width x channels), batches of images are 4D tensors.

</div>

---
layout: default
---

# Creating Tensors

<div class="grid grid-cols-2 gap-6">

<div>

### From Data

```python
import torch

# From a Python list
t1 = torch.tensor([1, 2, 3])

# From nested lists (matrix)
t2 = torch.tensor([[1, 2], [3, 4]])

# From NumPy array
import numpy as np
arr = np.array([1, 2, 3])
t3 = torch.from_numpy(arr)
```

</div>

<div>

### Factory Functions

```python
# Zeros and ones
zeros = torch.zeros(3, 4)      # 3x4 of zeros
ones = torch.ones(2, 3)        # 2x3 of ones

# Random values
rand = torch.rand(2, 2)        # uniform [0, 1)
randn = torch.randn(2, 2)      # normal dist

# Sequences
seq = torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
lin = torch.linspace(0, 1, 5)  # 5 points 0 to 1

# Like another tensor
like = torch.zeros_like(rand)  # same shape
```

</div>

</div>

---
layout: default
---

# Tensor Attributes

Every tensor has three key attributes:

```python
t = torch.rand(3, 4)

print(t.shape)   # torch.Size([3, 4]) - dimensions
print(t.dtype)   # torch.float32 - data type
print(t.device)  # cpu - where it lives
```

<div class="grid grid-cols-3 gap-6 mt-6">

<div class="p-3 border rounded">

### Shape
The size of each dimension.

```python
t.shape      # torch.Size([3, 4])
t.size()     # same thing
t.ndim       # 2 (number of dims)
t.numel()    # 12 (total elements)
```

</div>

<div class="p-3 border rounded">

### Data Type (dtype)
Numeric precision.

```python
torch.float32  # default for floats
torch.float64  # double precision
torch.int64    # default for ints
torch.int32    # 32-bit integer
torch.bool     # boolean
```

</div>

<div class="p-3 border rounded">

### Device
CPU or GPU location.

```python
t.device           # cpu
t.to('cuda')       # move to GPU
t.to('cpu')        # move to CPU
t.cuda()           # shorthand
t.cpu()            # shorthand
```

</div>

</div>

---
layout: default
---

# Data Types and Casting

<div class="text-sm opacity-70 mb-3">

**Data type (dtype):** How values are stored in memory, trading precision for speed/memory. **Casting:** Converting a tensor from one dtype to another.

</div>

<div class="grid grid-cols-2 gap-4 text-sm">

<div>

### Common dtypes

| Type | Size | Use Case |
|------|------|----------|
| `float32` | 4 bytes | Default for neural nets |
| `float16` | 2 bytes | Mixed precision training |
| `bfloat16` | 2 bytes | Same range as float32, less precision |
| `float64` | 8 bytes | High precision math |
| `int64` | 8 bytes | Indices, labels |
| `int32` | 4 bytes | Smaller indices |
| `bool` | 1 byte | Masks |

<div class="text-xs opacity-70 mt-2">

float16/bfloat16 halve memory and speed up GPU training.

</div>

</div>

<div>

### Casting Between Types

```python
t = torch.tensor([1, 2, 3])

# Using .to()
t.to(torch.float32)
t.to(torch.float16)
t.to(torch.bfloat16)

# Shorthand methods
t.float()   # float32
t.half()    # float16
t.double()  # float64
t.int()     # int32
t.long()    # int64
```

<div class="text-xs opacity-70 mt-2">

**Warning:** Casting to int truncates, doesn't round: `tensor([1.7]).int()` gives `[1]`

</div>

</div>

</div>
