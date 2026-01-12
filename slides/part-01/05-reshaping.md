---
layout: image-right
image: /images/05-reshaping.jpg
backgroundSize: cover
backgroundPosition: center
class: flex flex-col justify-center
---

# Reshaping and Memory

---
layout: default
---

# Reshaping Tensors

<div class="grid grid-cols-2 gap-6">

<div>

### reshape and view

```python
t = torch.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 3x4
t.reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Use -1 to infer dimension
t.reshape(3, -1)   # same as (3, 4)
t.reshape(-1, 6)   # shape (2, 6)

# view is similar but requires contiguous memory
t.view(3, 4)
```

### reshape vs view
- `view`: faster, but requires contiguous tensor
- `reshape`: always works, may copy data

</div>

<div>

### squeeze and unsqueeze

```python
# Remove dimensions of size 1
t = torch.zeros(1, 3, 1, 4)
t.squeeze().shape        # (3, 4)
t.squeeze(0).shape       # (3, 1, 4)
t.squeeze(2).shape       # (1, 3, 4)

# Add dimension of size 1
t = torch.zeros(3, 4)
t.unsqueeze(0).shape     # (1, 3, 4)
t.unsqueeze(1).shape     # (3, 1, 4)
t.unsqueeze(-1).shape    # (3, 4, 1)
```

### flatten

```python
t = torch.zeros(2, 3, 4)
t.flatten().shape        # (24,)
t.flatten(1).shape       # (2, 12) - keep first dim
```

</div>

</div>

---
layout: default
---

# Transpose and Permute

<div class="grid grid-cols-2 gap-6">

<div>

### Transpose (2D matrices)

```python
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape (2, 3)

m.T              # shape (3, 2)
m.t()            # same
m.transpose(0, 1)  # same

# Result:
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### transpose() on Higher Dimensions

```python
# transpose() swaps exactly two dimensions
t = torch.zeros(2, 3, 4, 5)
t.transpose(1, 3).shape  # (2, 5, 4, 3)
# Dims 1 and 3 swapped: 3<->5

# For 2D, these are equivalent:
m.T
m.transpose(0, 1)
```

</div>

<div>

### Permute (reorder all dimensions)

```python
t = torch.zeros(2, 3, 4)  # (batch, height, width)

# Specify new order of ALL dimensions
t.permute(0, 2, 1).shape  # (2, 4, 3)
t.permute(2, 1, 0).shape  # (4, 3, 2)
```

### Common Use: Channel Ordering

```python
# PyTorch default: channels first (C, H, W)
# Some libraries expect: channels last (H, W, C)

img = torch.zeros(3, 224, 224)  # (C, H, W)
img.permute(1, 2, 0).shape      # (H, W, C)

# Batch of images
batch = torch.zeros(32, 3, 224, 224)  # (N, C, H, W)
batch.permute(0, 2, 3, 1).shape       # (N, H, W, C)
```

</div>

</div>

---
layout: default
---

# Memory: Views vs Copies

Understanding when data is shared is crucial for performance and avoiding bugs.

<div class="grid grid-cols-2 gap-6">

<div>

### Views (Shared Memory)

```python
a = torch.tensor([1, 2, 3, 4])
b = a.view(2, 2)  # b is a VIEW of a

b[0, 0] = 99
print(a)  # tensor([99, 2, 3, 4]) - a changed!

# These create views:
# - view(), reshape() (usually)
# - transpose(), permute(), T
# - squeeze(), unsqueeze()
# - slicing: a[1:3]
# - expand()
```

</div>

<div>

### Copies (Independent Memory)

```python
a = torch.tensor([1, 2, 3, 4])
b = a.clone()  # b is a COPY

b[0] = 99
print(a)  # tensor([1, 2, 3, 4]) - unchanged

# These create copies:
# - clone()
# - contiguous() (when needed)
# - reshape() (sometimes, when non-contiguous)
# - to() when changing device/dtype
```

### Check if Shared

```python
a = torch.tensor([1, 2, 3, 4])
b = a.view(2, 2)
print(a.data_ptr() == b.data_ptr())  # True
```

</div>

</div>

---
layout: default
---

# Contiguous Memory

<div class="grid grid-cols-2 gap-6">

<div>

### What is Contiguous?

Tensor elements stored sequentially in memory, matching logical order.

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# Memory layout: [1, 2, 3, 4, 5, 6]
a.is_contiguous()  # True

b = a.T  # Transpose
# Logical: [[1, 4], [2, 5], [3, 6]]
# Memory still: [1, 2, 3, 4, 5, 6]
b.is_contiguous()  # False
```

</div>

<div>

### Why Does it Matter?

```python
b = a.T

# view() requires contiguous
b.view(6)  # RuntimeError!

# Two solutions:
b.reshape(6)              # works, may copy
b.contiguous().view(6)    # explicit copy
```

### Operations that Break Contiguity

```python
# These return non-contiguous views:
t.T
t.transpose(0, 1)
t.permute(...)
t[::2]  # stride slicing
```

</div>

</div>
