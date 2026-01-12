---
layout: image-right
image: /images/06-indexing.jpg
backgroundSize: cover
backgroundPosition: center
class: flex flex-col justify-center
---

# Indexing and Slicing

---
layout: default
---

# Basic Indexing

PyTorch indexing works like NumPy - zero-indexed, supports negative indices.

<div class="grid grid-cols-2 gap-6">

<div>

### 1D Indexing

```python
t = torch.tensor([10, 20, 30, 40, 50])

t[0]      # 10 (first element)
t[-1]     # 50 (last element)
t[2]      # 30 (third element)
```

### 2D Indexing

```python
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

m[0, 0]    # 1 (top-left)
m[1, 2]    # 6 (row 1, col 2)
m[-1, -1]  # 9 (bottom-right)

# Get entire row/column
m[0]       # tensor([1, 2, 3])
m[:, 0]    # tensor([1, 4, 7])
```

</div>

<div>

### Slicing Syntax

```python
# [start:stop:step] - stop is exclusive

t = torch.arange(10)  # [0, 1, 2, ..., 9]

t[2:5]     # [2, 3, 4]
t[:3]      # [0, 1, 2]
t[7:]      # [7, 8, 9]
t[::2]     # [0, 2, 4, 6, 8]
t[::-1]    # [9, 8, 7, ..., 0] (reversed)
```

### 2D Slicing

```python
m = torch.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

m[:2, :2]    # top-left 2x2
m[1:, 2:]    # bottom-right 2x2
m[:, ::2]    # every other column
```

</div>

</div>

---
layout: default
---

# Boolean Indexing (Masking)

Select elements based on a condition.

<div class="grid grid-cols-2 gap-6">

<div>

### Creating Masks

```python
t = torch.tensor([1, 5, 3, 8, 2, 9])

# Condition creates boolean tensor
mask = t > 4
print(mask)  # [False, True, False, True, False, True]

# Use mask to select elements
t[mask]      # tensor([5, 8, 9])

# Or directly
t[t > 4]     # tensor([5, 8, 9])
```

### Multiple Conditions

```python
# Combine with & (and), | (or)
t[(t > 2) & (t < 8)]  # [5, 3]
t[(t < 3) | (t > 7)]  # [1, 8, 2, 9]
```

</div>

<div>

### 2D Masking

```python
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Mask flattens the result
m[m > 5]     # tensor([6, 7, 8, 9])

# Mask specific rows/columns
row_mask = torch.tensor([True, False, True])
m[row_mask]  # rows 0 and 2
```

### Assigning with Masks

```python
t = torch.tensor([1, 5, 3, 8, 2])
t[t > 4] = 0
print(t)     # tensor([1, 0, 3, 0, 2])
```

</div>

</div>

---
layout: default
---

# Advanced Indexing

<div class="grid grid-cols-2 gap-6">

<div>

### Index with List/Tensor

```python
t = torch.tensor([10, 20, 30, 40, 50])
indices = torch.tensor([0, 2, 4])

t[indices]   # tensor([10, 30, 50])

# With lists
t[[0, 2, 4]] # same result
```

### 2D Fancy Indexing

```python
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Select specific elements
rows = torch.tensor([0, 1, 2])
cols = torch.tensor([2, 1, 0])
m[rows, cols]  # tensor([3, 5, 7])

# Select specific rows
m[[0, 2]]      # rows 0 and 2
```

</div>

<div>

### torch.where for Conditional Selection

```python
a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([10, 20, 30, 40, 50])

# Choose from a or b based on condition
torch.where(a > 3, a, b)
# tensor([10, 20, 30, 4, 5])

# Find indices where condition is true
torch.where(a > 3)
# (tensor([3, 4]),)  # indices 3 and 4
```

### gather (for specific selections)

```python
t = torch.tensor([[1, 2], [3, 4]])
idx = torch.tensor([[0, 0], [1, 0]])

torch.gather(t, dim=1, index=idx)
# [[1, 1], [4, 3]]  # per-row column selection
```

</div>

</div>
