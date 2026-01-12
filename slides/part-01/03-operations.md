---
layout: section
---

# Tensor Operations

---
layout: default
---

# Element-wise Operations

Operations applied to each element independently.

<div class="grid grid-cols-2 gap-6">

<div>

### Arithmetic

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Using operators
a + b        # tensor([5, 7, 9])
a - b        # tensor([-3, -3, -3])
a * b        # tensor([4, 10, 18])
a / b        # tensor([0.25, 0.4, 0.5])
a ** 2       # tensor([1, 4, 9])

# Using functions (equivalent)
torch.add(a, b)
torch.sub(a, b)
torch.mul(a, b)
torch.div(a, b)
```

</div>

<div>

### With Scalars

```python
a = torch.tensor([1, 2, 3])

a + 10       # tensor([11, 12, 13])
a * 2        # tensor([2, 4, 6])
a / 2        # tensor([0.5, 1.0, 1.5])
```

### Mathematical Functions

```python
t = torch.tensor([1.0, 4.0, 9.0])

torch.sqrt(t)    # [1, 2, 3]
torch.exp(t)     # e^x for each x
torch.log(t)     # natural log
torch.abs(t)     # absolute value
torch.sin(t)     # trigonometric
```

</div>

</div>

---
layout: default
---

# Reduction Operations

Operations that reduce dimensions by aggregating values.

<div class="grid grid-cols-2 gap-6">

<div>

### Basic Reductions

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

t.sum()      # 21 (all elements)
t.mean()     # 3.5
t.max()      # 6
t.min()      # 1
t.prod()     # 720 (product)
```

### Along an Axis

```python
# Sum along rows (axis 0)
t.sum(dim=0)     # tensor([5, 7, 9])

# Sum along columns (axis 1)
t.sum(dim=1)     # tensor([6, 15])

# Keep dimensions
t.sum(dim=1, keepdim=True)  # shape (2, 1)
```

</div>

<div>

### Finding Indices

```python
t = torch.tensor([3, 1, 4, 1, 5])

t.argmax()       # 4 (index of max)
t.argmin()       # 1 (index of min)

# Along axis
m = torch.tensor([[1, 5, 2],
                  [4, 3, 6]])
m.argmax(dim=1)  # [1, 2] (per row)
```

### Statistical

```python
t = torch.tensor([1.0, 2.0, 3.0, 4.0])

t.std()      # standard deviation
t.var()      # variance
t.median()   # median value
```

</div>

</div>

---
layout: default
---

# Matrix Operations

<div class="grid grid-cols-2 gap-6">

<div>

### Matrix Multiplication

```python
a = torch.tensor([[1, 2],
                  [3, 4]])
b = torch.tensor([[5, 6],
                  [7, 8]])

# Three equivalent ways
a @ b                  # preferred
torch.matmul(a, b)     # explicit
torch.mm(a, b)         # 2D only

# Result:
# [[1*5+2*7, 1*6+2*8],
#  [3*5+4*7, 3*6+4*8]]
# = [[19, 22],
#    [43, 50]]
```

**Note**: `*` is element-wise, `@` is matrix multiplication.

</div>

<div>

### Other Matrix Operations

```python
m = torch.tensor([[1, 2],
                  [3, 4]], dtype=torch.float32)

# Transpose
m.T              # or m.t() or m.transpose(0, 1)

# Inverse (square matrices)
torch.inverse(m)

# Determinant
torch.det(m)     # -2.0

# Matrix-vector multiplication
v = torch.tensor([1, 2])
m @ v            # tensor([5, 11])
```

### Dot Product (vectors)

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
torch.dot(a, b)  # 32 (1*4 + 2*5 + 3*6)
```

</div>

</div>

---
layout: default
---

# Comparison Operations

<div class="grid grid-cols-2 gap-6">

<div>

### Element-wise Comparisons

```python
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([2, 2, 2, 2])

a > b      # tensor([False, False, True, True])
a >= b     # tensor([False, True, True, True])
a < b      # tensor([True, False, False, False])
a == b     # tensor([False, True, False, False])
a != b     # tensor([True, False, True, True])
```

### With Scalars

```python
a > 2      # tensor([False, False, True, True])
```

</div>

<div>

### Logical Operations

```python
x = torch.tensor([True, True, False])
y = torch.tensor([True, False, False])

x & y      # tensor([True, False, False])
x | y      # tensor([True, True, False])
~x         # tensor([False, False, True])
```

### torch.where

```python
a = torch.tensor([1, 2, 3, 4])

# Select based on condition
torch.where(a > 2, a, torch.zeros_like(a))
# tensor([0, 0, 3, 4])

# Useful for conditional assignment
torch.where(a > 2, a * 10, a)
# tensor([1, 2, 30, 40])
```

</div>

</div>
