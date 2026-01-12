---
layout: image-right
image: /images/04-broadcasting.jpg
backgroundSize: cover
backgroundPosition: center
class: flex flex-col justify-center
---

# Broadcasting

---
layout: default
---

# What is Broadcasting?

Broadcasting allows operations on tensors of **different shapes** by automatically expanding dimensions.

```python
# Without broadcasting - shapes must match exactly
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 10, 10])
a + b  # tensor([11, 12, 13])

# With broadcasting - scalar expands to match
a = torch.tensor([1, 2, 3])
a + 10  # tensor([11, 12, 13])  # 10 broadcasts to [10, 10, 10]
```

<div class="mt-6">

### Why Broadcasting Matters
- **Memory efficient**: No need to create expanded copies
- **Concise code**: Express operations naturally
- **Performance**: Optimized at C++ level

</div>

---
layout: default
---

# Broadcasting Rules

Two tensors are **broadcastable** if, for each dimension (starting from the trailing dimension):
1. The dimensions are equal, OR
2. One of them is 1, OR
3. One of them doesn't exist (missing dimensions are treated as 1)

<div class="grid grid-cols-2 gap-6 mt-4">

<div>

### Valid Broadcasting

```python
# (3,) + (1,) -> (3,)
torch.tensor([1, 2, 3]) + torch.tensor([10])
# Result: [11, 12, 13]

# (2, 3) + (3,) -> (2, 3)
a = torch.ones(2, 3)
b = torch.tensor([1, 2, 3])
a + b
# [[2, 3, 4],
#  [2, 3, 4]]

# (2, 1) + (1, 3) -> (2, 3)
a = torch.tensor([[1], [2]])     # shape (2, 1)
b = torch.tensor([[10, 20, 30]]) # shape (1, 3)
a + b
# [[11, 21, 31],
#  [12, 22, 32]]
```

</div>

<div>

### Invalid Broadcasting

```python
# (3,) + (2,) -> ERROR
a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2])
a + b  # RuntimeError!
# Neither is 1, and they're not equal

# (2, 3) + (2,) -> ERROR
a = torch.ones(2, 3)
b = torch.tensor([1, 2])
a + b  # RuntimeError!
# Trailing dims: 3 vs 2 - incompatible
```

### Debugging Tip

```python
# Always check shapes when broadcasting fails
print(f"a.shape: {a.shape}")
print(f"b.shape: {b.shape}")
```

</div>

</div>

---
layout: default
---

# Broadcasting Patterns

<div class="grid grid-cols-2 gap-6">

<div>

### Row/Column Operations

```python
# Subtract row mean from each row
data = torch.tensor([[1, 2, 3],
                     [4, 5, 6]], dtype=torch.float32)

row_mean = data.mean(dim=1, keepdim=True)  # (2, 1)
centered = data - row_mean
# [[-1, 0, 1],
#  [-1, 0, 1]]

# Divide each column by its max
col_max = data.max(dim=0).values  # (3,)
normalized = data / col_max
```

</div>

<div>

### Outer Product via Broadcasting

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([10, 20])

# Reshape to enable broadcasting
# (3, 1) * (1, 2) -> (3, 2)
outer = a.unsqueeze(1) * b.unsqueeze(0)
# [[10, 20],
#  [20, 40],
#  [30, 60]]
```

### Common Pitfall

```python
# Accidentally broadcasting when you didn't mean to
a = torch.rand(100, 1)
b = torch.rand(1, 100)
c = a + b  # Shape: (100, 100) - 10,000 elements!
```

</div>

</div>
