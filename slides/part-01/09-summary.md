---
layout: section
---

# Exercise Preview

---
layout: default
---

# Part 1 Exercises

Practice what you've learned with hands-on tensor exercises.

<div class="grid grid-cols-2 gap-6">

<div>

### Exercise 1: Creating Tensors
- Create tensors from lists
- Use factory functions
- Specify data types

### Exercise 2: Tensor Operations
- Element-wise arithmetic
- Matrix multiplication
- Reduction operations

### Exercise 3: Reshaping
- Reshape tensors
- Understand views vs copies
- Use squeeze/unsqueeze

</div>

<div>

### Exercise 4: Indexing & Slicing
- Basic indexing
- Slicing patterns
- Boolean masking

### Exercise 5: Autograd Basics
- Enable gradient tracking
- Compute gradients
- Use no_grad context

</div>

</div>

<div class="mt-6 p-4 border rounded">

**File**: `exercises/01-tensor-operations.py`

Run with: `python exercises/01-tensor-operations.py`

</div>

---
layout: default
---

# Summary: Part 1

<div class="grid grid-cols-2 gap-6">

<div>

### What We Covered

- **Tensors** are n-dimensional arrays, PyTorch's core data structure
- **Creation**: `torch.tensor()`, `zeros()`, `ones()`, `rand()`, etc.
- **Attributes**: `shape`, `dtype`, `device`
- **Operations**: element-wise, reductions, matrix ops
- **Broadcasting**: automatic shape expansion
- **Reshaping**: `view`, `reshape`, `squeeze`, `unsqueeze`
- **Memory**: views share data, clones copy
- **Indexing**: NumPy-style, boolean masks, advanced indexing

</div>

<div>

### Key Takeaways

- **Autograd** tracks operations and computes gradients automatically
- `requires_grad=True` enables tracking
- `backward()` computes gradients via chain rule
- `zero_grad()` clears accumulated gradients
- `torch.no_grad()` disables tracking for efficiency

### Next Week (Part 2)

- Building neural networks with `nn.Module`
- Loss functions and optimizers
- The complete training loop
- Model evaluation

</div>

</div>
