---
layout: image-right
image: /images/07-numpy.jpg
backgroundSize: contain
backgroundPosition: center
class: flex flex-col justify-center
---

# PyTorch vs NumPy

---
layout: default
---

# Similarities

PyTorch's tensor API is heavily inspired by NumPy.

<div class="grid grid-cols-2 gap-6">

<div>

### Creation

```python
# NumPy
np.array([1, 2, 3])
np.zeros((3, 4))
np.ones((2, 3))
np.arange(10)
np.linspace(0, 1, 5)

# PyTorch
torch.tensor([1, 2, 3])
torch.zeros(3, 4)
torch.ones(2, 3)
torch.arange(10)
torch.linspace(0, 1, 5)
```

</div>

<div>

### Operations

```python
# NumPy
a + b
a * b
a @ b
np.sum(a)
np.mean(a)
a.reshape(3, 4)

# PyTorch
a + b
a * b
a @ b
torch.sum(a)  # or a.sum()
torch.mean(a) # or a.mean()
a.reshape(3, 4)
```

</div>

</div>

<div class="mt-4 text-sm">

**Most NumPy code translates directly to PyTorch** - just replace `np` with `torch` and `array` with `tensor`.

</div>

---
layout: default
---

# Key Differences

<div class="grid grid-cols-2 gap-6">

<div>

### GPU Support

```python
# NumPy - CPU only
a = np.array([1, 2, 3])

# PyTorch - CPU or GPU
a = torch.tensor([1, 2, 3])
a_gpu = a.to('cuda')  # move to GPU
a_gpu = a.cuda()      # shorthand
```

### Automatic Differentiation

```python
# NumPy - no gradients
a = np.array([1.0, 2.0])
# Can't compute gradients

# PyTorch - gradients tracked
a = torch.tensor([1.0, 2.0], requires_grad=True)
b = (a ** 2).sum()
b.backward()
print(a.grad)  # tensor([2., 4.])
```

</div>

<div>

### In-place Operations

```python
# PyTorch has explicit in-place ops (trailing _)
a = torch.tensor([1, 2, 3])
a.add_(10)     # modifies a in-place
a.mul_(2)      # modifies a in-place
a.zero_()      # sets all to zero

# Be careful with autograd!
a = torch.tensor([1.0], requires_grad=True)
a.add_(1)      # RuntimeError if gradient needed
```

### Naming Conventions

```python
# NumPy          # PyTorch
np.ndarray       torch.Tensor
a.ndim           a.dim()
np.concatenate   torch.cat
np.stack         torch.stack
axis=0           dim=0
```

</div>

</div>

---
layout: default
---

# Converting Between NumPy and PyTorch

<div class="grid grid-cols-2 gap-6">

<div>

### NumPy to PyTorch

```python
import numpy as np
import torch

arr = np.array([1, 2, 3])

# Method 1: from_numpy (shares memory!)
t1 = torch.from_numpy(arr)

# Method 2: torch.tensor (copies data)
t2 = torch.tensor(arr)

# Shared memory means changes propagate
arr[0] = 99
print(t1)  # tensor([99, 2, 3]) - changed!
print(t2)  # tensor([1, 2, 3]) - unchanged
```

</div>

<div>

### PyTorch to NumPy

```python
t = torch.tensor([1, 2, 3])

# Method 1: .numpy() (shares memory on CPU)
arr1 = t.numpy()

# If tensor is on GPU, must move to CPU first
t_gpu = t.cuda()
arr2 = t_gpu.cpu().numpy()

# If tensor requires grad, must detach first
t_grad = torch.tensor([1.0], requires_grad=True)
arr3 = t_grad.detach().numpy()
```

### Common Pattern

```python
# Safe conversion (always works)
arr = tensor.detach().cpu().numpy()
```

</div>

</div>
