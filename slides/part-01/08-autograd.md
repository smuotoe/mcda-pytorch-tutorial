---
layout: image-right
image: /images/08-autograd.jpg
backgroundSize: cover
backgroundPosition: center
class: flex flex-col justify-center
---

# Automatic Differentiation (Autograd)

---
layout: default
---

# Why Do We Need Gradients?

Neural networks learn by adjusting weights to minimize a loss function.

<div class="mt-4">

```text
                    Forward Pass
Input  --->  [ Weights ]  --->  Prediction  --->  Loss
                  ^                                 |
                  |         Backward Pass           |
                  +---- [ Gradients ] <-------------+
```

</div>

<div class="grid grid-cols-2 gap-6 mt-4">

<div>

### The Learning Process
1. Make a prediction (forward pass)
2. Compute how wrong it is (loss)
3. Figure out how to adjust weights (gradients)
4. Update weights (optimization)
5. Repeat

</div>

<div>

### Gradients Tell Us
- **Direction**: Which way to adjust each weight
- **Magnitude**: How much to adjust

Without automatic differentiation, you'd compute gradients by hand - impractical for millions of parameters.

</div>

</div>

---
layout: default
---

# requires_grad: Tracking Computations

<div class="grid grid-cols-2 gap-6">

<div>

### Enabling Gradient Tracking

```python
# By default, no tracking
a = torch.tensor([1.0, 2.0, 3.0])
print(a.requires_grad)  # False

# Enable tracking at creation
b = torch.tensor([1.0, 2.0], requires_grad=True)
print(b.requires_grad)  # True

# Enable on existing tensor
a.requires_grad_(True)  # in-place
# or
a = a.requires_grad_(True)
```

</div>

<div>

### What Gets Tracked?

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2      # y = 4.0
z = y * 3       # z = 12.0

# PyTorch builds a computation graph
print(y.grad_fn)  # <PowBackward0>
print(z.grad_fn)  # <MulBackward0>

# z "remembers" how it was computed
# This graph enables automatic differentiation
```

### Only for Floats

```python
# Integer tensors can't track gradients
t = torch.tensor([1, 2], requires_grad=True)
# RuntimeError: only floating point tensors
```

</div>

</div>

---
layout: default
---

# Computing Gradients with backward()

<div class="grid grid-cols-2 gap-6">

<div>

### Simple Example

```python
x = torch.tensor([2.0], requires_grad=True)

# Forward pass: y = x^2
y = x ** 2

# Backward pass: compute dy/dx
y.backward()

# Access gradient
print(x.grad)  # tensor([4.0])
# dy/dx = 2x, at x=2: 2*2 = 4
```

### With Multiple Variables

```python
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)

y = x * w  # y = 6.0
y.backward()

print(x.grad)  # tensor([3.0]) - dy/dx = w = 3
print(w.grad)  # tensor([2.0]) - dy/dw = x = 2
```

</div>

<div>

### Chain Rule in Action

```python
x = torch.tensor([2.0], requires_grad=True)

# y = (x^2 + 1)^3
y = (x ** 2 + 1) ** 3

y.backward()
print(x.grad)  # tensor([300.0])

# Manual calculation:
# Let u = x^2 + 1
# y = u^3
# dy/dx = dy/du * du/dx
#       = 3u^2 * 2x
#       = 3(x^2+1)^2 * 2x
#       = 3(5)^2 * 4 = 300
```

</div>

</div>

---
layout: default
---

# The Computation Graph

PyTorch builds a **dynamic computation graph** as operations execute.

```python
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

z = x * y        # z depends on x and y
out = z.sum()    # scalar output

out.backward()   # traverse graph backwards
```

<div class="grid grid-cols-2 gap-8 mt-4">

<div class="text-center p-4">

```text
     x          y
     |          |
     +----+-----+
          |
        z = x*y
          |
       out = sum
```

**Forward**: values flow down

</div>

<div class="text-center p-4">

```text
  x.grad      y.grad
     ^          ^
     +----+-----+
          |
       dz/d(x,y)
          ^
       dout/dz = 1
```

**Backward**: gradients flow up

</div>

</div>

---
layout: default
---

# Gradient Accumulation and zero_grad()

Gradients **accumulate** by default - they add up across multiple `backward()` calls.

<div class="grid grid-cols-2 gap-6">

<div>

### The Accumulation "Problem"

```python
x = torch.tensor([2.0], requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor([4.0])

# Second backward (gradients ADD)
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor([16.0]) - not 12!
# 4.0 (from y1) + 12.0 (from y2) = 16.0
```

</div>

<div>

### The Solution: zero_grad()

```python
x = torch.tensor([2.0], requires_grad=True)

# First backward
y1 = x ** 2
y1.backward()
print(x.grad)  # tensor([4.0])

# Clear gradients before next backward
x.grad.zero_()  # or x.grad = None

# Second backward (fresh start)
y2 = x ** 3
y2.backward()
print(x.grad)  # tensor([12.0]) - correct!
```

</div>

</div>

<div class="mt-4 text-sm">

**Why accumulate by default?** Useful for gradient accumulation across mini-batches when you can't fit a large batch in memory.

</div>

---
layout: default
---

# Disabling Gradient Tracking

Sometimes you don't want gradients (inference, evaluation, preprocessing).

<div class="grid grid-cols-2 gap-6">

<div>

### torch.no_grad() Context

```python
x = torch.tensor([2.0], requires_grad=True)

# Inside no_grad, operations don't track
with torch.no_grad():
    y = x ** 2
    print(y.requires_grad)  # False

# Back to tracking outside
z = x ** 2
print(z.requires_grad)  # True
```

### Why Use no_grad()?

- **Faster**: No graph construction overhead
- **Less memory**: No saved tensors for backward
- **Required for**: Evaluation, inference, weight updates

</div>

<div>

### detach()

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2

# Detach from graph
y_detached = y.detach()
print(y_detached.requires_grad)  # False

# y_detached shares data but no gradients
```

### torch.inference_mode() (PyTorch 1.9+)

```python
# Even faster than no_grad for pure inference
with torch.inference_mode():
    y = model(x)
# Can't even enable requires_grad inside
```

</div>

</div>

---
layout: default
---

# Autograd: Putting It Together

A complete example showing the gradient computation workflow:

```python
# Simulate learning a simple function: y = 2x + 1
# We'll learn the weight (w) and bias (b)

w = torch.tensor([0.0], requires_grad=True)  # start at 0
b = torch.tensor([0.0], requires_grad=True)  # start at 0

x = torch.tensor([1.0, 2.0, 3.0])  # inputs
y_true = torch.tensor([3.0, 5.0, 7.0])  # targets (2*x + 1)

# Training step
y_pred = w * x + b                  # forward pass
loss = ((y_pred - y_true) ** 2).mean()  # MSE loss

loss.backward()                     # compute gradients

print(f"w.grad: {w.grad}")          # gradient w.r.t. w
print(f"b.grad: {b.grad}")          # gradient w.r.t. b

# In real training, we'd update: w = w - lr * w.grad
# That's what optimizers do (next week!)
```

<div class="mt-2 text-sm">

This is the foundation of all neural network training - next week we'll see how `nn.Module` and optimizers build on this.

</div>
