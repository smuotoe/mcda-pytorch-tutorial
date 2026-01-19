---
layout: image-right
image: /images/04-optimizers.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# Optimizers

Updating weights to minimize loss

---
layout: default
---

# What is an Optimizer?

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### The Role of an Optimizer

An optimizer **updates model parameters** using gradients to minimize loss.

### The Update Rule

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \cdot \nabla L$$

<div class="text-sm mt-2">

- $\theta$ = model parameters (weights)
- $\alpha$ = learning rate (step size)
- $\nabla L$ = gradient (direction of steepest increase)

</div>

</div>

<div>

<div class="p-3 bg-blue-50 dark:bg-blue-900/30 rounded text-sm">

**Intuition**: Imagine you're blindfolded on a hilly landscape, trying to find the lowest point. The gradient tells you which way is uphill. To go downhill, you step in the **opposite direction** (the minus sign). The learning rate is how big each step is.

</div>

<div class="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/30 rounded text-sm">

**Key insight**: Every weight in your network gets its own gradient. Some weights need big adjustments, others tiny tweaks. The optimizer handles this automatically.

</div>

</div>

</div>

---
layout: default
---

# Using an Optimizer

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### The Three-Step Dance

Every training iteration follows this pattern:

<div class="mt-4 space-y-3">

<div class="flex items-start gap-3">
<div class="w-7 h-7 rounded-full bg-gray-500 text-white flex items-center justify-center text-sm font-bold flex-shrink-0">1</div>
<div><strong>zero_grad()</strong> - Clear old gradients (they accumulate by default!)</div>
</div>

<div class="flex items-start gap-3">
<div class="w-7 h-7 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm font-bold flex-shrink-0">2</div>
<div><strong>backward()</strong> - Compute gradients via backpropagation</div>
</div>

<div class="flex items-start gap-3">
<div class="w-7 h-7 rounded-full bg-green-500 text-white flex items-center justify-center text-sm font-bold flex-shrink-0">3</div>
<div><strong>step()</strong> - Update weights using the gradients</div>
</div>

</div>

</div>

<div>

### Code Example

```python
import torch.optim as optim

# Create optimizer (pass model parameters)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step
optimizer.zero_grad()   # 1. Clear old gradients
loss = criterion(model(x), y)
loss.backward()         # 2. Compute gradients
optimizer.step()        # 3. Update weights
```

<div class="mt-3 p-2 bg-green-50 dark:bg-green-900/30 rounded text-xs">

**Remember**: Always call `zero_grad()` before `backward()`, or gradients will accumulate across batches!

</div>

</div>

</div>

---
layout: default
---

# Common Optimizers

<div class="grid grid-cols-3 gap-4 mt-4">

<div class="p-4 border rounded">

### SGD

**Stochastic Gradient Descent**

```python
optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

<div class="text-sm mt-2">

- Simple and well-understood
- Often needs momentum
- Good for fine-tuning
- Can generalize better

</div>

</div>

<div class="p-4 border rounded bg-blue-50 dark:bg-blue-900/20">

### Adam

**Adaptive Moment Estimation**

```python
optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)
```

<div class="text-sm mt-2">

- **Most popular choice**
- Adapts learning rate per-param
- Works well out of the box
- Good default for beginners

</div>

</div>

<div class="p-4 border rounded">

### AdamW

**Adam with Weight Decay**

```python
optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)
```

<div class="text-sm mt-2">

- Better regularization
- Preferred for transformers
- Decoupled weight decay
- Current best practice

</div>

</div>

</div>

<div class="mt-4 text-center text-sm opacity-70">

**Rule of thumb**: Start with Adam (lr=0.001). Switch to AdamW for larger models or if overfitting.

</div>

---
layout: default
---

# Learning Rate: The Most Important Hyperparameter

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### What is Learning Rate?

The learning rate ($\alpha$) controls **step size** during optimization.

<div class="mt-4 text-sm">

| Learning Rate | Effect |
|--------------|--------|
| Too high | Overshoots, loss explodes |
| Too low | Very slow training |
| Just right | Smooth convergence |

</div>

<div class="mt-4 text-sm">

### Typical Starting Points

- **Adam**: `lr=0.001` (1e-3)
- **SGD**: `lr=0.01` to `0.1`
- **Fine-tuning**: `lr=1e-5` to `1e-4`

</div>

</div>

<div>

### Learning Rate Schedulers

For advanced training, you can **adjust the learning rate** during training using schedulers.

<div class="mt-3 text-sm">

| Scheduler | What it Does |
|-----------|--------------|
| `StepLR` | Reduce LR every N epochs |
| `ReduceLROnPlateau` | Reduce when loss stops improving |
| `CosineAnnealingLR` | Smooth decay following cosine curve |

</div>

<div class="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded text-sm">

```python
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Call after each epoch
scheduler.step()
```

</div>

<div class="mt-3 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-xs">

**For beginners**: Start without a scheduler. Add one later if training plateaus.

</div>

</div>

</div>
