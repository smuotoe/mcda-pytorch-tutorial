---
theme: default
title: "Building & Training Neural Networks"
info: |
  ## MCDA 5511: Introduction to PyTorch
  Part 2: Building & Training Neural Networks
author: Somto Muotoe
keywords: pytorch,deep-learning,neural-networks,training
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

<div class="absolute top-12 left-1/2 -translate-x-1/2 text-center">
  <div class="text-xs uppercase tracking-[0.3em] opacity-50">MCDA 5511</div>
  <div class="text-sm opacity-60 mt-1">Current Practices in Computing and Data Science</div>
</div>

<div class="mt-16">
  <h1 class="!text-5xl font-bold tracking-tight">Building & Training Neural Networks</h1>
  <p class="text-xl opacity-70 mt-4">Part 2: nn.Module, Loss Functions & Training Loops</p>
</div>

<div class="absolute bottom-14 left-1/2 -translate-x-1/2 text-center">
  <div class="text-base font-medium">Somto Muotoe</div>
  <div class="text-sm opacity-50 mt-1">{{ new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' }) }}</div>
</div>

---
layout: default
---

# Part 2 Overview

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

### What We'll Cover

1. **Neural Network Basics** - What they are and how they work
2. **nn.Module** - PyTorch's building block for networks
3. **Loss Functions** - Measuring prediction errors
4. **Optimizers** - Updating weights with gradients
5. **Training Loop** - Putting it all together
6. **Model Evaluation** - Testing and saving models

</div>

<div>

### Prerequisites

You should be comfortable with:

- Creating and manipulating tensors
- Basic tensor operations
- Autograd and `backward()`
- `requires_grad` and gradient computation

<div class="mt-4 p-3 bg-blue-50 dark:bg-blue-900/30 rounded text-sm">

**Recap**: Part 1 covered tensors and autograd - the foundation for everything today.

</div>

</div>

</div>

---
src: ./01-neural-networks.md
---

---
src: ./02-nn-module.md
---

---
src: ./03-loss-functions.md
---

---
src: ./04-optimizers.md
---

---
src: ./05-training-loop.md
---

---
src: ./06-evaluation.md
---

---
src: ./07-summary.md
---
