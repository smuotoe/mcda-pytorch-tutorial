---
theme: default
title: "Introduction to PyTorch"
info: |
  ## MCDA 5511: Introduction to PyTorch
  Part 1: Tensors, Operations & Autograd
author: Somto Muotoe
keywords: pytorch,deep-learning,tensors,neural-networks
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
  <h1 class="!text-5xl font-bold tracking-tight">Introduction to PyTorch</h1>
  <p class="text-xl opacity-70 mt-4">Part 1: Tensors, Operations & Autograd</p>
</div>

<div class="absolute bottom-14 left-1/2 -translate-x-1/2 text-center">
  <div class="text-base font-medium">Somto Muotoe</div>
  <div class="text-sm opacity-50 mt-1">{{ new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' }) }}</div>
</div>

---
layout: default
---

# Tutorial Overview

This is a **two-part series** on PyTorch fundamentals.

<div class="grid grid-cols-2 gap-8 mt-8">

<div class="p-4 border rounded-lg">

### Part 1: Foundations (This Week)
- What is PyTorch?
- Tensors: creation, attributes, types
- Tensor operations & broadcasting
- Reshaping & memory
- Indexing & slicing
- PyTorch vs NumPy
- Automatic differentiation (Autograd)

</div>

<div class="p-4 border rounded-lg">

### Part 2: Neural Networks (Next Week)
- The `nn.Module` class
- Building network architectures
- Loss functions & optimizers
- The training loop
- Data handling with DataLoader
- Model evaluation

</div>

</div>

---
layout: default
---

# Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/smuotoe/mcda-pytorch-tutorial.git
cd mcda-pytorch-tutorial
uv sync
```

<div class="mt-8">

### Prerequisites
- Python 3.10+
- Basic Python programming
- Familiarity with NumPy is helpful but not required

</div>

<div class="mt-8">

### Resources
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch 60-Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

</div>

---
src: ./01-introduction.md
---

---
src: ./02-tensors-basics.md
---

---
src: ./03-operations.md
---

---
src: ./04-broadcasting.md
---

---
src: ./05-reshaping.md
---

---
src: ./06-indexing.md
---

---
src: ./07-numpy-comparison.md
---

---
src: ./08-autograd.md
---

---
src: ./09-summary.md
---

---
layout: image-right
image: /images/09-closing.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# End of Part 1

<div class="text-2xl font-semibold text-orange-500 mt-4">
Next Week: Building & Training Neural Networks
</div>

<div class="mt-6 space-y-3 text-sm">

<div class="flex items-center gap-3">
<div class="w-2 h-2 rounded-full bg-blue-500"></div>
<span><strong>nn.Module</strong> - Define layers & forward pass</span>
</div>

<div class="flex items-center gap-3">
<div class="w-2 h-2 rounded-full bg-orange-500"></div>
<span><strong>Training Loop</strong> - Forward, loss, backward, step</span>
</div>

<div class="flex items-center gap-3">
<div class="w-2 h-2 rounded-full bg-green-500"></div>
<span><strong>Optimizers</strong> - SGD, Adam & learning rates</span>
</div>

</div>

<div class="mt-8 text-base opacity-80">
Questions? Bring them to the session!
</div>
