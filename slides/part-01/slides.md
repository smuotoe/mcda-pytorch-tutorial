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

# Introduction to PyTorch

## Part 1: Tensors, Operations & Autograd

MCDA 5511 | Current Practices in Data Science

<div class="abs-br m-6 text-sm opacity-50">
Somto Muotoe
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
layout: end
---

# End of Part 1

Next week: Building & Training Neural Networks

<div class="mt-8 text-sm opacity-70">

Questions? Bring them to the session!

</div>
