# Introduction to PyTorch

**MCDA 5511: Current Practices in Data Science**

A two-part tutorial series on PyTorch fundamentals.

## Slides

- [Part 1: Tensors, Operations & Autograd](https://mcda-pytorch-tutorial.pages.dev)
- [Part 2: Building & Training Neural Networks](https://mcda-pytorch-part2.pages.dev)

## Part 1: Tensors, Operations & Autograd

- What is PyTorch?
- Tensors: creation, attributes, data types
- Tensor operations & broadcasting
- Reshaping & memory (views vs copies)
- Indexing & slicing
- Automatic differentiation (Autograd)

## Part 2: Building & Training Neural Networks

- The `nn.Module` class
- Building network architectures
- Loss functions & optimizers
- The training loop
- Model evaluation

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

```bash
git clone https://github.com/smuotoe/mcda-pytorch-tutorial.git
cd mcda-pytorch-tutorial
uv sync
```

This installs the **CPU-only** version of PyTorch (smaller download, works on all devices).

### CUDA Support (Optional)

If you have an NVIDIA GPU with CUDA support:

```bash
uv sync --extra cu124
```

## Exercises

Complete the hands-on exercises in Jupyter Notebooks:

```
exercises/pytorch-1-exercises.ipynb  # Part 1: Tensors & Autograd
exercises/pytorch-2-exercises.ipynb  # Part 2: Neural Networks
```

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
