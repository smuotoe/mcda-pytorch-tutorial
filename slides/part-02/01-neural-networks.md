---
layout: image-right
image: /images/01-neural-networks.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# Neural Network Basics

What they are and how they learn

---
layout: default
---

# What is a Neural Network?

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### The Concept

A neural network is a **function approximator** composed of:

- **Layers** of interconnected nodes (neurons)
- **Weights** that are learned from data
- **Activation functions** that add non-linearity

<div class="mt-4">

### Why "Neural"?

Loosely inspired by biological neurons, but the math is what matters:

$$y = f(Wx + b)$$

Where $W$ = weights, $b$ = bias, $f$ = activation

</div>

</div>

<div>

```mermaid {scale: 0.75}
flowchart LR
    subgraph Input
        x1((x1))
        x2((x2))
        x3((x3))
    end

    subgraph Hidden
        h1((h1))
        h2((h2))
    end

    subgraph Output
        y1((y))
    end

    x1 --> h1
    x1 --> h2
    x2 --> h1
    x2 --> h2
    x3 --> h1
    x3 --> h2
    h1 --> y1
    h2 --> y1
```

<div class="text-center text-sm opacity-70 mt-2">

A simple feedforward network with one hidden layer

</div>

</div>

</div>

---
layout: default
---

# How Neural Networks Learn

<div class="mt-2">

```mermaid {scale: 0.9}
flowchart LR
    A[Input] --> B[Model]
    B --> C[Prediction]
    C --> D{Loss}
    E[Label] --> D
    D --> |backward| F[Gradients]
    F --> G[Optimizer]
    G --> |"step<br/>(update weights)"| B

    style D fill:#f97316,stroke:#c2410c,color:#fff
    style G fill:#3b82f6,stroke:#1d4ed8,color:#fff
```

</div>

<div class="grid grid-cols-4 gap-4 mt-6 text-sm">

<div class="p-3 border rounded text-center">

**1. Forward**

Pass input through model to get prediction

</div>

<div class="p-3 border rounded text-center">

**2. Loss**

Compare prediction to true label

</div>

<div class="p-3 border rounded text-center">

**3. Backward**

Compute gradients via autograd

</div>

<div class="p-3 border rounded text-center">

**4. Update**

Optimizer adjusts weights

</div>

</div>

<div class="mt-4 text-center text-sm opacity-70">

This cycle repeats until the model learns the patterns in the data.

</div>
