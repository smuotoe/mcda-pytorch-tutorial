---
layout: image-right
image: /images/02-nn-module.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# The nn.Module Class

PyTorch's building block for neural networks

---
layout: default
---

# What is nn.Module?

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### The Base Class

`nn.Module` is the **base class** for all neural network components in PyTorch.

<div class="mt-4">

### Key Responsibilities

- **Holds parameters** (weights & biases)
- **Defines forward pass** computation
- **Tracks submodules** (nested layers)
- **Provides utilities** for training/eval modes

</div>

<div class="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/30 rounded text-sm">

**Rule**: Every neural network you build will inherit from `nn.Module`.

</div>

</div>

<div>

```python
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers here
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

    def forward(self, x):
        # Define computation here
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

# Create an instance
model = MyNetwork()
```

<div class="text-sm opacity-70 mt-2">

Two required methods: `__init__` and `forward`

</div>

</div>

</div>

---
layout: default
---

# Anatomy of nn.Module

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### `__init__`: Define Your Layers

```python
def __init__(self):
    super().__init__()  # Always call this!

    # Define layers as attributes
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)

    # Activation (stateless, can reuse)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
```

<div class="mt-3 text-sm">

- Call `super().__init__()` first
- Assign layers as `self.` attributes
- PyTorch auto-registers parameters

</div>

</div>

<div>

### `forward`: Define the Computation

```python
def forward(self, x):
    # Input: (batch_size, 784)

    x = self.fc1(x)      # (batch, 256)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc2(x)      # (batch, 128)
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc3(x)      # (batch, 10)
    return x

# Call model like a function
output = model(input_tensor)
```

<div class="mt-3 text-sm">

- Defines data flow through layers
- Called automatically when you do `model(x)`
- Never call `model.forward(x)` directly

</div>

</div>

</div>

---
layout: default
---

# Introducing MNIST

<div class="grid grid-cols-2 gap-8 mt-2">

<div>

### The "Hello World" of Machine Learning

**MNIST** is a dataset of 70,000 handwritten digit images (0-9).

<div class="mt-3 text-sm">

| Property | Value |
|----------|-------|
| Images | 70,000 (60k train, 10k test) |
| Size | 28 x 28 pixels, grayscale |
| Classes | 10 (digits 0-9) |

</div>

<div class="mt-3 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-sm">

**Why MNIST?** Small, simple, and perfect for learning neural networks.

</div>

</div>

<div class="flex items-center justify-center">

<img src="/images/mnist-samples.png" class="rounded-lg shadow-lg max-h-80" />

</div>

</div>

---
layout: default
---

# Your First Neural Network

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### The Task

Build a classifier that takes a 28x28 image and predicts which digit (0-9) it represents.

<div class="mt-4">

**Input**: Image as tensor `(1, 28, 28)`

**Output**: 10 scores (one per digit)

**Prediction**: Digit with highest score

</div>

```python
# Image tensor shape: (channels, height, width)
# For MNIST: (1, 28, 28) - 1 channel, 28x28 pixels

# Flatten to vector: 1 * 28 * 28 = 784 features
# Output: 10 scores (one per digit)
```

</div>

<div>

### Building the Classifier

```python
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),        # (1,28,28) -> 784
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)    # 10 digits
        )

    def forward(self, x):
        return self.network(x)

model = MNISTClassifier()
# Total parameters: 109,386
```

</div>

</div>

---
layout: default
---

# Common Layer Types

<div class="grid grid-cols-2 gap-8 mt-4 text-sm">

<div>

### Core Layers

| Layer | Purpose |
|-------|---------|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.Conv2d(in, out, k)` | 2D convolution (images) |
| `nn.LSTM(in, hidden)` | Recurrent (sequences) |
| `nn.Embedding(vocab, dim)` | Word embeddings (NLP) |

<div class="mt-4 text-sm">

```python
# Fully connected: every input connects to every output
fc = nn.Linear(784, 256)  # 784 in, 256 out

# Convolution: slides filter over image
conv = nn.Conv2d(1, 32, kernel_size=3)
```

</div>

</div>

<div>

### How nn.Linear Works

<div class="text-sm">

A linear layer computes: **y = xW + b**

</div>

```python
# Input: 784 features, Output: 256 features
layer = nn.Linear(784, 256)

# Parameters created automatically:
# - weight: (256, 784) = 200,704 values
# - bias: (256,) = 256 values

x = torch.randn(1, 784)   # 1 sample
y = layer(x)              # shape: (1, 256)
```

<div class="mt-3 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-sm">

**Note**: Linear layers are the building blocks of most networks. Each neuron learns its own weights.

</div>

</div>

</div>

---
layout: default
---

# What are Activation Functions?

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### A Function Applied After Each Layer

Activation functions transform the output of a layer before passing it to the next:

```python
# Without activation (just linear)
x = linear1(x)
x = linear2(x)

# With activation
x = linear1(x)
x = relu(x)      # <-- Activation function
x = linear2(x)
```

<div class="mt-4 text-sm">

In PyTorch:

```python
self.network = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),          # Activation
    nn.Linear(128, 10)
)
```

</div>

</div>

<div>

### Why Do We Need Them?

Without activation functions, stacking layers just gives another linear function:

```python
# Two linear layers without activation
y = W2 @ (W1 @ x)
# Simplifies to:
y = W_combined @ x  # Still linear!
```

<div class="mt-4 p-3 bg-blue-50 dark:bg-blue-900/30 rounded text-sm">

**Key insight**: Linear functions can only learn linear patterns. Activation functions add **non-linearity**, allowing the network to learn complex patterns like curves and edges.

</div>

<div class="mt-3 text-sm opacity-80">

Think of it this way: linear = straight lines only. Activations = curves and complex shapes.

</div>

</div>

</div>

---
layout: default
---

# Common Activation Functions

<img src="/images/activation-functions.png" class="rounded-lg shadow-lg" />

<div class="grid grid-cols-4 gap-3 mt-3 text-xs">

<div class="p-2 border rounded">

**ReLU** - `max(0, x)`

Most common choice. Simple and fast to compute.

</div>

<div class="p-2 border rounded">

**Sigmoid** - Output: (0, 1)

Good for output layer when you need probabilities.

</div>

<div class="p-2 border rounded">

**Tanh** - Output: (-1, 1)

Zero-centered output. Sometimes used in RNNs.

</div>

<div class="p-2 border rounded">

**Leaky ReLU**

Like ReLU but allows small negative values through.

</div>

</div>

<div class="mt-2 p-2 bg-green-50 dark:bg-green-900/30 rounded text-sm text-center">

**Rule of thumb**: Start with ReLU for hidden layers. It works well in most cases.

</div>

---
layout: default
---

# Choosing an Activation Function

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### When to Use What

| Location | Recommended | Why |
|----------|-------------|-----|
| Hidden layers | **ReLU** | Fast, works well |
| Binary output | Sigmoid | Outputs 0-1 probability |
| Multi-class output | None* | Use CrossEntropyLoss |

<div class="mt-3 text-xs opacity-70">

*CrossEntropyLoss applies softmax internally, so don't add it yourself.

</div>

<div class="mt-4 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-sm">

**Simple rule**: Use ReLU between hidden layers, and let your loss function handle the output layer.

</div>

</div>

<div>

### Why Leaky ReLU Exists

ReLU has a problem: if a neuron outputs negative values, it becomes "dead" (always outputs 0).

```python
# ReLU: negative inputs -> 0
relu(-5)  # Returns 0
relu(-100)  # Returns 0 (neuron is "dead")

# Leaky ReLU: small slope for negatives
leaky_relu(-5)   # Returns -0.05
leaky_relu(-100) # Returns -1.0 (still learning!)
```

<div class="mt-3 text-sm">

**When to try Leaky ReLU:**
- If your model isn't learning well with ReLU
- Deep networks where neurons might "die"
- Usually not needed for simple networks

</div>

</div>

</div>

---
layout: default
---

# Regularization Layers

<div class="grid grid-cols-2 gap-6 mt-1">

<div>

### What is a Neuron?

A **neuron** is a single unit in a layer. It receives inputs, multiplies each by a learned weight, adds them up, and applies an activation.

<div class="text-xs p-2 bg-gray-100 dark:bg-gray-800 rounded font-mono">
output = activation(w1*x1 + w2*x2 + ... + bias)
</div>

<div class="text-sm mt-1">

In `nn.Linear(784, 128)`, there are **128 neurons**, each learning 784 weights.

</div>

### Dropout

Randomly "turns off" neurons during training (sets output to 0).

```python
dropout = nn.Dropout(p=0.5)  # 50% drop rate
model.train()  # Dropout ON
model.eval()   # Dropout OFF
```

</div>

<div>

### Why Drop Neurons?

Without dropout, neurons become **co-dependent** - they rely on specific other neurons. This causes **overfitting** (memorizing data instead of learning).

<div class="text-sm">

**Impact of dropout:**
- Forces each neuron to learn independently
- Creates redundancy (multiple neurons learn similar things)
- Like training many networks and averaging them

</div>

<div class="mt-1 p-2 bg-yellow-50 dark:bg-yellow-900/30 rounded text-xs">

**Note**: During eval, all neurons are active but scaled. Always call `model.eval()` before inference!

</div>

</div>

</div>

---
layout: default
---

# nn.Sequential: Quick Model Building

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### When to Use Sequential

`nn.Sequential` chains layers in order - data flows straight through A → B → C.

```python
# Simple way to build a network
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

output = model(input_tensor)
```

<div class="mt-2 text-sm">

**Use `nn.Sequential` when:**
- Layers flow linearly (no branching)
- Quick prototyping

</div>

</div>

<div>

### When You Need Custom nn.Module

Sometimes data doesn't just flow straight through. A **skip connection** lets data "skip over" some layers and get added back later:

```python
class BlockWithSkip(nn.Module):
    def forward(self, x):
        original = x          # Save the input
        out = self.layers(x)  # Process through layers
        return out + original # Add original back!
```

<div class="mt-2 text-sm">

**Why skip?** In very deep networks, gradients can become tiny. Skip connections give gradients a "shortcut" path, making training easier.

</div>

<div class="mt-2 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-xs">

**Note**: Skip connections are used in advanced architectures like ResNet. For now, `nn.Sequential` works great for simple networks.

</div>

</div>

</div>
