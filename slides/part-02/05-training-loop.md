---
layout: image-right
image: /images/05-training-loop.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# The Training Loop

Putting it all together

---
layout: default
---

# The Training Loop Overview

<div class="mt-2">

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()       # 1. Zero gradients
        predictions = model(batch_x)       # 2. Forward pass
        loss = criterion(predictions, batch_y)  # 3. Compute loss
        loss.backward()             # 4. Backward pass
        optimizer.step()            # 5. Update weights

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

</div>

<div class="grid grid-cols-5 gap-2 mt-3 text-xs">

<div class="p-2 border rounded text-center bg-gray-100 dark:bg-gray-800">

**zero_grad**

Clear accumulated gradients

</div>

<div class="p-2 border rounded text-center bg-blue-100 dark:bg-blue-900/30">

**forward**

Model makes predictions

</div>

<div class="p-2 border rounded text-center bg-orange-100 dark:bg-orange-900/30">

**loss**

Measure prediction error

</div>

<div class="p-2 border rounded text-center bg-red-100 dark:bg-red-900/30">

**backward**

Compute gradients

</div>

<div class="p-2 border rounded text-center bg-green-100 dark:bg-green-900/30">

**step**

Update weights

</div>

</div>

---
layout: default
---

# Datasets and DataLoaders

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### Built-in Datasets (torchvision)

PyTorch provides common datasets ready to use.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(
    './data', train=True, download=True,
    transform=transform)
test_data = datasets.MNIST(
    './data', train=False, transform=transform)
```

</div>

<div>

### DataLoader: Batching & Shuffling

```python
train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(
    test_data, batch_size=64, shuffle=False)

# Iterate over batches
for images, labels in train_loader:
    print(images.shape)  # (64, 1, 28, 28)
    print(labels.shape)  # (64,)
    break
```

<div class="mt-2 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-xs">

**DataLoader handles**: batching, shuffling, parallel loading with `num_workers`.

</div>

<div class="mt-2 text-xs opacity-70">

Other datasets: CIFAR-10, ImageNet, FashionMNIST in `torchvision.datasets`.

</div>

</div>

</div>

---
layout: default
---

# Training vs Evaluation Mode

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### Why Modes Matter

Some layers behave differently during training vs inference:

| Layer | Training | Evaluation |
|-------|----------|------------|
| `Dropout` | Randomly zeros neurons | Disabled |
| `BatchNorm` | Uses batch statistics | Uses running statistics |

<div class="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/30 rounded text-sm">

**Critical**: Forgetting `model.eval()` during testing will give wrong results!

</div>

</div>

<div>

### Setting the Mode

```python
# Training mode (default)
model.train()
for batch_x, batch_y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    loss.backward()
    optimizer.step()

# Evaluation mode
model.eval()
with torch.no_grad():  # Disable gradient computation
    for batch_x, batch_y in test_loader:
        predictions = model(batch_x)
```

<div class="mt-2 p-2 bg-blue-50 dark:bg-blue-900/30 rounded text-xs">

**`torch.no_grad()`**: Disables gradient tracking for faster inference and less memory usage. Always use during evaluation!

</div>

</div>

</div>

---
layout: default
---

# Complete MNIST Training Example

```python {all|1-3|5-8|10-12|14-20}{lines:false}
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. Create model, loss, optimizer
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
criterion, optimizer = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
for epoch in range(5):
    model.train(); total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad(); loss = criterion(model(images), labels)
        loss.backward(); optimizer.step(); total_loss += loss.item()
    print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(train_loader):.4f}")
```

<v-click at="4">

<div class="p-2 bg-green-50 dark:bg-green-900/30 rounded text-sm text-center">
This trains a digit classifier to ~97% accuracy in just a few epochs!
</div>

</v-click>

---
layout: default
---

# Tracking Training Progress

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### Recording Metrics

```python
train_losses, val_losses, val_accs = [], [], []

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        # ... training step ...
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            val_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum()
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(correct / len(val_dataset))
```

</div>

<div>

### Visualizing Progress

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train')
ax1.plot(val_losses, label='Validation')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.legend(); ax1.set_title('Loss Over Time')

ax2.plot(val_accs)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')

plt.tight_layout()
plt.show()
```

<div class="mt-2 text-xs opacity-70">

Watch for: loss decreasing, train/val gap (overfitting), accuracy improving.

</div>

</div>

</div>

---
layout: default
---

# GPU Training

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### Moving to GPU

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# Move model to GPU
model = model.to(device)

# Training loop with GPU
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    loss.backward()
    optimizer.step()
```

</div>

<div>

### Best Practices

<div class="space-y-2 text-sm">

<div class="p-2 bg-green-50 dark:bg-green-900/30 rounded">

**Do** define device once at the start and reuse it.

</div>

<div class="p-2 bg-green-50 dark:bg-green-900/30 rounded">

**Do** use `pin_memory=True` in DataLoader for faster GPU transfer.

</div>

<div class="p-2 bg-red-50 dark:bg-red-900/30 rounded">

**Don't** call `.to(device)` inside the loop more than needed.

</div>

</div>

<div class="mt-4">
Device-Agnostic Code
</div>
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
output = model(torch.randn(32, 10).to(device))
```

</div>

</div>
