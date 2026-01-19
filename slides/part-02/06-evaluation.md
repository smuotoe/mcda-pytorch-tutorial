---
layout: image-right
image: /images/06-evaluation.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# Model Evaluation

Testing, metrics, and saving models

---
layout: default
---

# Evaluating Your Model

<div class="grid grid-cols-2 gap-6 mt-1">

<div>

### Train / Validation / Test Split

<div class="text-sm">

| Set | Purpose | When to Use |
|-----|---------|-------------|
| **Train** | Learn patterns | During training |
| **Validation** | Tune hyperparams | After each epoch |
| **Test** | Final evaluation | Once, at the end |

</div>

<div class="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/30 rounded text-xs">

**Golden rule**: Never use test data during development!

</div>

<div class="mt-4"></div>

### Detecting Overfitting

<div class="text-sm">

- Train loss down, val loss up = overfitting
- Large train/val accuracy gap

</div>

</div>

<div>

### Evaluation Loop

```python
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total
```

</div>

</div>

---
layout: default
---

# Common Evaluation Metrics

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix)

# Get predictions
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds = model(x.to(device)).argmax(1).cpu()
        all_preds.extend(preds)
        all_labels.extend(y)

# Calculate metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='weighted')
rec = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
```

</div>

<div>

### What Each Metric Tells You

<div class="text-sm">

| Metric | Question Answered |
|--------|-------------------|
| **Accuracy** | What % of predictions are correct? |
| **Precision** | Of predicted positives, how many correct? |
| **Recall** | Of actual positives, how many found? |
| **F1 Score** | Harmonic mean of precision & recall |

</div>

### Confusion Matrix

```python
import seaborn as sns
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual')
```

</div>

</div>

---
layout: default
---

# Saving and Loading Models

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### Saving Models

```python
# Save state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Save checkpoint for resuming training
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

<div class="mt-2 p-2 bg-green-50 dark:bg-green-900/30 rounded text-xs">

**Best practice**: Save `state_dict()` - portable and doesn't depend on code structure.

</div>

</div>

<div>

### Loading Models

```python
# Load state dict
model = MyModel()  # Create model first
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode

# Load checkpoint for resuming
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load with device mapping (GPU -> CPU)
model.load_state_dict(
    torch.load('weights.pth', map_location=device))
```

</div>

</div>
