---
layout: image-right
image: /images/07-summary.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# Summary & Exercises

Wrapping up Part 2

---
layout: default
---

# Part 2 Recap

<div class="grid grid-cols-2 gap-8 mt-4">

<div>

### What We Learned

<div class="space-y-3">

<div class="flex items-start gap-3">
<div class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">1</div>
<div><strong>nn.Module</strong> - Base class for neural networks. Define layers in `__init__`, computation in `forward`.</div>
</div>

<div class="flex items-start gap-3">
<div class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">2</div>
<div><strong>Loss Functions</strong> - MSELoss for regression, CrossEntropyLoss for classification.</div>
</div>

<div class="flex items-start gap-3">
<div class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">3</div>
<div><strong>Optimizers</strong> - Adam is a great default. Learning rate is the key hyperparameter.</div>
</div>

<div class="flex items-start gap-3">
<div class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">4</div>
<div><strong>Training Loop</strong> - zero_grad, forward, loss, backward, step. Repeat!</div>
</div>

<div class="flex items-start gap-3">
<div class="w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">5</div>
<div><strong>Evaluation</strong> - Use model.eval() and torch.no_grad() for inference.</div>
</div>

</div>

</div>

<div>

### The Complete Picture

```python
# Define
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train
model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

# Evaluate
model.eval()
with torch.no_grad():
    predictions = model(test_x.to(device))

# Save
torch.save(model.state_dict(), 'model.pth')
```

</div>

</div>

---
layout: default
class: text-sm
---

# Part 2 Exercises

Build a banknote forgery detector using the Banknote Authentication dataset (1372 samples, 4 features, binary classification).

<div class="grid grid-cols-2 gap-6 mt-2">

<div>

### Exercise 1: Build a Binary Classifier

```python
# TODO: Create a model with:
# - Input: 4 features (variance, skewness, etc.)
# - Hidden: 32 neurons with ReLU
# - Output: 1 neuron (binary classification)
```

<div class="mt-4"></div>

### Exercise 2: Training Loop

- Use `BCEWithLogitsLoss` for binary classification
- Train for 100 epochs with Adam optimizer
- Track and plot train/validation loss

</div>

<div>

### Exercise 3: Hyperparameter Tuning

Compare different configurations:
- Learning rates: 0.01, 0.001, 0.0001
- Hidden sizes: 16, 32, 64

<div class="mt-4"></div>

### Exercise 4: Evaluation

- Calculate accuracy, precision, recall, F1
- Plot confusion matrix
- Save your best model

</div>

</div>

<div class="absolute bottom-4 left-12 right-12 flex justify-between items-center text-xs">

<span class="opacity-70">Exercises: `exercises/pytorch-2-exercises.ipynb`</span>

<span class="px-2 py-1 bg-yellow-100 dark:bg-yellow-900/30 rounded font-medium">Bonus: Train an MNIST classifier with >95% accuracy!</span>

</div>

---
layout: image-right
image: /images/07-closing.jpg
backgroundSize: cover
class: flex flex-col justify-center
---

# You Made It!

<div class="text-xl font-semibold text-blue-500 mt-4">
You now have the foundation for deep learning!
</div>

<div class="mt-6 space-y-3 text-sm">

<div class="flex items-center gap-3">
<div class="w-2 h-2 rounded-full bg-green-500"></div>
<span><strong>Build</strong> neural networks with nn.Module</span>
</div>

<div class="flex items-center gap-3">
<div class="w-2 h-2 rounded-full bg-blue-500"></div>
<span><strong>Train</strong> with loss functions and optimizers</span>
</div>

<div class="flex items-center gap-3">
<div class="w-2 h-2 rounded-full bg-purple-500"></div>
<span><strong>Evaluate</strong> and save your models</span>
</div>

</div>

<div class="mt-8 text-base">

Questions? Bring them to the session!

</div>

<div class="abs-bl m-6 text-sm opacity-50">
MCDA 5511
</div>
