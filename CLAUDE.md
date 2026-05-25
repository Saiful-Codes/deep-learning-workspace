# CLAUDE.md — Deep Learning (CSE3001) Lab Companion

> Personal learning roadmap built from a deep scan of every notebook and supporting
> file inside `Labs/` (Week 2 → Week 8). Use this as the single source of truth
> for what each week covers, the prerequisites you must hold in your head, and
> the order to study things before your assignment.

---

## 1. Overall summary of the course

This subject teaches **Deep Learning with PyTorch** through a progression that
mirrors how real DL practitioners build models:

```
Tensors → Datasets/DataLoaders → MLPs (regression + classification)
       → CNNs (image classification) → Debugging & normalisation
       → Data augmentation + Transfer learning
       → NLP with LSTM + Transformers (DistilBERT)
```

By the end of Week 8 you should be able to (independently):

* Load any tabular **or** image **or** text dataset into a `torch.utils.data.Dataset`.
* Build an MLP, a CNN and a Transformer-based classifier from scratch.
* Write a complete train/validate/test loop with the right loss + optimiser.
* Diagnose typical failure modes: wrong learning rate, class imbalance,
  unnormalised inputs, overfitting.
* Apply transfer learning with `torchvision` pretrained models.
* Use HuggingFace `transformers` (tokenizer + `AutoModelForSequenceClassification`).
* Track experiments with TensorBoard / Weights & Biases.

---

## 2. Major topics by week

### Week 2 — PyTorch & Tensor Fundamentals
Files: [Lab_1_demo.ipynb](Labs/Week2_lab/Lab_1_demo.ipynb), [lab01_exc_solution.ipynb](Labs/Week2_lab/lab01_exc_solution.ipynb)

* Python refresher (lists, tuples, functions, control flow)
* PyTorch tensors: creation (`tensor`, `rand`, `zeros`, `ones`, `eye`, `arange`)
* Shapes, dtypes, casting (`.int()`, `.float()`, `.bool()`)
* Vectorisation vs Python `for` loops (huge speed difference)
* **Reshape vs Permute** — reshape re-interprets memory; permute moves data
* `squeeze` / `unsqueeze` (removing/adding size-1 dimensions)
* Slicing, negative indexing, advanced multi-dim indexing
* Concatenation (`torch.cat`, `torch.stack`)
* Image tensor convention: `(C, H, W)` per image, `(B, C, H, W)` for batch

### Week 3 — First Neural Networks (MLP, Regression)
Files: [Lab 2 demo with solutions.ipynb](Labs/Week3_lab/Lab%202%20demo%20with%20solutions.ipynb), [lab02_exc.ipynb](Labs/Week3_lab/lab02_exc.ipynb), datasets `winequality-{red,white}.csv`

* Custom `Dataset` class (lazy vs eager loading)
* `DataLoader` (batching, shuffling, multiprocessing)
* Linear function learning task → "can a NN learn y = 2x?"
* `nn.Module` subclassing, `nn.Linear`, `nn.ReLU`, `nn.Sequential`
* **Standard training loop pattern** (memorise this!):
  1. zero grads → 2. forward → 3. loss → 4. backward → 5. optimizer.step
* Why normalisation matters (mean 0, std 1)
* MSE loss for regression; Adam vs SGD optimisers
* Wine-quality regression problem with `pandas`, `train_test_split`

### Week 4 — Classification & Modular Training Code
File: [lab 2b.ipynb](Labs/Week4_lab/lab%202b.ipynb)

* Convert a **regression problem into multi-class classification**
* Output size = `num_classes`; labels must be `long` and 1D
* `nn.CrossEntropyLoss` (combines softmax + NLL)
* `torchmetrics` library: `Accuracy(task='multiclass', num_classes=N)`
* `metric.reset()` between epochs/modes
* Writing your own custom metric by subclassing `torchmetrics.Metric`
  (`add_state`, `update`, `compute`)
* Modular `train_step` / `test_step` / `train()` / `test()` functions

### Week 5 — Convolutional Neural Networks (Images)
Files: [Lab 3 demo (how to load images...).ipynb](Labs/Week5_lab/Lab%203%20demo%20(how%20to%20load%20images%20in%20dataset)%20solutions.ipynb), [lab03_solutions.ipynb](Labs/Week5_lab/lab03_solutions.ipynb), `lab 3 model training results.xlsx`

* Writing an **image `Dataset`**: eager vs lazy loading
* Reading images with `PIL.Image`, converting to tensors, resizing
* MNIST-like classification task
* `nn.Conv2d`, kernel size, padding, feature maps
* Progression of CNN architectures explored in lab:
  * Basic ConvNet (1 → 4 → 8 → 16 → 20 channels)
  * `DeepConvNet` (extra layers, smaller kernels)
  * `DropoutMLP` / `DropoutConvNet` (`nn.Dropout`)
  * `BNConvNet` (`nn.BatchNorm2d` after each conv)
  * **Skip connections** (`SkipBlock`, `SkipConvNet`) — ResNet-style
  * `CustomModel` (your turn)
* **TensorBoard** (`SummaryWriter`) for logging loss/acc
* Visualising correct/incorrect predictions

### Week 6 — Debugging Neural Networks
File: [lab04 solution.ipynb](Labs/Week6_lab/lab04%20solution.ipynb)

* **Data normalisation** (z-score) on tabular `forest-cover` dataset
  * Compute mean/std on the train split only, then apply to val/test
* **Learning-rate diagnosis** — symptoms of LR too high (oscillation) vs too low
  (slow descent vs flat curves)
* **Class imbalance** detection & fixes:
  * Confusion matrix (raw + normalised) via `torchmetrics.ConfusionMatrix`
  * `WeightedRandomSampler` in the `DataLoader`
  * Weighted loss function (`weight=` in `CrossEntropyLoss`)
* Per-class metrics: `torchmetrics.Recall` / `Precision`

### Week 7 — Image Augmentation & Transfer Learning
Files: [lab05a solution.ipynb](Labs/Week7_lab/lab05a%20solution.ipynb), [lab05b solution.ipynb](Labs/Week7_lab/lab05b%20solution.ipynb)

* `torchvision.transforms` catalogue:
  * **Deterministic**: `Resize`, `CenterCrop`, `FiveCrop`, `TenCrop`,
    `Pad`, `Grayscale`, `ToTensor`, `ToPILImage`, `Normalize`
  * **Random**: `RandomCrop`, `RandomResizedCrop`, `RandomHorizontalFlip`,
    `RandomVerticalFlip`, `RandomRotation`, `RandomPerspective`,
    `RandomAffine`, `ColorJitter`, `RandomErasing`, `RandomGrayscale`
  * **Utility**: `Compose`, `Lambda`, `RandomApply`, `RandomChoice`, `RandomOrder`
* Different transform pipelines for **train (heavy aug) vs test (just resize/normalise)**
* `torchvision.datasets.ImageFolder` for "one folder per class" image datasets
* **Transfer learning** with `torchvision.models.resnet18(pretrained=True)`:
  * Replace the final `fc` layer with one matching your `num_classes`
  * Freeze backbone (`param.requires_grad = False`) → train only the head
  * Then **fine-tune** the entire model with a smaller learning rate

### Week 8 — NLP with LSTM & Transformers
Files: [Lab 6 demo (NLP Transformers).ipynb](Labs/Week8_lab/Lab%206%20demo%20(NLP%20Transformers).ipynb), [lab06.ipynb](Labs/Week8_lab/lab06.ipynb), plus modular `.py` files

* Project structure (modular Python):
  * [text_dataset.py](Labs/Week8_lab/text_dataset.py) — `TextDataset` (tokenize with `AutoTokenizer`)
  * [lstm.py](Labs/Week8_lab/lstm.py) — `nn.Embedding` → `nn.LSTM` → classification head
  * [distbert.py](Labs/Week8_lab/distbert.py) — `AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')`
  * [trainer.py](Labs/Week8_lab/trainer.py) — `train_epoch` / `test_epoch` / `train_model`
  * [utils.py](Labs/Week8_lab/utils.py) — device, `save_model_state`, `load_model_state`
* Tokenisation with HuggingFace `AutoTokenizer`
  (truncation, padding, `max_length`, `return_tensors='pt'`)
* Comparing **LSTM vs Transformer** on sentiment analysis (IMDB-style CSV)
* Visualising transformer embeddings (t-SNE) and attention
* Saving / loading model weights via `state_dict()`
* Experiment tracking with **Weights & Biases (`wandb`)**
* Dependencies: `transformers==4.51.1`, `torchmetrics==0.11.3`, `wandb==0.19.9`

---

## 3. Skills you will own by the end

| Skill area               | Concrete things you'll be able to do |
| ------------------------ | ------------------------------------ |
| Tensors                  | Reshape, permute, slice, batch, move to GPU |
| Data pipelines           | Custom Dataset, DataLoader, ImageFolder, TextDataset |
| Model authoring          | MLP, CNN, ResNet-style skips, LSTM, Transformer wrappers |
| Training loop            | Modular `train_step` / `test_step`, epochs, metrics, logging |
| Evaluation               | Accuracy, RMSE, R², Confusion matrix, Recall / Precision |
| Debugging                | LR diagnosis, normalisation, class imbalance fixes |
| Augmentation             | `torchvision.transforms` Compose pipelines |
| Transfer learning        | Freeze + replace head; fine-tune |
| NLP                      | HuggingFace tokenizer + `AutoModelForSequenceClassification` |
| Reproducibility          | `torch.manual_seed`, `seed_all` helper |
| Experiment tracking      | TensorBoard, Weights & Biases |
| Persistence              | `torch.save(model.state_dict(), path)` / `load_state_dict` |

---

## 4. Prerequisite knowledge required

You need **basic Python** (functions, classes, loops, list/tuple/dict),
**high-school linear algebra** (vectors, matrices, matrix multiplication
intuition), and **basic calculus intuition** (what a gradient means — you do
not need to derive backprop). Familiarity with `numpy` and `pandas` is helpful
but each is introduced gently. Everything else (PyTorch, transformers,
torchvision) is taught inside the labs.

---

## 5. Difficulty analysis

| Week | Topic                    | Difficulty | Estimated time |
| ---- | ------------------------ | ---------- | -------------- |
| 2    | Tensors                  | ★☆☆☆☆     | 2–3 h         |
| 3    | First MLP (regression)   | ★★☆☆☆     | 3–4 h         |
| 4    | Classification + metrics | ★★☆☆☆     | 2–3 h         |
| 5    | CNNs                     | ★★★☆☆     | 5–6 h         |
| 6    | Debugging                | ★★★☆☆     | 3–4 h         |
| 7a   | Augmentation             | ★★☆☆☆     | 2 h           |
| 7b   | Transfer learning        | ★★★☆☆     | 3–4 h         |
| 8    | NLP / Transformers       | ★★★★☆     | 5–6 h         |

**Total focused study budget: ~25–35 hours** to be able to handle the assignment confidently.

---

## 6. Recommended learning order (fast track)

If your assignment deadline is tight, walk through the labs in this exact order
— each one builds tools used by the next:

1. **Week 2** — Tensors. Don't skip; everything else assumes fluency here.
2. **Week 3** — MLP regression + the canonical training loop. Memorise the
   8-step loop.
3. **Week 4** — Same model but classification. Locks in `CrossEntropyLoss`,
   `torchmetrics`, modular `train_step`/`test_step`.
4. **Week 5** — CNNs. The big jump. Pay attention to the `Conv2d` shape math
   and the model progression: Conv → Deep → Dropout → BatchNorm → Skip.
5. **Week 6** — Debugging. Treat this as a checklist you'll re-apply for *every*
   future model: "Is the data normalised? Is the LR sensible? Is there class
   imbalance? What does the confusion matrix say?"
6. **Week 7a** — Augmentation. Quick.
7. **Week 7b** — Transfer learning. This is high-leverage; resnet18 + custom
   head is the realistic recipe for any small-image-dataset assignment.
8. **Week 8** — NLP. Read the supporting `.py` files first
   (`text_dataset.py`, `lstm.py`, `distbert.py`, `trainer.py`, `utils.py`),
   then run the notebook — they are the actual implementation.

---

## 7. Key coding patterns repeated across the labs

### 7.1 Canonical training step (memorise!)
```python
model.train()
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

### 7.2 Canonical test/eval step
```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### 7.3 Device selection (used in *every* lab)
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### 7.4 Reproducible seeding
```python
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
```

### 7.5 Custom `Dataset` skeleton
```python
class MyDataset(Dataset):
    def __init__(self, ...): ...
    def __len__(self):  return self.n
    def __getitem__(self, idx): return input, target
```

### 7.6 Sequential model definition
```python
self.seq = nn.Sequential(
    nn.Linear(in_features, hidden),
    nn.ReLU(),
    nn.Linear(hidden, num_classes),
)
```

### 7.7 Transfer learning head-swap
```python
model = torchvision.models.resnet18(pretrained=True)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### 7.8 HuggingFace classifier
```python
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokens = tokenizer(texts, max_length=128, truncation=True,
                   padding=True, return_tensors='pt')
logits = model(tokens['input_ids']).logits
```

### 7.9 Save / load
```python
torch.save(model.state_dict(), path)
model.load_state_dict(torch.load(path))
```

---

## 8. Common mistakes & pitfalls to avoid

* **Forgetting `optimizer.zero_grad()`** — gradients accumulate across steps, ruining training.
* **Forgetting `model.train()` / `model.eval()`** — Dropout and BatchNorm behave differently in eval. Always toggle.
* **Forgetting `.to(device)`** for *both* model and data — mismatched-device errors are the #1 PyTorch crash.
* **Wrong label dtype for classification** — `CrossEntropyLoss` expects `long` 1D labels of shape `[batch]`, not `[batch, 1]`. Use `labels.squeeze().long()`.
* **Wrong output size** — regression: 1 output; multi-class: `num_classes`; binary with `BCEWithLogitsLoss`: 1 output.
* **Confusing `reshape` and `permute`** — `reshape` keeps memory order, `permute` swaps semantics. Image channel reordering needs `permute`.
* **Normalising the test set with test statistics** — always compute mean/std on **train**, then apply that same mean/std to val/test.
* **Calling `nn.Linear(...)` inside `forward`** — instantiates new (untrained) weights every call. All layers must live in `__init__`.
* **Not resetting `torchmetrics` between epochs/modes** — state carries over.
* **Learning rate too high** with SGD on un-normalised data → loss explodes (you'll see the 264-trillion loss in lab02a — that's exactly this bug).
* **Class imbalance** producing fake-high accuracy — always look at a confusion matrix before celebrating.
* **`tqdm` import errors** — the labs use `from tqdm.notebook import tqdm`; you may need `pip install tqdm`.
* **Tokenizer length mismatch** — pass `truncation=True`, `padding=True`, and an explicit `max_length` so all sequences end up the same shape.
* **Treating `state_dict` files as portable models** — you must instantiate the *same* model class before calling `load_state_dict`.

---

## 9. Library / framework inventory

Used at least once across the labs:

* **PyTorch core**: `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`
* **Vision**: `torchvision` (`transforms`, `datasets.ImageFolder`, `models.resnet18`)
* **NLP**: `transformers` (`AutoTokenizer`, `AutoModelForSequenceClassification`, `DistilBertModel`)
* **Metrics**: `torchmetrics` (`Accuracy`, `Recall`, `MeanSquaredError`, `R2Score`, `ConfusionMatrix`, `MeanMetric`)
* **Tabular**: `pandas`, `numpy`, `sklearn.metrics` (small uses)
* **Imaging**: `PIL.Image`, `matplotlib.pyplot`
* **UX / logging**: `tqdm`, `torch.utils.tensorboard.SummaryWriter`, `wandb`
* **Dim-reduction (Week 8 demo)**: `sklearn.manifold.TSNE`

---

## 10. Datasets touched

| Week | Dataset | Type | Task |
| ---- | ------- | ---- | ---- |
| 3/4  | Wine quality (white, red) | Tabular | Regression then classification |
| 5    | MNIST-style digits | Image (greyscale) | Multi-class classification |
| 6    | Forest cover + toy squares/rectangles | Tabular + synthetic image | Multi-class & binary |
| 7    | African Wildlife (4 classes) | Image (RGB) | Classification + transfer learning |
| 8    | Text sentiment CSV | Text | Binary sentiment (LSTM vs DistilBERT) |

---

## 11. Final preparation checklist for the assignment

Before opening the assignment, make sure you can do **all of these** without
Googling:

- [ ] Build a `Dataset` subclass with `__init__` / `__len__` / `__getitem__`.
- [ ] Wrap it in a `DataLoader` with batching + shuffling.
- [ ] Write an MLP and a CNN by subclassing `nn.Module`.
- [ ] Write the 8-line training step from memory.
- [ ] Pick the correct loss (`MSELoss`, `CrossEntropyLoss`, `BCEWithLogitsLoss`).
- [ ] Pick the correct optimiser & sensible learning rate (Adam @ 1e-3 is a strong default).
- [ ] Compute Accuracy / Confusion matrix with `torchmetrics`.
- [ ] Apply normalisation (tabular: z-score; image: `transforms.Normalize`).
- [ ] Apply image augmentation only on train.
- [ ] Replace the final layer of a `torchvision` pretrained model.
- [ ] Save / load `state_dict()`.
- [ ] Tokenize text with `AutoTokenizer` and feed to `AutoModelForSequenceClassification`.

When every box above can be ticked confidently, **you're assignment-ready**.

---

*This file was produced by analysing every notebook + supporting `.py` file inside `Labs/`. If new lab content is added, regenerate this file rather than editing manually so the structure stays consistent.*
