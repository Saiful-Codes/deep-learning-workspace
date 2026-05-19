# Fast Learning Plan — Deep Learning Labs

> Goal: assignment-ready understanding, not academic mastery.
> Total budget: ~6–10 focused hours instead of the full 25–35h roadmap in [CLAUDE.md](CLAUDE.md).

---

## Priority tiers

| Tier | Weeks | Why |
| ---- | ----- | --- |
| **S — must own** | W3, W5, W7b | Core patterns: training loop, CNN, transfer learning. 90% of any DL assignment lives here. |
| **A — high leverage** | W6, W8 (text_dataset.py + lab06.ipynb) | Debugging skills + HuggingFace pipeline. Assignment may use either. |
| **B — skim** | W2, W4, W7a | W2 is reference-only; W4 is W3 reused; W7a is a transforms catalogue. |
| **C — optional** | W8 demo notebook | Just embedding visualisation; skip unless time. |

---

## Recommended notebook order (optimal for assignment prep)

1. **Week 2 — Tensors** — 20 min skim. Treat as reference. Just confirm you can `reshape`, `permute`, `unsqueeze`, slice, and move tensors to `cuda`.
2. **Week 3 — `Lab 2 demo with solutions.ipynb` + `lab02_exc.ipynb`** — 60 min. **Memorise the 8-step training loop.** This is the spine of every later lab.
3. **Week 4 — `lab 2b.ipynb`** — 20 min. Just note the 4 changes to convert regression → classification (output size, label dtype, CrossEntropyLoss, accuracy metric). Don't re-do the W3 work.
4. **Week 5 — `lab03_solutions.ipynb`** — 90 min. **High-value.** Walk through every CNN variant: ConvNet → DeepConvNet → DropoutConvNet → BNConvNet → SkipConvNet. Skip the eager-loading image demo (`Lab 3 demo (how to load images...)`) — lazy loading is what you'll actually use.
5. **Week 6 — `lab04 solution.ipynb`** — 60 min. **High-value.** This is your debugging checklist. Pay attention to: z-score normalisation, LR diagnosis, confusion matrix, WeightedRandomSampler, weighted CE loss.
6. **Week 7a — `lab05a solution.ipynb`** — 15 min skim. Just learn the **train vs test pipeline pattern** and which augmentations exist. You don't need to memorise every transform — you'll look it up.
7. **Week 7b — `lab05b solution.ipynb`** — 60 min. **High-value.** Transfer learning with resnet18 is the realistic recipe for image assignments. Focus on: ImageFolder, head replacement, freeze + unfreeze flow.
8. **Week 8 — modular `.py` files + `lab06.ipynb`** — 60 min. **High-value for NLP.** Read the `.py` files first (they're the implementation); the notebook just wires them together. Skip the demo notebook unless you have time.

---

## Per-notebook learning targets

### Week 2 (skim, 20 min)
- Shape conventions: image `(C,H,W)`, batch `(B,C,H,W)`, tabular `(B,F)`
- `reshape` vs `permute` (memory layout vs semantic swap)
- `torch.cat` vs `torch.stack` (one less dim vs one more dim)
- `unsqueeze` / `squeeze` for adding/removing batch dim
- **Skip**: the Python intro section

### Week 3 (60 min) — THE FOUNDATION
- The 8-step training loop (zero_grad → forward → loss → backward → step)
- `Dataset` skeleton (`__init__`, `__len__`, `__getitem__`)
- `DataLoader` + `random_split`
- Why normalisation matters (loss-explosion bug shown in cell)
- MSE for regression
- Adam vs SGD — Adam @ lr=1e-3 is the strong default
- Modular `train_step` / `test_step` / `train()` / `test()` pattern
- **Skip**: the redundant "Simple Training loop" cell (it's the non-modular version of the same thing)

### Week 4 (20 min) — DELTA ONLY
- Output dim = `num_classes` (not 1)
- Labels: `.squeeze().long()` and 1D
- Loss: `nn.CrossEntropyLoss()`
- Metric: `torchmetrics.Accuracy(task='multiclass', num_classes=N)`
- **Skip**: the data-loading cells — they're copy-paste from W3

### Week 5 (90 min) — CNN PROGRESSION
- Conv2d shape math: `out_size = (in - k + 2p)/s + 1`
- Common kernel choice: `3x3, padding=1` keeps size
- `MaxPool2d(2)` halves spatial size
- The model ladder (each adds one technique):
  1. ConvNet — bare convs
  2. DeepConvNet — more layers, smaller kernels
  3. DropoutConvNet — regularisation
  4. BNConvNet — BatchNorm after each conv (placement matters: Conv → BN → ReLU)
  5. SkipConvNet — ResNet-style residual blocks
- `model.train()` vs `model.eval()` (Dropout/BN behaviour)
- `SummaryWriter` (TensorBoard) usage
- **Skip**: the eager-loading image dataset demo (just understand the lazy one)

### Week 6 (60 min) — DEBUGGING CHECKLIST
- z-score: `(x - mean) / std` — compute on train only
- LR diagnosis: oscillating loss → too high; flat loss → too low
- Confusion matrix — *always* check before trusting accuracy
- Two fixes for class imbalance:
  1. `WeightedRandomSampler` in DataLoader
  2. `nn.CrossEntropyLoss(weight=class_weights)`
- **Skip**: nothing — every section earns its place here

### Week 7a (15 min) — REFERENCE
- Train pipeline: heavy aug (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `Normalize`)
- Test pipeline: deterministic only (`Resize`, `CenterCrop`, `Normalize`)
- `transforms.Compose([...])`
- **Skip**: individual transform examples — just remember names exist

### Week 7b (60 min) — TRANSFER LEARNING
- `ImageFolder` for "one folder per class" datasets
- `torchvision.models.resnet18(weights=...)` (newer API) / `pretrained=True` (older)
- Replace head: `model.fc = nn.Linear(model.fc.in_features, num_classes)`
- Freeze: `for p in model.parameters(): p.requires_grad = False` BEFORE swapping `fc`
- Fine-tune phase: unfreeze and use small LR (1e-4 or 1e-5)
- **Skip**: nothing critical

### Week 8 (60 min) — NLP
- Read `.py` files in this order:
  1. `text_dataset.py` — tokenizer pattern (`AutoTokenizer`, truncation+padding)
  2. `lstm.py` — `Embedding → LSTM → fc` (read but don't deeply study)
  3. `distbert.py` — wrap `AutoModelForSequenceClassification`; return `.logits`
  4. `trainer.py` — same train/test loop you already know
  5. `utils.py` — `state_dict()` save/load + device
- Notebook just wires them; understand the integration pattern
- **Skip**: Google Drive mount cells (Colab-specific), wandb (use TensorBoard if anything)

---

## Assignment-critical sections (the 20% that handles 80%)

If you only have time for these 9 things, learn them in this order:

1. The 8-step training loop (W3)
2. Custom `Dataset` class (W3, W5, W8 all use the same skeleton)
3. CNN model class with `Conv → BN → ReLU → Pool` (W5)
4. `model.train()` / `model.eval()` + `with torch.no_grad():` (W3, W6)
5. Confusion matrix + class-imbalance fix (W6)
6. `transforms.Compose` for image pipelines (W7a)
7. Transfer learning head swap (W7b)
8. HuggingFace `AutoTokenizer` + `AutoModelForSequenceClassification` (W8)
9. `torch.save(model.state_dict(), path)` + load (W8 utils.py)

---

## Hard skips (low value for assignment)

- Wine-quality regression *experiments* in W3 (knowing it works is enough)
- The Python intro at the top of W2
- The eager-image-loading demo in W5
- Individual transform demonstrations in W7a (just know they exist)
- t-SNE embedding visualisation in W8 demo notebook
- `wandb` setup — TensorBoard is sufficient
- Any cell labelled "Challenge" unless time allows

---

## What I'll do as we go

* Teach one week at a time in chat
* Show code with inline annotations (not screenshots of cells)
* Call out **assignment hooks** (places you'll re-use this pattern verbatim)
* Update `progress.md` after each week
* Flag any broken code / dependency issues / missing files immediately
