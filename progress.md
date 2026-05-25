# Learning Progress Tracker

> Updated continuously as we move through the labs.
> Status legend:  `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped intentionally

---

## Week-by-week status

| Week | Notebook(s) | Status | Time spent | Notes |
| ---- | ----------- | ------ | ---------- | ----- |
| 2    | Lab_1_demo + lab01_exc_solution | `[x]` | ~20 min | Skim done — covered shape conventions, reshape/permute, unsqueeze, cat/stack, device |
| 3    | Lab 2 demo with solutions + lab02_exc | `[x]` | ~60 min | Dataset, DataLoader, MLP, 8-step training loop, eval pattern, metrics, loss curves |
| 4    | lab 2b | `[x]` | ~20 min | 5 deltas: output dim, CrossEntropyLoss internals, label squeeze+long, torchmetrics Accuracy, argmax |
| 5    | lab03_solutions (CNN) | `[x]` | ~90 min | All 6 models trained on real MNIST via [_run_week5.py](Labs/Week5_lab/_run_week5.py). BN +5.5%, Skip+BN reached 98.2% in 1 epoch. |
| 5    | Lab 3 demo image loading | `[-]` | — | Skipped — lazy version inside lab03_solutions is enough |
| 6    | lab04 solution | `[x]` | ~60 min | Verified: normalisation +7.7pp on Forest Cover, WeightedRandomSampler doubled square recall in [_run_week6.py](Labs/Week6_lab/_run_week6.py) |
| 7a   | lab05a solution (augmentation) | `[x]` | ~20 min | Train pipeline = heavy aug + Normalize; test = Resize/CenterCrop/Normalize |
| 7b   | lab05b solution (transfer learning) | `[x]` | ~55 min | Verified: ConvNet 57%, ResNet18 frozen 92.2%, fine-tune 91.4%. See [_run_week7.py](Labs/Week7_lab/_run_week7.py) |
| 8    | lab06 + .py files | `[x]` | ~60 min | Verified: LSTM 71% (≈base rate), DistilBERT fine-tuned 92.2%, save/load/predict_sentiment all working via [_run_week8.py](Labs/Week8_lab/_run_week8.py). Transformer.pth saved. |
| 8    | Lab 6 demo NLP Transformers | `[-]` | — | Skipped — visualisation only |

---

## Concepts owned (mark as we go)

### Tensors & PyTorch basics (W2)
- [x] Shape conventions for image / batch / tabular
- [x] reshape vs permute
- [x] unsqueeze / squeeze
- [x] cat vs stack
- [x] device handling (`.to(device)`)

### MLP & training loop (W3)
- [x] Custom `Dataset` skeleton
- [x] `DataLoader` + `random_split`
- [x] 8-step training loop (memorised)
- [x] MSE loss & Adam optimiser
- [x] Train/val/test split convention
- [x] Modular train_step / test_step

### Classification (W4)
- [x] CrossEntropyLoss + label dtype gotcha
- [x] torchmetrics Accuracy
- [x] Multi-class output size = num_classes

### CNN (W5)
- [x] Conv2d math (output size formula)
- [x] Conv → BN → ReLU → Pool block
- [x] Dropout placement
- [x] BatchNorm placement
- [x] Residual / skip blocks
- [x] model.train() vs model.eval()
- [x] TensorBoard SummaryWriter

### Debugging (W6)
- [x] z-score normalisation (compute on train only)
- [x] LR diagnosis from loss curve
- [x] Confusion matrix
- [x] WeightedRandomSampler
- [x] Weighted CrossEntropyLoss

### Augmentation (W7a)
- [x] Train vs test transform pipelines
- [x] transforms.Compose

### Transfer learning (W7b)
- [x] ImageFolder dataset
- [x] resnet18 head replacement
- [x] freeze + unfreeze workflow

### NLP (W8)
- [x] AutoTokenizer (truncation + padding)
- [x] AutoModelForSequenceClassification
- [x] Reading `.logits`
- [x] state_dict save / load
- [x] LSTM model structure (Embedding → LSTM → fc)

---

## Pending topics (queued for teaching)

**ALL WEEKS COMPLETE.** Ready for the assignment.

---

## Blockers / issues / fixes applied

| Date | Issue | Fix / workaround | Status |
| ---- | ----- | ---------------- | ------ |
| —    | (Pre-flight) `tqdm` may be missing in fresh envs | `pip install tqdm torchmetrics torchvision transformers wandb` | Note |
| —    | Some notebooks use `pretrained=True` (deprecated) | Replace with `weights="DEFAULT"` if running on torchvision ≥ 0.13 | Note |
| —    | W8 `lab06.ipynb` mounts Google Drive (Colab-only) | Skip those cells when running locally — the dataset CSVs already exist at `Labs/Week8_lab/dataset/` | Note |
| —    | W3 wine dataset URL load (`pd.read_csv(white_wine_url, delimiter=';')`) needs internet | Local files `winequality-red.csv` / `winequality-white.csv` already in `Labs/Week3_lab/` — read locally | Note |

---

## Assignment readiness self-check

Tick when you can do each from memory, no notebooks open:

- [ ] Write a `Dataset` subclass from scratch
- [ ] Write the 8-step training loop from scratch
- [ ] Build an MLP and a CNN as `nn.Module` subclasses
- [ ] Pick correct loss + optimiser for a given task
- [ ] Compute accuracy + confusion matrix
- [ ] Apply train-only mean/std normalisation
- [ ] Build train-vs-test transform pipelines
- [ ] Swap a resnet18 head and freeze the backbone
- [ ] Tokenise text and feed to a HuggingFace classifier
- [ ] Save + load model weights

**Readiness: 10 / 10 (concept coverage)** — confirm by running the four `_run_weekX.py` scripts end-to-end before starting the assignment.

## Verified end-to-end on this machine

| Script | Status | Highlights |
| --- | --- | --- |
| [Labs/Week5_lab/_run_week5.py](Labs/Week5_lab/_run_week5.py) | ✓ | All 6 CNN variants on MNIST; SkipConvNet 98.2% in 1 epoch |
| [Labs/Week6_lab/_run_week6.py](Labs/Week6_lab/_run_week6.py) | ✓ | Forest Cover: +7.7pp from z-score; class imbalance fixes verified |
| [Labs/Week7_lab/_run_week7.py](Labs/Week7_lab/_run_week7.py) | ✓ | African Wildlife: ConvNet 57% → ResNet18-frozen 92.2% |
| [Labs/Week8_lab/_run_week8.py](Labs/Week8_lab/_run_week8.py) | ✓ | Financial sentiment: LSTM 71% (base) → DistilBERT 92.2%; weights saved + predict working |
