from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class DistBert(nn.Module):
    def __init__(self, device):
      super().__init__()
      self.model =  AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
      self.model.to(device)

    def forward(self, x):
      return self.model(x)['logits']

