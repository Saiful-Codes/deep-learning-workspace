from transformers import AutoModelForSequenceClassification
import torch.nn as nn

class DistBert(nn.Module):
    def __init__(self, device):
      super().__init__()

      # TODO:
      # fill in code below to instantiate the current model 'distilbert-base-uncased' to perform the classification
      # self.model = .....
   

      self.model.to(device)

    def forward(self, x):
      # TODO:
      # fill in code below to return outputs of the model. Note the model outputs the following
      # two things via a dictionary: loss and logits. Choose the right one to return
   
