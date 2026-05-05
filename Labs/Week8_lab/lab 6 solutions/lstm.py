import torch.nn as nn


"""
Long Short-Term Memory (LSTM) model. These were once the "go-to" type of model
for language modelling tasks, but have been replaced by transformer models.

Read through the code below to get an idea of how this model works. Don't worry
if you don't fully understand it, as LSTMs can be tricky!
"""


class LSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dims, num_layers, hidden_dims):
        super().__init__()
        self.num_classes = num_classes
        # Size of the LSTM's hidden state vector
        self.hidden_dims = hidden_dims
        # This transforms tokenised inputs into learnt embedding spaces
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        # The LSTM module performs the recursive processing for us
        self.lstm = nn.LSTM(embedding_dims, hidden_dims, num_layers, batch_first=True)
        # This is a classification head which takes the LSTM's output and predicts classes
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, num_classes)
        )

    def forward(self, x):
        # Embed the input sequence
        embedded_words = self.embedding(x)
        # Feed the embedded sequence through the LSTM
        lstm_out, _ = self.lstm(embedded_words)
        # Keep the LSTM outputs for only the last time-step
        lstm_out = lstm_out[:, -1]
        # Feed the outputs through the classifier to get class predictions
        output = self.fc(lstm_out)
        return output
