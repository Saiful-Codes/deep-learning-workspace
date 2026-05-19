import torch
import pandas as pd
from transformers import AutoTokenizer


"""
Welcome to the TextDataset class! This is a pretty standard PyTorch Dataset class,
except that it also reads a text classification CSV from disk and tokenizes it.

The class takes three init arguments:
    path: the path of the CSV to read
    max_sequence_len: the maximum number of tokens for any example

You need fill in missing parts inside the __init__ function to tokenize the texts.
We will set truncation and padding to True, so all of the tokenised inputs are of 
the same length,filled with zeros where necessary. 
You will need to pass the max_sequence_len parameter into the tokenizer so the
tokenizer knows when it needs to truncate the input.


**Fill in the TODOs in the __init__ function**

You will also need to complete the __getitem__ function. This should grab the 
example and label for the given index, initialise a tensor for each, and return
them. 

**Fill in the TODOs in the __getitem__ function**

When you're finished, return to the "Write the Dataset Class" section of the notebook.
"""


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_sequence_len=128):

        df = pd.read_csv(path)

        texts = df['text'].tolist()        
        labels = df['label'].tolist()
        
        # TODO: set the tokenizer by using the AutoTokenizer to get 
        #       a tokenizer for the pretrained distilbert-base-uncased
        #       transformer
        # Solution
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Store the number of words in the tokenizer's vocabulary
        self.vocab_size = tokenizer.vocab_size
    
        # TODO: Call tokenizer with texts as an argument, setting the arguments
        # truncation and padding to True, set max_length to max_sequence_len,
        # finally set return_tensors to "pt" (which is a pytorch tensor).
        # The max_length is use to truncate the tokens for each example to be less
        # than a maximum length. This is because each row of your data needs to have
        # the same fixed length.
        # Take a look at this link for some examples of using the tokenizer function.
        #    https://huggingface.co/course/chapter2/6?fw=pt
        # tokens = tokenizer( ...... )
    
        # SOLUTION LINE
        tokens = tokenizer(texts, max_length=max_sequence_len, truncation=True, 
                           padding=True,return_tensors="pt")

        # TODO: Store the input ids of the tokens to self.tokens and labels from
        # above to self.labels
        # See the "write a dataclass" section for how to select the input ids.
        # Note: Each token is actually a word index into a dictionary, so we are
        # storing data in numerical form; all ready to be given to a model.
        # self.tokens = ...
        # self.labels = ...

        # SOLUTION LINE
        self.tokens = tokens["input_ids"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # TODO: Set the tokens variable to be the tokens at index idx
        #       Set the label variable to be the label at index idx.
        # Note: self.tokens is a block of data shaped
        #   [N, max_sequence_len]
        # where N is the number of examples in our dataset, and the tokens
        # variable here should be shaped
        #   [max_sequence_len]
        # tokens = ...
        # label = ...

        # Solutions
        tokens = self.tokens[idx]
        label = self.labels[idx]

        return tokens, label
