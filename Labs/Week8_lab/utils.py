import os
import torch


"""
A collection of utility functions for re-use throughout the codebase.

When you reach the "Save Model Weights" task:
In the directory WEIGHT_DIR, we will store the weights in a binary file of the
given name. This will overwrite any weights that already exist with that name.

** Finish the implementations of save_model_state and load_model_state. **

Most of the code has been written already, so all that's left is actually saving
and loading the model state. See the documentation for help.
https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
"""


WEIGHT_DIR = "model_weights"

def save_model_state(model, name):
    """Save model weights to file using the given name"""
    # Make the weights directory in case it doesn't yet exist
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    # Build the full file path. PyTorch uses the .pth extension by convention
    path = os.path.join(WEIGHT_DIR, name + ".pth")

    # TODO: Save the model weights to disk
    # SOLUTION LINE
    torch.save(model.state_dict(), path)
    
    
    print("Model state saved to", path)


def load_model_state(model, name):
    """Restore model weights from the given file name"""
    # Build the full file path
    path = os.path.join(WEIGHT_DIR, name + ".pth")

    # Check that the weights exist
    if os.path.exists(path):
        
        # TODO: Load the model weights from disk
        # SOLUTION LINE
        model.load_state_dict(torch.load(path))
        
        print("Model state loaded from", path)
    else:
        print("No weights found with this name - no action taken...")


def get_training_device():
    """Returns a device appropriate for training."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    return device
