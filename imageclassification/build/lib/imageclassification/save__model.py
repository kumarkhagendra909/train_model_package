import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

def save_model(model):
    """
    Contains code to save the model

    Args: 
    input: Model which we are using for predictions 

    Output: It Save the model in .pt format with name mymodel_script. 
    file is saved at main folder or your project. 
    to check where your main folder is You can use os.walkthrough or os.getpath.
    """
    torchscript_model = torch.jit.script(model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized, "mymodel_script.pt")
    print("Saved your model")
    
