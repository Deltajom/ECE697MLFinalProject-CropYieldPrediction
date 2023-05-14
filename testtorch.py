import torch

def test_version():
    return "Torch version" + str(torch.__version__)

def test_CUDA():
    if(torch.cuda.is_available()):
        return 1, "Success! CUDA acceleration enabled"
    return 0, "Fail! CUDA acceleration NOT available"