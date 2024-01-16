import torch
import gc
import os
def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()

def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)