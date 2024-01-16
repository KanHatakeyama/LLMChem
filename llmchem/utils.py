import torch
import gc
def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()