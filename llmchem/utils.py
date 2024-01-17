import torch
import gc
import os
def clean_vram():
    gc.collect()
    torch.cuda.empty_cache()

def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_project_dirs(project_dir):
    mk_dir(project_dir)
    mk_dir(project_dir+"/eval")
    mk_dir(project_dir+"/self_reasoning")
    mk_dir(project_dir+"/train")