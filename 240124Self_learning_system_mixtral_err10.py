# # (Q)分子構造+(R)理由+(A)物性データセットのLLMによる学習と予測
# - Q&A: 融点データセットを使用
# - R: 自分自身で考えさせて､正解のデータを学習させる

# %%


from llmchem.reasoning import self_reasoning
from llmchem.eval import eval_model
from llmchem.train import train_model
from llmchem.model import init_model
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer
import pandas as pd
import random
import copy
import glob
import json
from datetime import datetime
from llmchem.utils import mk_dir, clean_vram

# %%
# dataset settings
n_test = 50  # number of testing data
# number of training data for checking (i.e., checking everything takes too long, so we check only a part of training data)
n_train_check = 50
n_GPT_reasoning = 10  # number of reasoning data made by GPT
n_generation_iterations = 50   # trial numbers to generate new self reasoning data
max_generations = 10**5
n_prompt_examples = 5

# model settings
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
target_modules = [
    "lm_head",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate",
    "w1",
    "w2",
    "w3"
]

# LoRA settings
r = 32
lora_alpha = r
bit = 16
# bit=8
# bit=4

# train settings
gradient_checkpointing = True
per_device_train_batch_size = 1
epochs = 3
lr = 10**-5

# device settings
device_map = "auto"

# dataset path
dataset_path = "dataset/231225AutoReasoning/240117best_reason_record_11k.csv"

# project path
project_dir = f"results/projects/240124mixtral_self_reasoning_err10_{n_GPT_reasoning}"

# reasoning options
error_threshold = 10  # if abolute error is smaller than this, add to training data

# %%
mk_dir(project_dir)
mk_dir(project_dir+"/eval")
mk_dir(project_dir+"/self_reasoning")
mk_dir(project_dir+"/train")

# %%
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# %%
# load base dataset

df = pd.read_csv(dataset_path)
dataset = df.to_dict(orient="records")
random.seed(0)
random.shuffle(dataset)

base_train_dataset = dataset[:-n_test]
example_reasoning_dataset = base_train_dataset[:n_GPT_reasoning]
test_dataset = dataset[-n_test:]

# %%
# Loop: training, evaluation, data generation
model = None
for generation in range(max_generations):
    # clear_output()
    print(f"Generation: {generation}")
    # prepare train dataset

    # reason data made by GPT4
    train_dataset = copy.deepcopy(example_reasoning_dataset)

    print(f"GPT-generated reasons: {len(train_dataset)}")

    # reason data made by model itself
    for path in glob.glob(f"{project_dir}/self_reasoning/*.json"):
        with open(path) as f:
            train_dataset.append(json.load(f))

    print(f"All-generated reasons: {len(train_dataset)}")
    random.shuffle(train_dataset)

    # train model
    del model
    clean_vram()
    model = init_model(model_name, r, lora_alpha,
                       target_modules, bit=bit, device_map=device_map)
    train_result = train_model(model, tokenizer, train_dataset,
                               project_dir=project_dir,
                               epochs=epochs,
                               lr=lr,
                               per_device_train_batch_size=per_device_train_batch_size,
                               gradient_checkpointing=gradient_checkpointing,
                               )

    # eval
    # if len(train_dataset) < 1000:
    #    n_prompt_examples = 5
    # else:
    #    n_prompt_examples = 0

    train_check_dataset = copy.deepcopy(train_dataset[:n_train_check])
    random.shuffle(train_check_dataset)
    train_eval_result = eval_model(model, tokenizer, train_check_dataset,
                                   f"{project_dir}/eval",
                                   n_prompt_examples=n_prompt_examples,
                                   prompt_dataset=example_reasoning_dataset,
                                   prefix=f"train_{generation}"
                                   )

    test_eval_result = eval_model(model, tokenizer, test_dataset,
                                  f"{project_dir}/eval",
                                  n_prompt_examples=n_prompt_examples,
                                  prompt_dataset=example_reasoning_dataset,
                                  prefix=f"test_{generation}"
                                  )

    # generate additional training data by self-reasoning
    self_reasoning(model, tokenizer, base_train_dataset,
                   example_reasoning_dataset, project_dir,
                   generation=generation,
                   n_iterations=n_generation_iterations,
                   error_threshold=error_threshold,
                   n_max_trials=2,
                   n_prompt_numbers=(5, 5),
                   )

# %%


# %%
