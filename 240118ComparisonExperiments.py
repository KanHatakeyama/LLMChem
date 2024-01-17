# %%

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer
import pandas as pd
import random
import copy
from llmchem.utils import clean_vram, make_project_dirs
from llmchem.model import init_model
from llmchem.train import train_model
from llmchem.eval import eval_model

# %%
# dataset settings
n_test = 50  # number of testing data
# number of training data for checking (i.e., checking everything takes too long, so we check only a part of training data)
n_train_check = 50


bit = 16
# bit=8
# bit=4

# train settings
gradient_checkpointing = False
per_device_train_batch_size = 1
lr = 10**-5

# device settings
device_map = "auto"

# dataset path
dataset_path = "dataset/231225AutoReasoning/240117best_reason_record_11k.csv"


# %%
model_dict = {
    "Llama2-7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "modules": [
            # "embed_tokens",
            "lm_head",
            # "q_proj",
            # "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            # "down_proj",
        ]
    },
    "Mixtral": {
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "modules": [
            "lm_head",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate",
            "w1",
            "w2",
            "w3"],
    },
    "Llama2-7b": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "modules": [
            # "embed_tokens",
            "lm_head",
            # "q_proj",
            # "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            # "down_proj",
        ]
    },
}


df = pd.read_csv(dataset_path)
# %%
for model_nickname in model_dict:
    model_name = model_dict[model_nickname]["name"]
    target_modules = model_dict[model_nickname]["modules"]
    for epochs in [3, 1, 5]:
        for r in [32,  64, 128]:
            lora_alpha = r
            for n_train in [10, 50, 100, 500, 1000, 5000, 10000]:
                # project path
                project_dir = f"results/projects/240118comparisons/{model_nickname}_{epochs}_{r}_{n_train}"

                # make project dir
                make_project_dirs(project_dir)

                # load dataset
                dataset = df.to_dict(orient="records")
                random.seed(0)
                random.shuffle(dataset)

                train_dataset = dataset[:n_train]
                test_dataset = dataset[-n_test:]

                # prepare train dataset
                print(f"GPT-generated reasons: {len(train_dataset)}")

                random.shuffle(train_dataset)

                # train model
                clean_vram()
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
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
                train_check_dataset = copy.deepcopy(
                    train_dataset[:n_train_check])
                random.shuffle(train_check_dataset)

                # eval_settings
                if len(train_dataset) < 1000:
                    n_prompt_examples = 3
                else:
                    n_prompt_examples = 0

                train_eval_result = eval_model(model, tokenizer, train_check_dataset,
                                               f"{project_dir}/eval",
                                               n_prompt_examples=n_prompt_examples,
                                               prefix=f"train"
                                               )

                test_eval_result = eval_model(model, tokenizer, test_dataset,
                                              f"{project_dir}/eval",
                                              n_prompt_examples=n_prompt_examples,
                                              prefix=f"test"
                                              )
