import random
from datasets import Dataset


def gen_compound_text(chemical_record,
    reason="",prediction=""):
    name=chemical_record["name"]
    smiles=chemical_record["smiles"]
    prompt=f"""
#Problem
##Name: {name}
##SMILES: {smiles}"""
    if reason !="" and prediction!="":
        prompt+=f"""
##Reason: {reason}
##Prediction: {prediction}
"""
    else:
        #test mode
        prompt+="""
##Reason: 
"""
    return prompt



def generate_question_prompt(dataset,
                             test_id,
                             n_prompt_examples=5,
                             prompt_dataset=None):

    if prompt_dataset is None:
        candidate_prompt_ids=[i for i in range(len(dataset))]
        candidate_prompt_ids.remove(test_id)
        prompt_dataset=dataset
    else:
        candidate_prompt_ids=[i for i in range(len(prompt_dataset))]
    prompt=""

    #train prompt
    for _ in range(n_prompt_examples):
        id=random.choice(candidate_prompt_ids)
        prompt+=gen_compound_text(prompt_dataset[id],
                                reason=prompt_dataset[id]["Reason"],
                                prediction=prompt_dataset[id]["Prediction(integer)"])
        prompt+="\n"

    #test prompt
    prompt+=gen_compound_text(dataset[test_id])

    return prompt


def tokenize_dataset(context_list, tokenizer):
    data_list = [{"text": i} for i in context_list]
    random.shuffle(data_list)

    # tokenize
    dataset = Dataset.from_dict(
        {"text": [item["text"] for item in data_list[:]]})
    dataset = dataset.map(lambda samples: tokenizer(
        samples['text']), batched=True)

    return dataset


def gen_train_text(dataset):
    train_text_list=[]
    for id in range(len(dataset)):
        prompt=gen_compound_text(dataset[id],
                                    reason=dataset[id]["Reason"],
                                    prediction=dataset[id]["Prediction(integer)"])
        train_text_list.append(prompt)

    return train_text_list