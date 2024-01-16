import random
from datetime import datetime
import json
from llmchem.utils import clean_vram
from llmchem.inference import ask_value
from llmchem.dataset import generate_question_prompt
from tqdm import tqdm
import copy


def self_reasoning(model,tokenizer,dataset,prompt_dataset,save_dir,
    n_iterations=100,
    n_max_trials=4,  # 値を返さなかったときの再試行の最大数
    error_threshold=30, #結果を保存する許容誤差
                   ):

    #prompt tuningをランダムに変えながら､訓練データで予測(自習)していく
    for train_id in tqdm(range(n_iterations)):
        #clear_output()
        clean_vram()
        for _ in range(n_max_trials):
            try:
            #if True:

                n_prompt_examples=random.randint(1,5) #number of prompt tuning
                prompt=generate_question_prompt(dataset,train_id,
                                                n_prompt_examples=n_prompt_examples,
                                                prompt_dataset=prompt_dataset,
                                               )
                reason,value=ask_value(prompt,model,tokenizer)
            except Exception as e:
                print(e)
                continue

            if len(reason)<30:
                continue

            if value is not None:
                try:
                    value=float(value)
                except:
                    continue

                record=copy.deepcopy(dataset[train_id])
                record["Reason"]=reason
                record["Prediction(integer)"]=value

                err=abs(record["mpC"]-float(value))
                print("actual: ",record["mpC"],"predicted: ", value,"err: ",err)
                print(reason)

                
                if err<error_threshold:
                    save_path=save_dir+f"/{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
                    with open(save_path,"w") as f:
                        json.dump(record,f,indent=4)

                    break