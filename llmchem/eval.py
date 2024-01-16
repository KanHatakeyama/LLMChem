from tqdm import tqdm
import copy
from .dataset import generate_question_prompt
from .inference import ask_value
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from datetime import datetime
import json


def eval_model(model,tokenizer,dataset,save_dir,
                prompt_dataset=None,
               n_prompt_examples=0,
               n_max_trials=1,
               prefix=""):

    model.eval()

    n_problems=len(dataset)

    res_list=[]
    for test_id in tqdm(range(n_problems)):
        print(f"promlem {test_id+1} / {n_problems}")
        for _ in range(n_max_trials):
            try:
                prompt=generate_question_prompt(dataset,test_id,n_prompt_examples=n_prompt_examples,prompt_dataset=prompt_dataset)
                reason,value=ask_value(prompt,model,tokenizer)
            except Exception as e:
                print(e)
                continue


            if value is not None:
                record=copy.deepcopy(dataset[test_id])
                record["Test (Predicted reason)"]=reason
                record["Test (Predicted value)"]=value
                print("actual: ",record["mpC"],"predicted: ", record["Test (Predicted value)"],)
                res_list.append(record)
                break


    save_name=save_dir+f"/{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    score_dict=score_result(res_list,save_name,n_problems)

    save_json_filename=save_name+".json"
    with open(save_json_filename,"w") as f:
        json.dump(score_dict,fp=f,indent=4)
    return score_dict

def score_result(records,save_name,n_problems,
                     vmin=-200,
                        vmax=400,):
    if len(records)==0:
        return None
    
    sel_df=pd.DataFrame(records)
    #convert to float
    sel_df["Test (Predicted value)"] = pd.to_numeric(sel_df["Test (Predicted value)"], errors='coerce')
    sel_df=sel_df[sel_df["Test (Predicted value)"].notnull()]

    #スコア
    mse=mean_squared_error(sel_df["mpC"],sel_df["Test (Predicted value)"])
    mae=mean_absolute_error(sel_df["mpC"],sel_df["Test (Predicted value)"])
    r2=r2_score(sel_df["mpC"],sel_df["Test (Predicted value)"])

    if len(sel_df)==0:
        return None


    plt.figure()
    sns.scatterplot(data=sel_df,x="mpC",y="Test (Predicted value)")
    plt.title(f"MSE={mse:.0f}")

    #x,yの範囲を揃える
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    #対角線を描く
    plt.plot([vmin,vmax],[vmin,vmax],color="gray")

    plt.savefig(save_name+".png")
    #break

    results={
        "MSE":mse,
        "MAE":mae,
        "R2":r2,
        "Answer ratio":sel_df.shape[0]/n_problems,
        "plot":records
    }



    return results


