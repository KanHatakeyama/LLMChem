from .utils.GPT import json_generate
from .utils.prompt import prompt_ask_for_reason,prompt_predict_value,mask_target_value
def generate_reason_and_predict(name,value,
                                qa_gen_command,
                                predict_command,
                                llm_ask_func=json_generate,
                                smiles="",
                                gen_reason=True,
                                n_trials=1):
    """
    Generate reason from Q&A and predict value from Q&A.
    """
    record={
        "name":name,
        "value":value,
        "smiles":smiles,
    }

    prompt=prompt_ask_for_reason(name=record['name'],value=record["value"],
                        smiles=record['smiles'],command=qa_gen_command)

    if gen_reason:
        #generate reason
        res=llm_ask_func(prompt=prompt)
        reason=mask_target_value(res["Reason"],record["value"])
    else:
        reason=""

    #predict value with reason
    prompt=prompt_predict_value(name=record['name'],reason=reason,
                                smiles=record['smiles'],command=predict_command)

    #try multiple times
    value_list=[]
    for _ in range(n_trials):
        res=llm_ask_func(prompt=prompt)
        value_list.append(res["Value"])

    record["generated_reason"]=reason
    record["predicted_values"]=value_list

    return record