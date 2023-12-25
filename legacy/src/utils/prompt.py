default_ask_command="""
Provide the quantitative reasons within 300 words so that a scientist, who does not know the melting point, can predict the value.
We must quantitatively consider how the melting point shifts.
I absolutely forbid you to make qualitative generalizations.

#Bad example reasons
## Its molecular weight compared to simpler aromatic compounds, contributing to a higher melting point. (qualitative discussion is practically meaningless!!!)
## Therefore, the compound has a melting point of 110°C (Never include the answer in the reason!!).

#Good example reasons
## Benzene has a boiling point of 80 degrees, but toluene with a methyl group improves the boiling point by about +30 degrees due to its larger molecular weight and other reasons (+30 contribution).
## Butane has a boiling point of -1°C, but butanol with a hydroxy group has a boiling point increase of about 115°C due to the strong influence of hydrogen bonding (+115 contribution).

#Output: Reason key
"""

default_predict_command="""
Predict the melting point of the following compound.
In any case, output some value.

#Output: Value key
"""

# prompt to generate "reason" from Q&A
def prompt_ask_for_reason(name,value,smiles="",command=default_ask_command):
    prompt=command

    prompt+=f"""
Name: {name}"""

    if smiles!="":
        prompt+=f"""
SMILES: {smiles}"""

    prompt+=f"""
Value: {value}
Reason: """

    return prompt.strip()

# predict value from Q( and reason)
def prompt_predict_value(name,reason="",smiles="",command=default_predict_command):
    prompt=command

    prompt+=f"""
Name: {name}"""

    if smiles!="":
        prompt+=f"""
SMILES: {smiles}"""

    if reason!="":
        prompt+=f"""
Reason: {reason}"""
    
        prompt+=f"""
Value: """
            
    return prompt.strip()


#mask target value in the reason text, if included
def mask_target_value(reason,value):
    reason=reason.replace(str(value),"[MASK]")
    return reason

