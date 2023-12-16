default_command="""Provide the quantitative reasons within 300 words so that a scientist, who does not know the melting point, can predict the value."""

def prompt_ask_for_reason(name,value,smiles="",command=default_command):
    prompt=command

    prompt+=f"""
Name: {name}"""

    if smiles!="":
        prompt+=f"""
SMILES: {smiles}"""

    prompt+=f"""
Value: {value}
Reason: """

    return prompt   