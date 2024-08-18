import subprocess
import os

# definir o token
os.environ['HF_TOKEN'] = '' # precisa criar o token no hugging face

command = ['python', r"C:\Users\emily\OneDrive\Documents\GitHub\CVLface\cvlface\apps\verification\verify.py", '--data_root', r"C:\Users\emily\Downloads\RFW_dataset\separacao\clusters"]

# executar
result = subprocess.run(command, capture_output=True, text=True)

# mostrar a saída
print(result.stdout)
print(result.stderr)
