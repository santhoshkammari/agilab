import ai
from file_tools import tools as file_tools

# --- AI Config ---
lm = ai.LM(api_base="http://192.168.170.76:8000")
ai.configure(lm)
# -----------------

pred = ai.Predict("q -> answer", tools=file_tools, verbose=True)
result = pred(q="List files in current directory")
print(result)
