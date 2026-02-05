import ai
lm = ai.LM(api_base="http://192.168.170.76:8000")
pred = ai.Predict("q -> classify:bool", lm=lm)
print(pred(q="2+2=4"))