from transformers import pipeline

classifier = pipeline("sentiment-analysis")
res = classifier("Today is a nice day !")
print(res)
