from transformers import pipeline

classifier = pipeline("zero-shot-classification")
res = classifier(
    "This is a course about python list comprehension",
    candidate_labels = ["edution","politics","business"],
)
print(res)
