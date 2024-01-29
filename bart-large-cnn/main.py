from transformers import pipeline
import json

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
file = open("../samples.json")
samples = json.load(file)

for article in samples["samples"]:
    print(summarizer(article, max_length=130, min_length=30, do_sample=False))
