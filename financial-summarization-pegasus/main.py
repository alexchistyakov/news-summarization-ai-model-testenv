from transformers import PegasusTokenizer, PegasusForConditionalGeneration, TFPegasusForConditionalGeneration
import json

# Let's load the model and the tokenizer
model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name) # If you want to use the Tensorflow model
                                                                    # just replace with TFPegasusForConditionalGeneration


# Some text to summarize here
file = open("../samples.json",encoding="utf8")
samples = json.load(file)

for text_to_summarize in samples["samples"]:
    # Tokenize our text
    # If you want to run the code in Tensorflow, please remember to return the particular tensors as simply as using return_tensors = 'tf'
    input_ids = tokenizer(text_to_summarize, return_tensors="pt").input_ids

    # Generate the output (Here, we use beam search but you can also use any other strategy you like)
    output = model.generate(
        input_ids,
        max_length=32,
        num_beams=5,
        early_stopping=True
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

file.close()
