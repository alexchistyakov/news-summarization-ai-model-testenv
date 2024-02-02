import json
import time

from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mhenrichsen/context-aware-splitter-1b-english").to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("mhenrichsen/context-aware-splitter-1b-english")
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

file = open("../samples.json")
samples = json.load(file)

prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to segment text into smaller blocks. Split the text where it makes sense and be vary of the context. The ideal split should be close to {word_count} words.

### Input:
{text}

### Response:
"""

WORD_SPLIT_COUNT = 200

for article in samples["long-samples"]:

    start = time.process_time()

    tokens = tokenizer(
        prompt_template.format(text=article, word_count=WORD_SPLIT_COUNT),
        return_tensors='pt'
    )['input_ids'].to("cuda:0")

    # Generate output
    generation_output = model.generate(
        tokens,
        streamer=streamer,
        max_length = 8192,
        eos_token_id = 29913
    ).to("cuda:0")

    print(generation_output)

    #Print time elapsed
    end = time.process_time()
    print("Time elapsed: {0} ms".format(end - start))

