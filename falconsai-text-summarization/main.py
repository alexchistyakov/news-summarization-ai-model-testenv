from transformers import pipeline
import json
import time
import nvidia_smi


summarizer = pipeline("summarization", model="Falconsai/text_summarization", device=0)
file = open("../samples.json")
samples = json.load(file)

for article in samples["samples"]:

    start = time.process_time()

    #Run Model on sample
    print(summarizer(article, max_length=220, min_length=60, do_sample=False))

    #Print time elapsed
    end = time.process_time()
    print("---------------------")
    print("Time elapsed: {0} ms".format(end - start))

    #Show memory usage
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory Used : ({:.2f}%) : {} bytes (total), {} bytes (free), {} bytes (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100-(100*info.free/info.total), info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()
    print("---------------------")

