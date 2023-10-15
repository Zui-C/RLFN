import torch
import numpy as np
import re
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForPreTraining

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '5'


l1,l2,l3,l4 = 0,0,0,0
p1,p2,p3,p4 = 0,0,0,0


def calculate_perplexity(tokens):
    global l1,l2,l3,l4
    global p1,p2,p3,p4

    # Traverse each position of a sentence
    for i in range(len(tokens)):
        # Create a copy and replace the word in the current position with the [MASK] mark
        masked_tokens = tokens[:]
        symbol = masked_tokens[i]
        masked_tokens[i] = '[MASK]'
        # Encode sentences and add batch dimensions
        input_ids = torch.tensor([[0] + tokenizer.convert_tokens_to_ids(masked_tokens) + [2]]).to(device)
        # input_ids = torch.tensor([tokenizer.encode(masked_tokens, add_special_tokens=True)])
        # Forward propagation through the model to obtain predicted words
        outputs = model(input_ids)
        predictions = outputs.prediction_logits

        # Calculate the Perplexity of the current position
        perplexity = torch.nn.CrossEntropyLoss()(predictions.squeeze(), input_ids.squeeze()).detach()

        if symbol == 'sos' or symbol == 'eos':
            continue
        category = classify_symbol(symbol)
        if category == "letter":
            l1 += 1
            p1 += perplexity
        elif category == "number":
            l2 += 1
            p2 += perplexity
        elif category == "structure":
            l3 += 1
            p3 += perplexity
        else:  # category == "math"
            l4 += 1
            p4 += perplexity




def classify_symbol(symbol):
    if re.match(r"(\\theta|\\beta|\\sigma|\\phi|\\mu|\\gamma|\\Delta|\\pi|\\alpha|\\lambda|[a-zA-Z]+)", symbol):
        return "letter"
    if re.match(r"\d+", symbol):
        return "number"
    if re.match(r"[_^{}]|\\{|\\}", symbol):
        return "structure"
    return "math"

def stat(labels):
    perplexities_label = []
    for line in tqdm(labels):
        name, *prediction = line.strip().split()
        calculate_perplexity(prediction)

label_path = 'predictions/DWAP_2014.txt'
with open(label_path, 'r') as file:
    labels = file.readlines()

tokenizer = AutoTokenizer.from_pretrained("AnReu/math_pretrained_bert")
model = AutoModelForPreTraining.from_pretrained("AnReu/math_pretrained_bert")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print('label----------------------------------------')
stat(labels)
print(l1,l2,l3,l4,l1+l2+l3+l4)
print(p1/l1,p2/l2,p3/l3,p4/l4)
print((p1+p2+p3+p4)/(l1+l2+l3+l4))

# print('predictions----------------------------------------')
# stat(predictions)
# print('model_predictions----------------------------------------')
# stat(model_predictions)
