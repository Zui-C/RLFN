import os
import cv2
import argparse
import torch
import json
import pickle as pkl
from tqdm import tqdm
import time

from utils import load_config, compute_edit_distance, load_infer_checkpoint
from models.infer_model import Inference
from dataset import Words

parser = argparse.ArgumentParser(description='model testing')
parser.add_argument('--dataset', default='2014', type=str)

parser.add_argument('--word_path', default='./datasets/words_dict.txt', type=str)
args = parser.parse_args()


config_file = 'config.yaml'
if args.dataset == '2014':
    args.image_path = './datasets/14_test_images.pkl'
    args.label_path = './datasets/14_test_labels.txt'
elif args.dataset == '2016':
    args.image_path = './datasets/16_test_images.pkl'
    args.label_path = './datasets/16_test_labels.txt'
elif args.dataset == '2019':
    args.image_path = './datasets/19_test_images.pkl'
    args.label_path = './datasets/19_test_labels.txt'


params = load_config(config_file)

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
params['device'] = device
words = Words(args.word_path)
params['word_num'] = len(words)

if 'use_label_mask' not in params:
    params['use_label_mask'] = False
print(params['decoder']['net'])
model = Inference(params)
model = model.to(device)

load_infer_checkpoint(model, None, params['checkpoint'])
model.eval()

with open(args.image_path, 'rb') as f:
    images = pkl.load(f)

with open(args.label_path) as f:
    lines = f.readlines()

line_right = 0
e1, e2, e3 = 0, 0, 0
bad_case = {}
model_time = 0

prediction_dict = {}
with torch.no_grad():
    for line in tqdm(lines):
        name, *labels = line.split()
        name = name.split('.')[0] if name.endswith('jpg') else name

        input_labels = labels
        labels = ' '.join(labels)
        img = images[name]
        img = torch.Tensor(255-img) / 255
        img = img.unsqueeze(0).unsqueeze(0)
        img = img.to(device)
        a = time.time()


        probs = model(img)
        model_time += (time.time() - a)

        prediction = words.decode(probs)

        if prediction == labels:
            line_right += 1
        else:
            bad_case[name] = {
                'label': labels,
                'predi': prediction
            }

        prediction_dict[name] = prediction
        distance = compute_edit_distance(prediction, labels)
        if distance <= 1:
            e1 += 1
        if distance <= 2:
            e2 += 1
        if distance <= 3:
            e3 += 1

print(f'model time: {model_time}')
print(f'ExpRate: {line_right / len(lines)}')

print(f'e1: {e1 / len(lines)}')
print(f'e2: {e2 / len(lines)}')
print(f'e3: {e3 / len(lines)}')

#
# with open('predictions/RLFN_2019.txt', 'w') as file:
#     for name, prediction in prediction_dict.items():
#         file.write(name + '\t' + prediction + '\n')