import os
import cv2
import yaml
import math
import torch
import numpy as np
from difflib import SequenceMatcher


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params

def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr   
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_checkpoint(model, optimizer, word_score, ExpRate_score, epoch, optimizer_save=False, path='checkpoints'):
    filename = f'{os.path.join(path, model.name)}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'
    if optimizer_save:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.state_dict()
        }
    torch.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path):
    state = torch.load(path, map_location='cpu')
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')
    model.load_state_dict(state['model'])


def load_infer_checkpoint(model, optimizer, path):
    state = torch.load(path, map_location='cpu')
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')
    model.load_state_dict(state['model'],strict=False)


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
              for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate


def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance


def token_word_map(words_path, token_path):
    import json
    with open(words_path) as f:
        words = f.readlines()
    with open(token_path, "r") as file:
        token_dict = json.load(file)
    token_index = [token_dict[words[i].strip()] for i in range(len(words))]
    return token_index