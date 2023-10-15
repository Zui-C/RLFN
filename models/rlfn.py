import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.densenet import DenseNet
import torch
from torch import nn

from models.fusion import DetachGatedMultimodalLayer,GatedMultimodalLayer
from models.correct import MathbertaEmbeddingNet,CorrectionModel


class RLFN(nn.Module):
    def __init__(self, params=None):
        super(RLFN, self).__init__()
        self.params = params
        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)
        self.out_channel = params['words_num']
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()

        self.sizein1 = params['fusion']['sizein1']
        self.sizein2 = params['fusion']['sizein2']
        self.sizeout = params['fusion']['sizeout']

        self.language_encoder = MathbertaEmbeddingNet(self.params)
        self.gate_fusion = GatedMultimodalLayer(self.out_channel,self.sizein1,self.sizein2,self.sizeout)
        self.detach_gate_fusion = DetachGatedMultimodalLayer(self.out_channel,self.sizein1,self.sizein2,self.sizeout)

        self.ratio = params['densenet']['ratio']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):
        cnn_features = self.encoder(images)

        word_probs, word_alphas, word_out_states = self.decoder(cnn_features, labels,  images_mask, labels_mask, is_train=is_train)
        word_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels.view(-1))
        word_average_loss = (word_loss * labels_mask.view(-1)).sum() / (labels_mask.sum() + 1e-10) if self.use_label_mask else word_loss

        fusion_losses = []
        fusion_probs = word_probs
        predicts = torch.argmax(fusion_probs, dim=2)

        correct_outs = self.language_encoder(predicts)
        fusion_probs = self.detach_gate_fusion(word_out_states, correct_outs)

        fusion_loss = self.cross(fusion_probs.contiguous().view(-1, fusion_probs.shape[-1]), labels.view(-1))
        fusion_losses.append(fusion_loss)
        fusion_loss = sum(fusion_losses) / len(fusion_losses)


        return fusion_probs, word_average_loss, fusion_loss
