import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from models.densenet import DenseNet
from models.attention import Attention
from models.decoder import PositionEmbeddingSine

from models.fusion import GatedMultimodalLayer,DetachGatedMultimodalLayer
from models.correct import CorrectionModel,MathbertaEmbeddingNet

class Inference(nn.Module):
    def __init__(self, params=None):
        super(Inference, self).__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']
        self.encoder = DenseNet(params=self.params)
        self.out_channel = params['words_num']
        self.decoder = AttDecoder(params=self.params)
        self.cross = nn.CrossEntropyLoss(reduction='none') if self.use_label_mask else nn.CrossEntropyLoss()


        self.sizein1 = params['fusion']['sizein1']
        self.sizein2 = params['fusion']['sizein2']
        self.sizeout = params['fusion']['sizeout']

        self.language_encoder = MathbertaEmbeddingNet(self.params)
        self.gate_fusion = GatedMultimodalLayer(self.out_channel,self.sizein1,self.sizein2,self.sizeout)
        self.detach_gate_fusion = DetachGatedMultimodalLayer(self.out_channel,self.sizein1,self.sizein2,self.sizeout)

    def forward(self, images):
        cnn_features = self.encoder(images)
        batch_size, _, height, width = cnn_features.shape

        word_probs, word_alphas, word_out_states = self.decoder(cnn_features)
        word_probs = torch.stack(word_probs).to(images.device).unsqueeze(0)
        word_out_states = torch.stack(word_out_states).to(images.device).unsqueeze(0)


        predicts = torch.argmax(word_probs, dim=2)
        correct_outs = self.language_encoder(predicts)
        fusion_probs = self.detach_gate_fusion(word_out_states, correct_outs)

        correct_words = torch.argmax(fusion_probs, dim=2).squeeze(0)

        return correct_words



class AttDecoder(nn.Module):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        self.ratio = params['densenet']['ratio']

        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim, kernel_size=1)
        self.word_attention = Attention(params)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)


        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features):
        batch_size, _, height, width = cnn_features.shape
        image_mask = torch.ones((batch_size, 1, height, width)).to(self.device)

        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, image_mask[:, 0, :, :])
        cnn_features_trans = cnn_features_trans + pos

        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        hidden = self.init_hidden(cnn_features, image_mask)
        word_embedding = self.embedding(torch.ones([batch_size]).long().to(device=self.device))
        word_out_states = []
        word_probs = []
        word_alphas = []

        i = 0
        while i < 200:
            hidden = self.word_input_gru(word_embedding, hidden)
            word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                               word_alpha_sum, image_mask)

            current_state = self.word_state_weight(hidden)
            word_weighted_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)

            if self.params['dropout']:
                word_out_state = self.dropout(
                    current_state + word_weighted_embedding + word_context_weighted)
            else:
                word_out_state = current_state + word_weighted_embedding + word_context_weighted

            word_prob = self.word_convert(word_out_state)
            _, word = word_prob.max(1)
            word_embedding = self.embedding(word)
            if word.item() == 0:
                return word_probs, word_alphas, word_out_states
            word_alphas.append(word_alpha)
            word_probs.append(word_prob.squeeze(0))
            word_out_states.append(word_out_state.squeeze(0))
            i += 1
        return word_probs, word_alphas, word_out_states

    def init_hidden(self, features, feature_mask):
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)


