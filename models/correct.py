
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from transformers import AutoModel
from utils import token_word_map


class MathbertaEmbeddingNet(torch.nn.Module):
    def __init__(self, params):
        super(MathbertaEmbeddingNet, self).__init__()

        self.bert_model = AutoModel.from_pretrained('witiko/mathberta')
        self.token_map = torch.tensor(token_word_map(params['word_path'], params['token_path'])).to(params['device'])

        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, words):

        batch_size = words.shape[0]
        expanded_token_map = self.token_map.unsqueeze(0).expand(batch_size, -1)
        tokens = torch.gather(expanded_token_map, 1, words)

        with torch.no_grad():
            outputs = self.bert_model(tokens)

        return outputs.last_hidden_state


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CorrectionModel(nn.Module):
    def __init__(self, params):
        super(CorrectionModel, self).__init__()
        self.out_channel = params['words_num']

        embedding_dim = 768
        num_heads = 8
        hidden_dim = 256
        dropout = 0.1
        num_layers = 2
        self.embedding_dim = embedding_dim
        self.params = params
        self.encode = MathbertaEmbeddingNet(self.params)
        self.embedding = nn.Embedding(self.out_channel, self.embedding_dim)
        self.pos_encoder = PositionalEncoding(self.embedding_dim, dropout)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(self.embedding_dim, num_heads, hidden_dim,
                                       dropout),
            num_layers
        )
        self.bert_to_decoder = nn.Linear(768, self.embedding_dim)
        self.dropout = nn.Dropout(0.3)

        self.linear = nn.Linear(self.embedding_dim, self.out_channel)



    def forward(self, label, label_mask, predicts, is_train):
        batch_size, num_steps = label.shape
        sos_token = torch.ones(batch_size, 1).long().to(label.device)
        predicts = torch.cat((sos_token, predicts), dim=1)
        predicts_embedded = self.encode(predicts)  # shape: (batch_size, seq_len, embedding_dim)
        predicts_embedded = predicts_embedded.transpose(0, 1)  # shape: (seq_len, batch_size, bert_dim)
        # 0606
        # predicts_embedded = self.bert_to_decoder(predicts_embedded)

        # predicts_embedded = self.dropout(predicts_embedded)
        if is_train:
            input = torch.cat((sos_token, label[:,:-1]), dim=1)  # shape: (batch_size, seq_len + 1)
            # sos_mask = torch.ones_like(sos_token).long()  # shape: (batch_size, 1)
            # decoder_mask = torch.cat((sos_mask, label_mask[:,:-1]), dim=1)  # shape: (batch_size, seq_len + 1)
            decoder_input = self.embedding(input)  # shape: (batch_size, seq_len + 1, embedding_dim)
            decoder_input = self.pos_encoder(
                decoder_input.transpose(0, 1))  # shape: (seq_len + 1, batch_size, embedding_dim)

            tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(0)).to(label.device)
            output = self.transformer_decoder(
                tgt=decoder_input,
                memory=predicts_embedded,
                tgt_mask=tgt_mask,
                # tgt_key_padding_mask=~(decoder_mask.bool())
            )
            output = output.transpose(0, 1)  # shape: (batch_size, seq_len, vocab_size)
            corr_probs = self.linear(output)  # shape: (seq_len, batch_size, vocab_size)
            # output = output[:,1:]

            return corr_probs, output

        else:
            word_probs = torch.zeros((batch_size, num_steps, self.out_channel)).to(device=label.device)
            word_outs = torch.zeros((batch_size, num_steps, self.embedding_dim)).to(device=label.device)

            # Auto-regressive inference
            input = torch.tensor([1]).to(label.device).unsqueeze(0)  # start token

            for i in range(num_steps):

                decoder_input = self.embedding(input)
                decoder_input = self.pos_encoder(decoder_input.transpose(0, 1))
                # decoder_input = self.encode(input)
                # decoder_input = decoder_input.transpose(0, 1)

                tgt_mask = self._generate_square_subsequent_mask(decoder_input.size(0)).to(label.device)


                output = self.transformer_decoder(
                    tgt=decoder_input,
                    memory=predicts_embedded,
                    tgt_mask=tgt_mask
                )
                output = output.transpose(0, 1)  # shape: (batch_size, seq_len, vocab_size)
                corr_probs = self.linear(output)  # shape: (seq_len, batch_size, vocab_size)

                word_prob = corr_probs[:, -1, :]
                word_out = output[:, -1, :]

                next_token = word_prob.argmax(dim=-1).unsqueeze(-1)
                input = torch.cat([input, next_token], dim=-1)
                word_probs[:, i] = word_prob
                word_outs[:, i] = word_out
            # print(input_sequence)

            return word_probs, word_outs


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask





