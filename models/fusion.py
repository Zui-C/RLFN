
import torch
import torch.nn as nn


class DetachGatedMultimodalLayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, out_channel, size_in1, size_in2, size_out):
        super(DetachGatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out)
        self.hidden2 = nn.Linear(size_in2, size_out)
        self.hidden_att = nn.Linear(size_out * 2, size_out)
        self.cls = nn.Linear(size_out, out_channel)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):


        x1 = x1.detach()  # detach x1
        x2 = x2.detach()  # detach x2

        h1 = self.hidden1(x1)
        h2 = self.hidden2(x2)

        h1 = self.tanh_f(h1)
        h2 = self.tanh_f(h2)

        x = torch.cat((h1, h2), dim=2)
        z = self.sigmoid_f(self.hidden_att(x))
        fusion_features = z * h1 + (1 - z) * h2

        fusion_probs = self.cls(fusion_features)

        return fusion_probs

class GatedMultimodalLayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, out_channel, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out)
        self.hidden2 = nn.Linear(size_in2, size_out)
        self.hidden_att = nn.Linear(size_out * 2, size_out)
        self.cls = nn.Linear(size_out, out_channel)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):

        h1 = self.hidden1(x1)
        h2 = self.hidden2(x2)

        h1 = self.tanh_f(h1)
        h2 = self.tanh_f(h2)

        x = torch.cat((h1, h2), dim=2)
        z = self.sigmoid_f(self.hidden_att(x))
        fusion_features = z * h1 + (1 - z) * h2

        fusion_probs = self.cls(fusion_features)

        return fusion_probs


