from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, trainable: bool = False) -> None:
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        for parameter in self.features.parameters():
            parameter.requires_grad = trainable

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        visual = self.pool(self.features(images))
        return self.project(visual)


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        embeddings = self.dropout(self.embed(captions[:, 1:-1]))
        inputs = torch.cat((image_features.unsqueeze(1), embeddings), dim=1)
        hidden, _ = self.lstm(inputs)
        return self.fc(hidden)

    def step(self, token: torch.Tensor, states=None):
        embedding = self.embed(token).unsqueeze(1)
        output, states = self.lstm(embedding, states)
        logits = self.fc(output.squeeze(1))
        return logits, states


class SpatialEncoderCNN(nn.Module):
    def __init__(self, trainable: bool = False) -> None:
        super().__init__()
        backbone = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = backbone.features
        for parameter in self.features.parameters():
            parameter.requires_grad = trainable

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.features(images)
        maps = maps.permute(0, 2, 3, 1)
        return maps.reshape(maps.size(0), -1, maps.size(-1))


class AdditiveAttention(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.feature_att = nn.Linear(feature_size, attention_size)
        self.hidden_att = nn.Linear(hidden_size, attention_size)
        self.full_att = nn.Linear(attention_size, 1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        scores = self.full_att(
            torch.tanh(self.feature_att(features) + self.hidden_att(hidden).unsqueeze(1))
        ).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = (features * alpha.unsqueeze(-1)).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        attention_size: int,
        feature_size: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.attention = AdditiveAttention(feature_size, hidden_size, attention_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.init_h = nn.Linear(feature_size, hidden_size)
        self.init_c = nn.Linear(feature_size, hidden_size)
        self.lstm_cell = nn.LSTMCell(embed_size + feature_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, features: torch.Tensor):
        mean_features = features.mean(dim=1)
        return self.init_h(mean_features), self.init_c(mean_features)

    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        batch_size, seq_len = captions.shape
        hidden, cell = self.init_hidden_state(features)
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        predictions = []
        alphas = []

        for t in range(seq_len - 1):
            context, alpha = self.attention(features, hidden)
            hidden, cell = self.lstm_cell(torch.cat([embeddings[:, t], context], dim=1), (hidden, cell))
            predictions.append(self.fc(self.dropout(hidden)))
            alphas.append(alpha)

        return torch.stack(predictions, dim=1), torch.stack(alphas, dim=1)

    def step(self, features: torch.Tensor, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        context, alpha = self.attention(features, hidden)
        embedding = self.embed(token)
        hidden, cell = self.lstm_cell(torch.cat([embedding, context], dim=1), (hidden, cell))
        logits = self.fc(hidden)
        return logits, hidden, cell, alpha
