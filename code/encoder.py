import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """FeedForward 모듈.

    Encoder에서 사용되는 Feed-Forward Network (FFN)를 구현합니다.
    """
    def __init__(self, in_channels, ffn_expand):
        """FeedForward 모듈을 초기화합니다.

        Args:
            in_channels (int): 입력값의 임베딩 차원.
            ffn_expand (int): 첫 번째 선형 계층에서 확장할 차원.
        """
        super().__init__()
        self.in_channels = in_channels
        self.ffn_expand = ffn_expand

        self.fc_out1 = nn.Linear(self.in_channels, self.ffn_expand)
        self.relu = nn.ReLU()
        self.fc_out2 = nn.Linear(self.ffn_expand, self.in_channels)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        """Feed-Forward Network을 적용합니다.

        Args:
            x (Tensor): Multihead Attention의 결과 (B, S, D).

        Returns:
            Tensor: FFN과 잔차 연결을 거친 후의 출력 (B, S, D).
        """
        print(f'\t\t[ FFN ]')

        residual = x
        out = self.fc_out1(x)
        out = self.relu(out)
        out = self.fc_out2(out)

        out = out + residual
        out = self.layer_norm(out)

        return out


class MultiheadAttention(nn.Module):
    """Multihead Attention 모듈.

    Multi-Head Attention을 구현하며, Q, K, V를 여러 헤드로 분리하여
    각각 독립적으로 어텐션을 수행한 후 결과를 연결(concat)하고 선형 변환을 적용합니다.
    """
    def __init__(self, in_channels, num_head):
        """MultiheadAttention 모듈을 초기화합니다.

        Args:
            in_channels (int): 입력의 임베딩 차원.
            num_head (int): 사용할 헤드의 개수.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head

        assert in_channels % self.num_head == 0, 'in_channels는 num_head로 나누어 떨어져야 합니다.'
        self.d_model = in_channels // self.num_head

        self.WQ = nn.Linear(self.in_channels, self.in_channels)
        self.WK = nn.Linear(self.in_channels, self.in_channels)
        self.WV = nn.Linear(self.in_channels, self.in_channels)

        self.WO = nn.Linear(self.in_channels, self.in_channels)

        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        """Multi-Head Attention을 적용합니다.

        Args:
            x (Tensor): 입력 텐서 (B, S, D).

        Returns:
            Tensor: Multi-Head Attention과 잔차 연결을 거친 출력 (B, S, D).
        """
        print(f'\t\t[ MultiHead Attention ]')

        residual = x
        B, S, D = x.size()

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = Q.view(B, S, self.num_head, self.d_model).transpose(1, 2)  # (B, num_head, S, d_model)
        K = K.view(B, S, self.num_head, self.d_model).transpose(1, 2)  # (B, num_head, S, d_model)
        V = V.view(B, S, self.num_head, self.d_model).transpose(1, 2)  # (B, num_head, S, d_model)

        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model)
        attention_value = torch.matmul(F.softmax(attention_score, dim=-1), V)

        out = attention_value.transpose(1, 2).contiguous().view(B, S, D)  # (B, S, D)

        out = self.WO(out)

        out = out + residual
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    """Encoder 모듈.

    Multi-Head Attention과 Feed-Forward Network을 관리하고 적용합니다.
    """
    def __init__(self, in_channels, num_head, ffn_expand):
        """Encoder 모듈을 초기화합니다.

        Args:
            in_channels (int): 임베딩 차원.
            num_head (int): 헤드의 개수.
            ffn_expand (int): FFN에서 확장할 차원.
        """
        super().__init__()
        self.multihead_atten = MultiheadAttention(in_channels, num_head)
        self.ffn = FeedForward(in_channels, ffn_expand)

    def forward(self, x):
        """Encoder를 Multi-Head Attention과 FFN 순서대로 적용합니다.

        Args:
            x (Tensor): 입력 임베딩 (B, S, D).

        Returns:
            Tensor: Encoder를 통과한 출력 (B, S, D).
        """
        print(f'\t[ Encoder ]')

        attn_output = self.multihead_atten(x)
        ffn_output = self.ffn(attn_output)
        encoder_result = ffn_output
        return encoder_result


class StackedEncoder(nn.Module):
    """StackedEncoder 모듈.

    Encoder를 여러 번 반복하여 깊은 인코딩을 가능하게 합니다.
    """
    def __init__(self, in_channels, num_head, ffn_expand, num_layers=8):
        """StackedEncoder를 초기화합니다.

        Args:
            in_channels (int): 임베딩 차원.
            num_head (int): 헤드의 개수.
            ffn_expand (int): FFN에서 확장할 차원.
            num_layers (int, optional): Encoder 반복 횟수. Defaults to 8.
        """
        super().__init__()
        self.encoder = nn.ModuleList([
            Encoder(in_channels, num_head, ffn_expand) for _ in range(num_layers)
        ])

    def forward(self, x):
        """Encoder를 여러 번 반복하여 적용합니다.

        Args:
            x (Tensor): 입력 행렬 (B, S, D).

        Returns:
            Tensor: 모든 Encoder 층을 통과한 출력 (B, S, D).
        """
        print(f'=====[ Stacked Encoder ]=====')
        for layer in self.encoder:
            x = layer(x)
        return x