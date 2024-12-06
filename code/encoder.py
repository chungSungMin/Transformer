import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, in_channels, ffn_expand) :
        super().__init__()
        self.in_channels = in_channels
        self.ffn_expand = ffn_expand

        self.fc_out1 = nn.Linear(self.in_channels, self.ffn_expand)
        self.relu = nn.ReLU()
        self.fc_out2 = nn.Linear(self.ffn_expand, self.in_channels)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x) :
        ffn = x
        out = self.fc_out1(x)
        out = self.relu(out)
        out = self.fc_out2(out)

        out = out + ffn
        out = self.layer_norm(out)

        return out


class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, num_head):
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head

        assert in_channels % self.num_head == 0, 'inchannel이 num_head로 나누어 떨어지지 않습니다.'
        self.d_model = in_channels // self.num_head

        self.WQ = nn.Linear(self.in_channels, self.in_channels)
        self.WK = nn.Linear(self.in_channels, self.in_channels)
        self.WV = nn.Linear(self.in_channels, self.in_channels)

        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, S, D = x.size()

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = Q.view(B, S, self.num_head, self.d_model).transpose(1,2) 
        K = K.view(B, S, self.num_head, self.d_model).transpose(1,2)
        V = V.view(B, S, self.num_head, self.d_model).transpose(1,2)

        attention_score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model)
        attention_value = torch.matmul(F.softmax(attention_score, dim = -1), V)

        out = attention_value.transpose(1,2).contiguous().view(B,S,D)
        out = out + x
        out = self.layer_norm(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self, in_channels, num_head, ffn_expand):
        super().__init__()
        self.multihead_atten = MultiheadAttention(in_channels, num_head)
        self.ffn = FeedForward(in_channels, ffn_expand)

    def forward(self,x):
        attn_output = self.multihead_atten(x)
        ffn_output = self.ffn(attn_output)
        return ffn_output


if __name__ == '__main__':
    batch_size = 32
    seq_len = 8
    in_channels = 512
    num_heads = 8
    ffn_expand = 2048
    
    encoder = Encoder(in_channels, num_heads, ffn_expand)
    x = torch.randn(batch_size, seq_len, in_channels)
    output = encoder(x)
    print(output.shape)  # 출력: (32, 8, 512)
