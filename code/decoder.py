import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  


class FFN(nn.Module):
    def __init__(self, in_channels, ffn_expand):
        super().__init__()
        self.in_channels = in_channels
        self.ffn_expand = ffn_expand

        self.ffn1 = nn.Linear(self.in_channels, self.ffn_expand)
        self.ffn2 = nn.Linear(self.ffn_expand, self.in_channels)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.in_channels)

    def forward(self, x):
        print(f'\t\t[ FFN ]')

        residual = x 

        ffn1_output = self.relu(self.ffn1(x))
        ffn2_output = self.ffn2(ffn1_output)

        result = ffn2_output + residual
        result = self.layer_norm(result)

        return result


class EncoderDecoderAttention(nn.Module):
    def __init__(self, in_channels, num_head) :
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head

        assert self.in_channels % self.num_head == 0, 'in_channel이 num_head로 나누어 떨어지지 않습니다.'
        self.d_model = self.in_channels // self.num_head

        self.WQ = nn.Linear(self.in_channels, self.in_channels)
        self.WK = nn.Linear(self.in_channels, self.in_channels)
        self.WV = nn.Linear(self.in_channels, self.in_channels)

        self.WO = nn.Linear(self.in_channels, self.in_channels)

        self.layer_norm = nn.LayerNorm(self.in_channels)

    def forward(self, encoder_output, decoder_input) :
        print(f'\t\t[ Encoder - Deocder MultiHead Attention ]')

        B, S, D = decoder_input.size()
        residual = decoder_input

        Q = self.WQ(decoder_input)
        K = self.WK(encoder_output)
        V = self.WV(encoder_output)

        Q = Q.view(B, S, self.num_head, self.d_model).transpose(1,2)
        K = K.view(B, S, self.num_head, self.d_model).transpose(1,2)
        V = V.view(B, S, self.num_head, self.d_model).transpose(1,2)


        attention_score = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(self.d_model)
        attention_score = F.softmax(attention_score, dim = -1)        
        attention_value = torch.matmul(attention_score, V)

        attention_result = attention_value.transpose(1,2).contiguous().view(B,S,D)

        result = self.WO(attention_result)
        
        result = attention_result + residual
        result = self.layer_norm(result)
        
        return result


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_head):
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head

        assert self.in_channels % self.num_head == 0 , 'in_channel이 num_head로 나누어 떨어지지 않습니다.'
        self.d_model = self.in_channels // self.num_head

        self.WQ = nn.Linear(self.in_channels, self.in_channels)
        self.WK = nn.Linear(self.in_channels, self.in_channels)
        self.WV = nn.Linear(self.in_channels, self.in_channels)

        self.WO = nn.Linear(self.in_channels, self.in_channels)

        self.layer_norm = nn.LayerNorm(self.in_channels)

    def forward(self, x):
        print(f'\t\t[ Masked MultiHead Attention ]')

        B, S, D = x.size()
        residual = x
        
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = Q.view(B, S, self.num_head, self.d_model).transpose(1,2)
        K = K.view(B, S, self.num_head, self.d_model).transpose(1,2)
        V = V.view(B, S, self.num_head, self.d_model).transpose(1,2)

        attention_score = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(self.d_model)

        masked = torch.tril(torch.ones(S,S)).unsqueeze(0).unsqueeze(0)
        masked_attention_score = attention_score.masked_fill(masked == 0, float('-inf'))
        masked_attention_score = F.softmax(masked_attention_score, dim = -1)

        attention_value = torch.matmul(masked_attention_score, V)

        attention_result = attention_value.transpose(1,2).contiguous().view(B,S,D)
        attention_result = self.WO(attention_result)
        
        result = attention_result + residual
        result = self.layer_norm(result)
        return result
    

class Decoder(nn.Module):
    def __init__(self, in_channels, num_head, ffn_expand):
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.ffn_expand = ffn_expand

        self.masked_atten = MaskedMultiHeadAttention(self.in_channels, self.num_head)
        self.encoder_decoder_atten = EncoderDecoderAttention(self.in_channels, self.num_head)
        self.feedforward = FFN(self.in_channels, self.ffn_expand)

    def forward(self, encoder_output, decoder_input):
        print(f'\t[ Deocder Block ]')

        masked_attention_output = self.masked_atten(decoder_input)
        encoder_decoder_attention = self.encoder_decoder_atten(encoder_output, masked_attention_output)
        feed_forward = self.feedforward(encoder_decoder_attention)
        return feed_forward
    

class StackedDecoder(nn.Module):
    def __init__(self, in_channels, num_head, ffn_expand, num_layers):
        super().__init__()

        self.decoder = nn.ModuleList(
            Decoder(in_channels, num_head, ffn_expand) for _ in range(num_layers)
        )

    def forward(self, encoder_output, decoder_input):
        print(f'\n\n=====[ Decoder ]=====[')
        for layer in self.decoder:
            decoder_input = layer(encoder_output, decoder_input)
        return decoder_input
