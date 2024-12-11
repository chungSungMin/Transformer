import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F  


class FFN(nn.Module):
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

        self.ffn1 = nn.Linear(self.in_channels, self.ffn_expand)
        self.ffn2 = nn.Linear(self.ffn_expand, self.in_channels)

        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.in_channels)

    def forward(self, x):
        """Feed-Forward Network을 적용합니다.
        
        Args:
            x (Tensor): Multihead Attention의 결과 (B, S, D).
        
        Returns:
            Tensor: FFN과 잔차 연결을 거친 후의 출력 (B, S, D).
        """
        print(f'\t\t[ FFN ]')

        residual = x 

        ffn1_output = self.relu(self.ffn1(x))
        ffn2_output = self.ffn2(ffn1_output)

        result = ffn2_output + residual
        result = self.layer_norm(result)

        return result


class EncoderDecoderAttention(nn.Module):
    """Encoder-Decoder Attention 모듈.
    
    Encoder의 출력과 Decoder의 입력을 기반으로 Multi-Head Attention을 수행합니다.
    """
    def __init__(self, in_channels, num_head):
        """EncoderDecoderAttention 모듈을 초기화합니다.
        
        Args:
            in_channels (int): 입력의 임베딩 차원.
            num_head (int): 사용할 헤드의 개수.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head

        assert self.in_channels % self.num_head == 0, 'in_channels는 num_head로 나누어 떨어져야 합니다.'
        self.d_model = self.in_channels // self.num_head

        self.WQ = nn.Linear(self.in_channels, self.in_channels)
        self.WK = nn.Linear(self.in_channels, self.in_channels)
        self.WV = nn.Linear(self.in_channels, self.in_channels)

        self.WO = nn.Linear(self.in_channels, self.in_channels)

        self.layer_norm = nn.LayerNorm(self.in_channels)

    def forward(self, encoder_output, decoder_input):
        """Encoder와 Decoder의 출력을 기반으로 Multi-Head Attention을 적용합니다.
        
        Args:
            encoder_output (Tensor): Encoder의 출력 (B, S, D).
            decoder_input (Tensor): Decoder의 입력 (B, S, D).
        
        Returns:
            Tensor: Encoder-Decoder Attention과 잔차 연결을 거친 출력 (B, S, D).
        """
        print(f'\t\t[ Encoder-Decoder MultiHead Attention ]')

        B, S, D = decoder_input.size()
        residual = decoder_input

        Q = self.WQ(decoder_input)
        K = self.WK(encoder_output)
        V = self.WV(encoder_output)

        Q = Q.view(B, S, self.num_head, self.d_model).transpose(1, 2) 
        K = K.view(B, S, self.num_head, self.d_model).transpose(1, 2) 
        V = V.view(B, S, self.num_head, self.d_model).transpose(1, 2) 

        attention_score = torch.matmul(Q, K.transpose(-1,-2)) / np.sqrt(self.d_model)
        attention_score = F.softmax(attention_score, dim=-1)        
        attention_value = torch.matmul(attention_score, V)

        attention_result = attention_value.transpose(1,2).contiguous().view(B, S, D)
        attention_result = self.WO(attention_result)
        
        result = attention_result + residual
        result = self.layer_norm(result)
        
        return result


class MaskedMultiHeadAttention(nn.Module):
    """Masked MultiHead Attention 모듈.
    
    Self-Attention 에서 미래의 토큰을 마스킹하여 Auto Regressive를 유지합니다.
    """
    def __init__(self, in_channels, num_head):
        """MaskedMultiHeadAttention 모듈을 초기화합니다.
        
        Args:
            in_channels (int): 입력의 임베딩 차원.
            num_head (int): 사용할 헤드의 개수.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head

        assert self.in_channels % self.num_head == 0, 'in_channels는 num_head로 나누어 떨어져야 합니다.'
        self.d_model = self.in_channels // self.num_head

        self.WQ = nn.Linear(self.in_channels, self.in_channels)
        self.WK = nn.Linear(self.in_channels, self.in_channels)
        self.WV = nn.Linear(self.in_channels, self.in_channels)

        self.WO = nn.Linear(self.in_channels, self.in_channels)

        self.layer_norm = nn.LayerNorm(self.in_channels)

    def forward(self, x):
        """Masked Multi-Head Attention을 적용합니다.
        
        Args:
            x (Tensor): 입력 텐서 (B, S, D).
        
        Returns:
            Tensor: Masked Multi-Head Attention과 잔차 연결을 거친 출력 (B, S, D).
        """
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

        # 마스킹을 적용하여 미래의 토큰을 무시
        masked = torch.tril(torch.ones(S, S)).unsqueeze(0).unsqueeze(0).to(x.device)  # (1, 1, S, S)
        attention_score = attention_score.masked_fill(masked == 0, float('-inf'))
        attention_score = F.softmax(attention_score, dim=-1)

        attention_value = torch.matmul(attention_score, V)

        attention_result = attention_value.transpose(1,2).contiguous().view(B, S, D)
        attention_result = self.WO(attention_result)
        
        result = attention_result + residual
        result = self.layer_norm(result)
        return result


class Decoder(nn.Module):
    """Decoder 블록 모듈.
    
    Masked Multi-Head Attention, Encoder-Decoder Attention, 그리고 FeedForward 네트워크를 포함합니다.
    """
    def __init__(self, in_channels, num_head, ffn_expand):
        """Decoder 모듈을 초기화합니다.
        
        Args:
            in_channels (int): 임베딩 차원.
            num_head (int): Multi-Head Attention의 헤드 개수.
            ffn_expand (int): FeedForward 네트워크의 확장 차원.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_head = num_head
        self.ffn_expand = ffn_expand

        self.masked_atten = MaskedMultiHeadAttention(self.in_channels, self.num_head)
        self.encoder_decoder_atten = EncoderDecoderAttention(self.in_channels, self.num_head)
        self.feedforward = FFN(self.in_channels, self.ffn_expand)

    def forward(self, encoder_output, decoder_input):
        """Decoder 블록을 순차적으로 적용합니다.
        
        Args:
            encoder_output (Tensor): Encoder의 출력 (B, S, D).
            decoder_input (Tensor): Decoder의 입력 (B, S, D).
        
        Returns:
            Tensor: Decoder를 통과한 출력 (B, S, D).
        """
        print(f'\t[ Decoder Block ]')

        masked_attention_output = self.masked_atten(decoder_input)
        encoder_decoder_attention = self.encoder_decoder_atten(encoder_output, masked_attention_output)
        feed_forward = self.feedforward(encoder_decoder_attention)
        return feed_forward


class StackedDecoder(nn.Module):
    """StackedDecoder 모듈.
    
    여러 개의 Decoder 블록을 쌓아 깊은 디코딩을 가능하게 합니다.
    """
    def __init__(self, in_channels, num_head, ffn_expand, num_layers):
        """StackedDecoder를 초기화합니다.
        
        Args:
            in_channels (int): 임베딩 차원.
            num_head (int): Multi-Head Attention의 헤드 개수.
            ffn_expand (int): FeedForward 네트워크의 확장 차원.
            num_layers (int): Decoder 블록의 반복 횟수.
        """
        super().__init__()

        self.decoder = nn.ModuleList(
            Decoder(in_channels, num_head, ffn_expand) for _ in range(num_layers)
        )

    def forward(self, encoder_output, decoder_input):
        """StackedDecoder를 통해 입력을 순차적으로 처리합니다.
        
        Args:
            encoder_output (Tensor): Encoder의 출력 (B, S, D).
            decoder_input (Tensor): Decoder의 입력 (B, S, D).
        
        Returns:
            Tensor: 모든 Decoder 블록을 통과한 최종 출력 (B, S, D).
        """
        print(f'\n\n=====[ Decoder ]=====[')
        for layer in self.decoder:
            decoder_input = layer(encoder_output, decoder_input)
        return decoder_input