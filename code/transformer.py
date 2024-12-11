import torch.nn as nn
import torch
from encoder import StackedEncoder
from decoder import StackedDecoder

class Transformer(nn.Module):
    """Transformer 모델.

    이 클래스는 Transformer 아키텍처를 구현하며, Encoder와 Decoder를 초기화하고 호출하여 결과를 도출합니다.
    """
    def __init__(self, in_channels, num_heads, ffn_expand, num_layers):
        """Encoder와 Decoder 클래스를 초기화합니다.

        Args:
            in_channels (int): 임베딩할 행렬의 차원.
            num_heads (int): Multihead의 개수.
            ffn_expand (int): FFN 시 확장할 차원.
            num_layers (int): Encoder 및 Decoder의 반복 회수.
        """
        super().__init__()

        self.encoder = StackedEncoder(in_channels, num_heads, ffn_expand, num_layers)
        self.decoder = StackedDecoder(in_channels, num_heads, ffn_expand, num_layers)

    def forward(self, source, target):
        """Encoder와 Decoder를 순서대로 호출합니다.

        Args:
            source (Tensor): 임베딩된 입력값.
            target (Tensor): 임베딩된 정답값.

        Returns:
            Tensor: 디코더의 결과인 (B, S, D) 형태의 행렬.
        """
        encoder_output = self.encoder(source)
        decoder_output = self.decoder(encoder_output, target)

        result = decoder_output
    
        return result


if __name__ == '__main__':
    batch_size = 32
    seq_len = 8
    in_channels = 512
    num_heads = 8
    ffn_expand = 2048
    num_layers = 3

    source = torch.randn(batch_size, seq_len, in_channels)
    target = torch.randn(batch_size, seq_len, in_channels)

    transformer = Transformer(in_channels, num_heads, ffn_expand, num_layers)
    result = transformer(source, target)
    print(f'Transformer Result Shape : {result.size()}')