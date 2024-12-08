import torch.nn as nn
import torch
from encoder import StackedEncoder
from decoder import StackedDecoder

class Transformer(nn.Module):
    def __init__(self, in_channels, num_heads, ffn_expand, num_layers):
        super().__init__()

        self.encoder = StackedEncoder(in_channels, num_heads, ffn_expand, num_layers)
        self.decoder = StackedDecoder(in_channels, num_heads, ffn_expand, num_layers)

    def forward(self, source, target):
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