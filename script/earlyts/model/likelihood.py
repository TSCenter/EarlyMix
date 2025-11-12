import torch
import torch.nn as nn
import numpy as np

from earlyts.model.common import Lambda, MyLinear
from earlyts.model.transformer import PositionalEncoder, Transformer



# class LikelihoodClassifier(nn.Module):

#     def __init__(self, in_channels, units, len_ts, strides=1,
#                  heads=4,
#                  drop_rate=0.3,
#                  *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.model = nn.Sequential(
#             Lambda(lambda x: x.permute(0, 2, 1)),
#             nn.Conv1d(in_channels, units, kernel_size=1, stride=1, padding="same"),
#             nn.ReLU(),

#             Lambda(lambda x: x.permute(0, 2, 1)),

#             PositionalEncoder(units, len_ts // strides),
#             nn.Dropout(drop_rate),
#             Transformer(units, len_ts // strides, heads=heads, drop_rate=drop_rate, residual=True),
#             Transformer(units, len_ts // strides, heads=heads, drop_rate=drop_rate, residual=True),
#             MyLinear(units, 1)
#         )

#     def forward(self, inputs):
#         return self.model(inputs)
    

class LikelihoodClassifierDecoder(nn.Module):

    def __init__(self, in_channels, units, len_ts, strides=1,
                 heads=4,
                 drop_rate=0.3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)


        real_len = np.ceil(len_ts / strides).astype(np.int32)
        self.model = nn.Sequential(

            Lambda(lambda x: x.permute(0, 2, 1)),
            nn.Conv1d(in_channels, units, kernel_size=1, stride=1, padding="same"),
            nn.ReLU(),
            Lambda(lambda x: x.permute(0, 2, 1)),

            PositionalEncoder(units, real_len),
            nn.Dropout(drop_rate),
            Transformer(units, units, real_len, heads=heads, drop_rate=drop_rate, residual=True, layer_norm=False, fast_forward=False),
            Transformer(units, units, real_len, heads=heads, drop_rate=drop_rate, residual=True, layer_norm=False, fast_forward=False),
            Transformer(units, 1, real_len, heads=1, drop_rate=0.0, residual=False, layer_norm=False)
            # tf.keras.layers.Dense(1, name="likelihood_fc")
        )



    def forward(self, inputs):
        logits = self.model(inputs)
        return logits



# model = LikelihoodClassifierDecoder(1, 64, 100, strides=20)

# x = torch.randn(2, 5, 1)

# y = model(x)
# print(y)
# print(y.size())


