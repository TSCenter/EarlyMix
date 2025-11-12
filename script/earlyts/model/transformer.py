import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from earlyts.model.common import MyLinear


class PositionalEncoder(nn.Module):
    def __init__(self, units, len_ts, pos_rate=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_encoding =nn.parameter.Parameter(PositionalEncoder.positional_encoding(len_ts, units), requires_grad=False)
       
        self.units = units
        self.pos_rate = pos_rate

    @classmethod
    def get_angles(cls, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    @classmethod
    def positional_encoding(cls, position, d_model):
        angle_rads = PositionalEncoder.get_angles(np.arange(position)[:, np.newaxis],
                                             np.arange(d_model)[np.newaxis, :],
                                             d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return torch.tensor(pos_encoding.astype(np.float32))


    def forward(self, inputs):
        # h = inputs * tf.sqrt(tf.cast(self.units, tf.float32)) * 1.0 + self.pos_encoding
        h = inputs * torch.sqrt(torch.tensor(self.units).float()) * self.pos_rate + self.pos_encoding
        # print(tf.reduce_mean(tf.abs(inputs)), tf.reduce_mean(tf.abs(self.pos_encoding)))
        # h = inputs + self.pos_encoding
        return h
    



# class TransformerV1(nn.Module):
#     def __init__(self, units, len_ts,
#                  heads=1,
#                  drop_rate=0.0,
#                  att_drop_rate=0.0,
#                  residual=True,
#                  fast_forward=False,
#                  fast_forward_rate=4.0,
#                  layer_norm=True,
#                  *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # att_drop_rate = 0.1

#         self.residual = residual
#         self.fast_forward = fast_forward
#         self.layer_norm = layer_norm
#         self.att_drop_rate = att_drop_rate

#         self.dropout = nn.Dropout(drop_rate)

#         self.mha = torch.nn.MultiheadAttention(units, heads, batch_first=True)
#         self.attn_mask = nn.parameter.Parameter(torch.triu(torch.ones(len_ts, len_ts).bool(), diagonal=1), requires_grad=False)
        

#         if fast_forward:
#             self.ff = nn.Sequential([
#                 MyLinear(units, units * fast_forward_rate),
#                 nn.ReLU(),
#                 MyLinear(units * fast_forward_rate, units)
#             ])

#         if layer_norm:
#             self.ln = nn.LayerNorm([len_ts, units])
#             self.ff_ln = nn.LayerNorm([len_ts, units])

#         self.heads = heads

#     def forward(self, inputs):

#         if isinstance(inputs, list):
#             queries, keys = inputs
#         else:
#             queries = inputs
#             keys = inputs


#         outputs = self.mha(queries, keys, keys, attn_mask=self.attn_mask)[0]
#         outputs = self.dropout(outputs)

#         if self.residual:
#             outputs = outputs + queries 
#         if self.layer_norm:
#             outputs = self.ln(outputs)

#         if self.fast_forward:
#             ff_outputs = self.ff(outputs)
#             ff_outputs = self.dropout(ff_outputs)
#             outputs = outputs + ff_outputs
#             if self.layer_norm:
#                 outputs = self.ff_ln(outputs)

#         return outputs


class Transformer(nn.Module):
    def __init__(self, in_channels, units, len_ts,
                 heads=1,
                 drop_rate=0.0,
                 att_drop_rate=0.0,
                 residual=True,
                 fast_forward=False,
                 fast_forward_rate=4.0,
                 layer_norm=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # att_drop_rate = 0.1

        self.units = units
        self.residual = residual
        self.fast_forward = fast_forward
        self.layer_norm = layer_norm
        self.att_drop_rate = att_drop_rate

        self.dropout = nn.Dropout(drop_rate)
        self.att_dropout = nn.Dropout(att_drop_rate)

        self.dense_key = MyLinear(in_channels, units)
        self.dense_query = MyLinear(in_channels, units)
        self.dense_value = MyLinear(in_channels, units)

        if fast_forward:
            self.ff = nn.Sequential(
                MyLinear(units, units * fast_forward_rate),
                nn.ReLU(),
                MyLinear(units * fast_forward_rate, units)
            )

        self.tri = nn.parameter.Parameter(torch.tril(torch.ones(len_ts, len_ts).bool()), requires_grad=False)

        if layer_norm:
            self.ln = nn.LayerNorm([len_ts, units])
            self.ff_ln = nn.LayerNorm([len_ts, units])

        self.heads = heads

    def forward(self, inputs):

        if isinstance(inputs, list):
            queries, keys = inputs
        else:
            queries = inputs
            keys = inputs

        # Q = self.dense_query(tf.reduce_mean(queries, axis=-2, keepdims=True))

        Q = self.dense_query(queries)
        K = self.dense_key(keys)
        V = self.dense_value(keys)

        if self.heads > 1:
            
            Q_ = torch.concat(torch.split(Q, self.units // self.heads, dim=-1), dim=0)
            K_ = torch.concat(torch.split(K, self.units // self.heads, dim=-1), dim=0)
            V_ = torch.concat(torch.split(V, self.units // self.heads, dim=-1), dim=0)
        else:
            Q_ = Q
            K_ = K
            V_ = V

        sim_matrix = Q_ @ torch.permute(K_, [0, 2, 1]) / torch.sqrt(torch.tensor(Q_.size(-1)).float())

        mask = torch.tile(self.tri.unsqueeze(dim=0), [sim_matrix.size(0), 1, 1])
        sim_matrix = torch.where(mask, sim_matrix, torch.ones_like(mask, dtype=torch.float32) * -1e9)
        sim_matrix = F.softmax(sim_matrix, dim=-1)
        # sim_matrix = tf.where(mask, sim_matrix, tf.zeros_like(mask, dtype=tf.float32))
        # print(sim_matrix)
        # if training and self.att_drop_rate > 0.0:
        sim_matrix = self.att_dropout(sim_matrix)

        outputs_ = sim_matrix @ V_
        if self.heads > 1:
            outputs = torch.concat(torch.split(outputs_, outputs_.size(0) // self.heads, dim=0), dim=-1)
        else:
            outputs = outputs_
        outputs = self.dropout(outputs)

        if self.residual:
            outputs = outputs + queries 
        if self.layer_norm:
            outputs = self.ln(outputs)

        if self.fast_forward:
            ff_outputs = self.ff(outputs)
            ff_outputs = self.dropout(ff_outputs)
            outputs = outputs + ff_outputs
            if self.layer_norm:
                outputs = self.ff_ln(outputs)

        return outputs


# x = torch.randn(2, 50, 64)

# layer = Transformer(64, 64, 50, heads=4, residual=True)

# y = layer(x)
# print(y)
