import torch
import torch.nn as nn
from earlyts.model.common import Lambda, MyConv1d, MyLinear, Residual

class FCN(nn.Module):
    def __init__(self, in_channels, units_list, out_channels, num_cnns=None, kernel=30, stride=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        padding = "same"
        drop_rate = 0.3

        print("stride = {}".format(stride))

        self.permute1 = Lambda(lambda x: torch.permute(x, [0, 2, 1]))
        self.conv1 = MyConv1d(in_channels, units_list[0], kernel, stride=stride, padding="same" if stride == 1 else kernel // 2)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(drop_rate)

        self.residual_blocks1 = nn.ModuleList([
            nn.Sequential(
                Residual(MyConv1d(units_list[0], units_list[0], kernel, padding=padding)),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            ) for _ in range(num_cnns)
        ])

        self.conv2 = MyConv1d(units_list[0], units_list[1], kernel, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(drop_rate)

        self.residual_blocks2 = nn.ModuleList([
            nn.Sequential(
                Residual(MyConv1d(units_list[1], units_list[1], kernel, padding=padding)),
                nn.ReLU(),
                nn.Dropout(drop_rate),
            ) for _ in range(num_cnns)
        ])

        self.permute2 = Lambda(lambda x: torch.permute(x, [0, 2, 1]))
        self.concat = Lambda(lambda x: torch.concat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1))
        self.dropout3 = nn.Dropout(0.5)
        self.fc = MyLinear(units_list[1] * 2, out_channels)

    def forward(self, inputs, labels=None):
        x = self.permute1(inputs)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        for block in self.residual_blocks1:
            x = block(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        for block in self.residual_blocks2:
            x = block(x)

        x = self.permute2(x)  # [B, T', C'] -> [B, T', D']
        x = self.concat(x)    # [B, F]

        if labels is not None:
            unique_labels = labels.unique()
            mixed_x_list = []
            mixed_y_list = []
            for label in unique_labels:
                indices = torch.where(labels == label)[0]
                class_feats = x[indices]  # [N_class, F]
                x_agg = class_feats.mean(dim=0, keepdim=True)  # [1, F]
                mixed_x_list.append(x_agg)
                mixed_y_list.append(label.unsqueeze(0))

            if len(mixed_x_list) > 0:
                mixed_x = torch.cat(mixed_x_list, dim=0) # [num_classes_in_batch, F]
                mixed_y = torch.cat(mixed_y_list, dim=0) # [num_classes_in_batch]

                x = torch.cat([x, mixed_x], dim=0)
                labels = torch.cat([labels, mixed_y], dim=0)

        x = self.dropout3(x)
        x = self.fc(x) # [B + num_classes_in_batch, out_channels]

        return x
