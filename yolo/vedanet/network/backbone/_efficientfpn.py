from efficientnet_pytorch import EfficientNet
import os
from collections import OrderedDict, Iterable
import logging
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import darknet53 as bdkn

__all__ = ['EfficientFPN']

class EfficientFPN(nn.Module):
    custom_layers = (bdkn.Stage, bdkn.HeadBody, bdkn.Transition,
                    bdkn.Stage.custom_layers, bdkn.HeadBody.custom_layers, bdkn.Transition.custom_layers)
    def __init__(self, model_name):
        input_channels = 32
        super.__init__()
        layer_dict= {'efficientnet-b0': [5,11,16], 'efficientnet-b1': [8,16,23], 'efficientnet-b2': [8,16,23],
            'efficientnet-b3': [8, 18, 26]}

        a, b, c = layer_dict[model_name]

        model_enet = EfficientNet.from_pretrained(model_name)
        stem = nn.Sequential(*list(model_enet.children())[:2])
        blocks = nn.Sequential(*list(model_enet.children())[2])
        self.stage1 = nn.Sequential(*list(blocks.children())[:a])
        self.stage2 = nn.Sequential(*list(blocks.children())[a:b])
        self.stage3 = nn.Sequential(*list(blocks.children())[b:c])
        layer_list = [
            OrderedDict([
                ('head_body_1', bdkn.HeadBody(input_channels*(2**5), first_head=True)),
            ]),


            OrderedDict([
                ('trans_1', bdkn.Transition(input_channels*(2**4))),
            ]),

            OrderedDict([
                ('head_body_2', bdkn.HeadBody(input_channels*(2**4+2**3))),
            ]),

            OrderedDict([
                ('trans_2', bdkn.Transition(input_channels*(2**3))),
            ]),


            OrderedDict([
                ('head_body_3', bdkn.HeadBody(input_channels*(2**3+2**2))),
            ]),
        ]
    def forward(self, x):
        features = []
        stage_4 = self.layers[0](x)
        stage_5 = self.layers[1](stage_4)
        stage_6 = self.layers[2](stage_5)

        head_body_1 = self.layers[3](stage_6)
        trans_1 = self.layers[4](head_body_1)

        concat_2 = torch.cat([trans_1, stage_5], 1)
        head_body_2 =  self.layers[5](concat_2)
        trans_2 = self.layers[6](head_body_2)

        concat_3 = torch.cat([trans_2, stage_4], 1)
        head_body_3 =  self.layers[7](concat_3)

        # stage 6, stage 5, stage 4
        features = [head_body_1, head_body_2, head_body_3]

        return features
