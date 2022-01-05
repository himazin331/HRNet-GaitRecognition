import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tensorflow information message disabled.

MOMENTUM = 0.1

# BasicBlock
class BasicBlock(tf.keras.Model):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels

        self.basicblockLayer = L.Conv2D(self.output_channels, kernel_size=3, stride_size=1, padding='same'),
        
        self.bn = L.BatchNormalization(momentum=MOMENTUM)
        self.relu = L.ReLU()
        self.sum = L.Add()

    def call(self, x):
        for _ in range(4):
            res_x = x
            x = self.relu(self.bn(self.basicblockLayer(x)))
            x = self.bn(self.basicblockLayer(x))
            x = self.relu(self.sum([x, res_x]))

        return x

# Bottleneck
class Bottleneck(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        # Bottleneck Layer-list
        self.bottleneckLayers = [
            L.Conv2D(64, kernel_size=1, stride_size=1, padding='valid'),
            L.Conv2D(64, kernel_size=3, stride_size=1, padding='same'),
            L.Conv2D(256, kernel_size=1, stride_size=1, padding='valid'),
        ]

        # Downsampling Layer
        self.downsampling = L.Conv2D(256, kernel_size=1, stride_size=1, padding='valid'),

        self.bn = L.BatchNormalization(momentum=MOMENTUM)
        self.relu = L.ReLU()
        self.sum = L.Add()

    def call(self, x):
        # Bottleneck with downsampling
        res_x = self.bn(self.downsampling(x))
        for idx, layer in enumerate(self.bottleneckLayers):
            if not (idx == len(self.bottleneckLayers)-1):
                x = self.relu(self.bn(layer(x)))
            else:
                x = self.bn(layer(x))
                x = self.relu(self.sum([x, res_x]))

        # Bottleneck
        for _ in range(3):
            for layer in self.bottleneckLayers:
                x = self.relu(self.bn(layer(x)))

        return x

# HRNet
class HRNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.bn = L.BatchNormalization(momentum=MOMENTUM)
        self.relu = L.ReLU()
        self.sum = L.Add()

        # Pre-Downsampling Layer
        self.preDS = L.Conv2D(64, 3, 2, 'same')

        # Bottleneck Block
        self.bottleneck = Bottleneck()

        # Basic Block
        self.basicblock = []
        for x in range(1, 5):
            self.basicblock.append(BasicBlock(32*x))

        # Transition Block Layer-list
        self.transitionLayers = [
            L.Conv2D(32, 3, 1, 'same'),
            L.Conv2D(64, 3, 2, 'same'),
            L.Conv2D(128, 3, 2, 'same'),
            L.Conv2D(256, 3, 2, 'same')
        ]

        # Fuse Block-list
        self.fuseblock = [
            # 1 to 2
            [
                L.Conv2D(64, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM)
            ],
            # 1 to 3
            [
                L.Conv2D(32, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.ReLU(),
                L.Conv2D(128, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM),
            ],
            # 1 to 4
            [
                L.Conv2D(32, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.ReLU(),
                L.Conv2D(32, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.ReLU(),
                L.Conv2D(256, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM)
            ],
            # 2 to 1
            [
                L.Conv2D(32, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(2)
            ],
            # 2 to 3
            [
                L.Conv2D(128, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM)
            ],
            # 2 to 4
            [
                L.Conv2D(64, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.ReLU(),
                L.Conv2D(256, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM)
            ],
            # 3 to 1
            [
                L.Conv2D(32, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(4)
            ],
            # 3 to 2
            [
                L.Conv2D(64, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(2)
            ],
            # 3 to 4
            [
                L.Conv2D(256, 3, 2, 'same'),
                L.BatchNormalization(momentum=MOMENTUM)
            ],
            # 4 to 1
            [
                L.Conv2D(32, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(8)
            ],
            # 4 to 2
            [
                L.Conv2D(64, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(4)
            ],
            # 4 to 3
            [
                L.Conv2D(128, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(2)
            ]
        ]

        # Output Block-list
        self.outputblock = [
            [
                L.Conv2D(32, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(2)
            ],
            [
                L.Conv2D(32, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(4)
            ],
            [
                L.Conv2D(32, 1, 1, 'valid'),
                L.BatchNormalization(momentum=MOMENTUM),
                L.UpSampling2D(8)
            ]
        ]

        # Last Output Layer
        self.last_output_layer = L.Conv2D(17, 1, 1, 'valid')

    # Forward Block
    def forward_block_func(self, x, block):
        for layer in block:
            x = layer(x)
        return x

    def call(self, x):
        x_list = [x]
        x2x_list = np.zeros((4, 3))
        """
            # fusion matrix
            # x1x2 - x1 to x2
            x2x_list = [
                [x1x2, x1x3, x1x4],
                [x2x1, x2x3, x2x4],
                [x3x1, x3x2, x3x4],
                [x4x1, x4x2, x4x3]
            ]
        """

        #* Stage 1
        # Pre-Downsampling Block
        for _ in range(2):
            x_list[0] = self.relu(self.bn(self.preDS(x_list[0])))

        # Bottleneck Block
        x_list[0] = self.bottleneck(x_list[0])
        x_list.append(x_list[0])

        #* Stage 2
        for idx, x in enumerate(x_list):
            # Transition Block
            x = self.relu(self.bn(self.transitionLayers[idx](x)))

            # BasicBlock
            x = self.basicblock[idx](x)

            # FuseBlock
            x2x_list[idx][0] = x
            x2x_list[idx][0] = self.forward_block_func(x2x_list[idx][0], self.fuseblock[idx*3])
        # Fusion
        x_list[0] = self.relu(self.sum([x_list[0], x2x_list[1][0]]))
        x_list[1] = self.relu(self.sum([x_list[1], x2x_list[0][0]]))
        x_list.append(x_list[1])

        #* Stage 3
        # Transition
        x_list[2] = self.relu(self.bn(self.transitionLayers[2](x_list[2])))

        for idx, x in enumerate(x_list):
            # BasicBlock
            x = self.basicblock[idx](x)

            # FuseBlock
            x2x_list[idx][0], x2x_list[idx][1] = [x]*2
            for j in range(2):
                x2x_list[idx][j] = self.forward_block_func(x2x_list[idx][j], self.fuseblock[idx*3+j])


                
        # Fusion
        x_list[0] = self.relu(self.sum([self.sum([x_list[0], x2x_list[1][0]]), x2x_list[2][0]]))
        x_list[1] = self.relu(self.sum([self.sum([x_list[1], x2x_list[0][0]]), x2x_list[2][1]]))
        x_list[2] = self.relu(self.sum([self.sum([x_list[2], x2x_list[0][1]]), x2x_list[1][1]]))
        x_list.append(x_list[2])

        #* Stage 4
        # Transition
        x_list[3] = self.relu(self.bn(self.transitionLayers[3](x_list[3])))

        for idx, x in enumerate(x_list):
            # BasicBlock
            x = self.basicblock[idx](x)

            x2x_list[idx][0], x2x_list[idx][1], x2x_list[idx][2] = [x]*3

            # FuseBlock
            for j in range(3):
                x2x_list[idx][j] = self.forward_block_func(x2x_list[idx][j], self.fuseblock[idx*3+j])
        # Fusion
        x_list[0] = self.relu(self.sum([self.sum([self.sum([x_list[0], x2x_list[1][0]]), x2x_list[2][0]]), x2x_list[3][0]]))
        x_list[1] = self.relu(self.sum([self.sum([self.sum([x_list[1], x2x_list[0][0]]), x2x_list[2][1]]), x2x_list[3][1]]))
        x_list[2] = self.relu(self.sum([self.sum([self.sum([x_list[2], x2x_list[0][1]]), x2x_list[1][1]]), x2x_list[3][2]]))
        x_list[3] = self.relu(self.sum([self.sum([self.sum([x_list[3], x2x_list[0][2]]), x2x_list[1][2]]), x2x_list[2][2]]))

        for idx, x in enumerate(x_list):
            # Last BasicBlock
            x = self.basicblock[idx](x)

            # Output Block
            x = self.forward_block_func(x, self.outputblock[idx])

        out = self.last_output_layer(self.sum([self.sum([self.sum([x_list[0], x_list[1]]), x_list[2]]), x_list[3]]))

        return out