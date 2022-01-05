import argparse as arg
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from model import HRNet

import matplotlib.pyplot as plt

class Trainer(tf.keras.Model):
    def __init__(self, batch_size, epoch):
        super().__init__()

        self.hrnet = HRNet()
        self.hrnet.build(input_shape=(None, 384, 288, 3))
        self.hrnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss=self.JointsMSELoss(),
                        metrics=['accuracy'])
    
        self.loss_criterion = tf.keras.losses.MeanSquaredError()
    
    def JointsMSELoss(self, output, target):
        batch_size = output.shape[0]
        num_joints = output.shape[1]

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            loss += 0.5 * self.loss_criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

    def train(self, train_ds, train_lb, test_ds, test_lb, batch_size, epochs):
        his = self.hrnet.fit(train_ds, train_lb, batch_size=batch_size, epochs=epochs)
        
        outputGraph(his)

        self.hrnet.evaluate(test_ds, test_lb)


def outputGraph(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

def init():
    print("================================================")
    print("Pose Estimation by HRNet Trainer")
    print("")
    print("Developer: himazin331")
    print("(Website: https://himazin331.com)")
    print("================================================")

    # Command line option
    parser = arg.ArgumentParser(description='Pose Estimation by HRNet Trainer')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Specify the mini-batch size. (default: 8)')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Specify the number of times to train. (default: 50)')
    args = parser.parse_args()

    print("Batch Size:", args.batch_size)
    print("Epoch:", args.epoch)

    train = Trainer(batch_size=args.batch_size, epoch=args.epoch)
    

if __name__ == '__main__':
    init()