import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tensorflow information message disabled.

# FeatureExtractor(VGG16)
class FeatureExtractor(tf.keras.Model):
    def __init__(self, input_shape):
        self.input_tensor = tf.keras.Input(shape=input_shape)
        self.vgg = VGG16(include_top=False, input_tensor=self.input_tensor)

    def get_model(self):
        self.vggmodel = tf.keras.Model(inputs=self.input_tensor, outputs=self.vgg.layers[-2].output)
        return self.vggmodel

# Region Proposal Network
class RPN(tf.keras.Model):
    def __init__(self, image_data, num_anchors):
        super().__init__()
        # FeatureExtractor(VGG16)
        vgg = FeatureExtractor(image_data.shape)
        self.vggmodel = vgg.get_model()

        # RPN layers
        self.conv = L.Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")
        self.rpn_cls = L.Conv2D(num_anchors, (1, 1), activation="sigmoid", name="rpn_cls")    # Classification layer
        self.rpn_reg = L.Conv2D(num_anchors * 4, (1, 1), activation="linear", name="rpn_reg") # Regression layer

    def call(self, image_data):
        # Feature extraction
        feature_map = self.vggmodel(image_data)

        # RPN
        x = self.conv(feature_map)
        cls_output = self.rpn_cls(x) # Classification layer: object or not.
        reg_output = self.rpn_reg(x) # Regression layer: bbox coordinate deviation

        return [cls_output, reg_output]


class RoiPooling():
    def __init__(self, pool_size, num_rois):
        super().__init__()
        self.dim_ordering = K.image_dim_ordering()
        self.pool_size = pool_size
        self.num_rois = num_rois

    def call(self, x):
        # Image with shape(rows, cols, channels)
        img = x[0]
        input_shape = K.shape(img)
        outputs = []

        # ROI with shape(num_rois, 4)
        rois = x[1]

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels) : (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, 3))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output


class classifier():
    def __init__(self, nb_classes):
        super().__init__()

        self.flatten = L.Flatten(name="flatten")
        self.dense = L.Dense(4096, activation="relu", name="fc")
        self.dropout = L.Dropout(0.5)

        self.dense_cls = L.Dense(nb_classes, activation="softmax", kernel_initializer="zero", name="dense_cls") # Dense for classification
        self.dense_reg = L.Dense(4*(nb_classes-1), activation="linear", kernel_initializer="zero", name="dense_reg") # Dense for coordinate regression

    def call(self, x):
        x = self.flatten(x)
        x = self.dropout(self.dense(x))
        x = self.dropout(self.dense(x))

        cls_output = self.dense_cls(x)
        reg_output = self.dense_reg(x)
        
        return [cls_output, reg_output]