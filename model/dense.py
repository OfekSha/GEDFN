from Residual import Residual
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add,BatchNormalization,Conv2D
class DenseModel(Model):
    def add_layer(self,name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = Residual(l)
                c = Conv2D('conv1', c, self.growthRate, 1)
                l = tf.concat([c, l], 3)
            return l
    def __init__(self):
        super(DenseModel, self).__init__()
        
    def call(self, x):
        
        return x

