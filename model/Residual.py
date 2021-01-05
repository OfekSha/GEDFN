import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D,GlobalAveragePooling2D,Activation,add
from tensorflow.keras import Model
class Layer(Model):
   def __init__(self):
    super(Layer, self).__init__()
    self.conv1 = Conv2D(4, 1, activation='relu')
   def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)
class Residual(Model):
  def __init__(self):
    super(Residual, self).__init__()
    self.conv1 = Conv2D(2, 1,input_shape=(4,28,28,3))
    self.conv2 = Conv2D(2, 3,input_shape=(4,28,28,3))
    self.conv3 = Conv2D(2, 1, activation='relu',input_shape=(4,28,28,3))
    self.avgPool= GlobalAveragePooling2D()
    self.sigmoid= Activation("sigmoid")
    
  def call(self, x):
    res = self.conv1(x)
    res = self.conv2(res)
    res = self.conv3(res)
    add((res,x))
    res= self.avgPool(res)
    res=self.sigmoid(res)
    return add((res,x))

# Create an instance of the model
input_shape=(4,28,28,3)
x=tf.random.normal(input_shape)
model = Residual()
model.call(x)
print(model)