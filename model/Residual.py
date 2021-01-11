import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Dense, Flatten, Conv2D,GlobalAveragePooling2D,Activation,Add,Multiply,Lambda,Masking
from tensorflow.keras import Model
k=30
input_shape=(1,k,k,2)
class AttentionBlock(Masking): # A is the mask
    def __init__(self):
        super(AttentionBlock, self).__init__()
        self.avgPool= GlobalAveragePooling2D()
        self.dense=Dense(units=(4))
        self.sigmoid= Activation("sigmoid")
    def call(self, x):
        x=self.avgPool(x)
        x=self.dense(x)
        x=self.sigmoid(x)
        return x
class ResidualBlock(Model): # R(x)*(A(x)+1)
  def __init__(self):
    super(ResidualBlock, self).__init__()
    self.conv1 = Conv2D(4,1,input_shape=input_shape)
    self.conv2 = Conv2D(4,3)
    self.conv3 = Conv2D(4,1)
    self.attentionBlock=AttentionBlock()
    self.plusone=Lambda(lambda a: 1 + a)
    self.scale=Multiply()
    
  def call(self, x):
    res = self.conv1(x)
    res = self.conv2(res)
    res = self.conv3(res)
    x=res
    res= self.attentionBlock.call(res)
    res = self.plusone(res) # mask result: A(x)+1
    x=self.scale([res,x]) #R(x)*(1+A(x))
    return x
class Residual(Model):
  def __init__(self):
    super(Residual, self).__init__()
    self.residualBlock=ResidualBlock()
    self.add=Add()
    
  def call(self, x):
    res=self.residualBlock.call(x)
    x= self.add([res,x])
    return x

# Create an instance of the model
input_shape1=(1,30,30,2)
x=tf.random.normal(input_shape1)

model = Residual()
model.call(x)
print(model)