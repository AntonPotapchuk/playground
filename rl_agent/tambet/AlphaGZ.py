from Basic import BasicModel
from keras.layers import Input,Conv2D,Dense,Activation,BatchNormalization,add
from keras.layers.core import Flatten
from keras.models import Model
from keras.utils import to_categorical
import numpy as np

class AlphaGZ(BasicModel):
    def ConvBlock(self,mod):
        mod = Conv2D(filters=256,kernel_size=3,strides=1,padding="same")(mod)
        mod = BatchNormalization()(mod)
        mod= Activation('relu')(mod)
        return mod

    def ResidualBlock(self,mod):
        tmp =mod
        mod = self.ConvBlock(mod)
        mod = Conv2D(filters=256,kernel_size=3,strides=1,padding="same")(mod)
        mod = BatchNormalization()(mod)
        mod = add([mod,tmp])
        mod = Activation('relu')(mod)
        return mod

    def PolicyHead(self,mod):
        mod = self.ConvBlock(mod)
        mod = Flatten()(mod)
        mod = Dense(6,activation='softmax')(mod)
        return mod

    def ValueHead(self,mod):
        mod = Conv2D(filters=1,kernel_size=1,strides=1,padding="same")(mod)
        mod = BatchNormalization()(mod)
        mod = Activation('relu')(mod)
        mod = Flatten()(mod)
        mod = Dense(256)(mod)
        mod = Activation('relu')(mod)
        mod = Dense(1)(mod)
        mod = Activation('tanh')(mod)
        return mod

    def create(self,inshape,outshape,res_blks=1,conv_blks=1):
        self.Y_train_v = self.Y_train[:,1]
        self.Y_test_v = self.Y_test[:,1]
        self.Y_train = to_categorical(self.Y_train[:,0],6)
        self.Y_test = to_categorical(self.Y_test[:,0],6)

        I = Input(shape=inshape)
        mod = I
        for i in range(conv_blks):
            mod = self.ConvBlock(mod)
        for i in range(res_blks):
            mod = self.ResidualBlock(mod)
        policy = self.PolicyHead(mod)
        value = self.ValueHead(mod)
        self.clf = Model(inputs=I,outputs=[policy,value])
        self.clf.compile(loss=['categorical_crossentropy','mse'], optimizer='rmsprop',metrics=['accuracy'])
    def train():
        pass
    def train(self,bs=32,eps=10):
        self.clf.fit(self.X_train,[self.Y_train,self.Y_train_v],batch_size=bs,epochs=eps,validation_data=(self.X_test, [self.Y_test,self.Y_test_v]),shuffle=True)
