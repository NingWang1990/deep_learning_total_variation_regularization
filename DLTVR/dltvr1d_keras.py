
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
from sklearn.utils import shuffle

def penalty_l1(grad, alpha=1e-3):
    def loss(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true)) + alpha*K.mean(K.abs(grad))
    return loss

def penalty_l2(grad, alpha=1e-3):
    def loss(y_true,y_pred):
        return K.mean(K.square(y_pred - y_true)) + alpha*K.mean(K.square(grad))
    return loss

def build_model(hidden_layers, activation='tanh',alpha=1e-3,penalty_order=3,penalty_loss='l1'):
    inputs = keras.Input(shape=(1,))
    for i,hidden in enumerate(hidden_layers):
        if i == 0:
            h = keras.layers.Dense(hidden,activation='linear', kernel_initializer=keras.initializers.glorot_normal)(inputs)
        else:
            h = keras.layers.Dense(hidden,activation='linear')(h)
        if activation == 'tanh':
            h = K.tanh(h)
        elif activation == 'sine':
            h = K.sin(h)
        elif activation == 'elu':
            h = K.elu(h)
        elif activation == 'sigmoid':
            h = K.sigmoid(h)
        elif activation == 'relu':
            h = K.relu(h)
        #h = keras.layers
        #h = keras.layers.Dropout(rate=0.8)(h)
        #h = keras.layers.BatchNormalization()(h)
    outputs = keras.layers.Dense(1,activation='linear')(h)
    model = keras.Model(inputs, outputs)
    grad1 = K.gradients(model.output, model.input)[0]
    iterate1 = K.function([model.input], [grad1])
    grad2 = K.gradients(grad1, model.input)[0]
    iterate2 = K.function([model.input], [grad2])
    if penalty_order == 2:
        tt = grad2
    elif penalty_order == 3:
        grad3 = K.gradients(grad2, model.input)[0]
        tt = grad3
    if penalty_loss == 'l1':
        model.compile(optimizer='Adam', loss=penalty_l1(tt, alpha=alpha))
    elif penalty_loss == 'l2': 
        model.compile(optimizer='Adam', loss=penalty_l2(tt, alpha=alpha))
    return model,iterate1,iterate2

class DLTVR1D_Keras():
    def __init__(self,hidden_layers= [2,5,8,11,14,17,17,17,17,17,11,8,5,2], 
                 activation='tanh',regularization_para=1e-3,penalty_order=2,penalty_loss='l1'):
        if not penalty_order in [2,3]:
            raise ValueError('penalty_order can only be 2 or 3')
        self.model, self.iterate1, self.iterate2 = build_model(hidden_layers=hidden_layers,
                                                    activation=activation,alpha=regularization_para,
                                                    penalty_order=penalty_order,penalty_loss=penalty_loss)

    def fit(self,x_train,y_train,epochs=10000,batch_size=1000,patience=100):
        x_train = np.expand_dims(np.array(x_train,dtype=np.float32).flatten(),1)
        y_train = np.expand_dims(np.array(y_train,dtype=np.float32).flatten(),1)
        x_train, y_train = shuffle([x_train, y_train],random_state=1001)
        callback_es = keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        self.model.fit(x_train, y_train,epochs=epochs,batch_size=batch_size,callbacks=[callback_es])
   
    #  def check_data(self,x_train,y_train):
    #      if not isinstance(x_train, np.ndarray):
    #           raise TypeError('x_train must be np.ndarray')
    #       elif not isinstance(y_train, np.ndarray):
    #           raise TypeError('y_train must be np.ndarray')
    #       elif not len(x_train.shape) == 2:
    #           raise TypeError('x_train must be 2D array')
    #       elif not len(y_train.shape) == 2:
    #           raise TypeError('y_train must be 2D array')
    #       elif not len(x_train.shape[1]) == 1:
    #           raise TypeError('x_train ')
           
        
    def predict(self,x): 
        x = np.expand_dims(np.array(x,dtype=np.float32).flatten(),1)
        y_pred = self.model.predict(x)
        y_x_pred = self.iterate1(x)[0]
        y_xx_pred = self.iterate2(x)[0]
        return y_pred, y_x_pred, y_xx_pred
        
