"""
@author: Ning Wang
adpted from https://github.com/maziarraissi/PINNs/blob/master/main/
continuous_time_identification%20(Navier-Stokes)/NavierStokes.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
np.random.seed(1234)
#tf.random.set_seed(1234)
tf.set_random_seed(1234) #for an earlier version

class DLTVR1D:
    # Initialize the class
    def __init__(self, x, u, hidden_layers=[20,20,20,20,20,20], alpha=0.1,loss='l1l2', 
                keep_prob = 1.,batch_norm=False,
                log_file=None, penalty_order=2, use_LBFGS=False,
                activation='tanh'):
        """
        log_file..............None or string. If None, no log to write out
                                              If string, write out log into this file.
        use_LBFGS.............Boolean. If True, run LBFGS optimizer after Adam optimizer
        loss..................str, l1 or l2
        penalty_order.........int, to specify which order the penalty is applied to.
        
        """
        
        layers = [1,] + hidden_layers + [1,]
        self.x = x
        self.u = u
        X = self.x
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.log_file = log_file
        self.layers = layers
        self.keep_prob = keep_prob
        self.batch_norm = batch_norm 
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        if not loss in ['l1','l2','l1l1', 'l2l2','l1l2']:
            raise ValueError('loss can only be l1, l1l1, l2l2 or l1l2')
        if not activation in ['tanh','sine']:
            raise ValueError('activation can only be tanh or sine')
        self.activation = activation
        self.loss_name = loss
        if not penalty_order in [1,2,3]:
            raise Valuerror('penalty order can only be 1, 2, or 3')
        self.penalty_order = penalty_order
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)        
        # Initialize parameters
        self.alpha = tf.Variable([alpha], dtype=tf.float32,trainable=False)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        # model
        self.u_pred = self.neural_net(self.x_tf, self.weights, self.biases)
        self.build_grads(max_order=self.penalty_order)
        
        if self.loss_name == 'l2l2':
            if self.penalty_order == 1:
                self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) + \
                            self.alpha*tf.reduce_mean(tf.square(self.u_x_pred))
            elif self.penalty_order == 2:
                self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) + \
                            self.alpha*(tf.reduce_mean(tf.square(self.u_xx_pred)) )
            elif self.penalty_order == 3:
                self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                            self.alpha*(tf.reduce_mean(tf.square(self.u_xxx_pred)) )
        elif self.loss_name == 'l1l1':
            if self.penalty_order == 1:
                self.loss = tf.reduce_mean( tf.abs(self.u_tf - self.u_pred) ) + \
                            self.alpha*(tf.reduce_mean(tf.abs(self.u_x_pred))  )
            elif self.penalty_order == 2:
                self.loss = tf.reduce_mean( tf.abs(self.u_tf - self.u_pred) ) + \
                            self.alpha*( tf.reduce_mean(tf.abs(self.u_xx_pred)) )
            elif self.penalty_order == 3:
                self.loss = tf.reduce_mean(tf.abs(self.u_tf - self.u_pred)) + \
                            self.alpha*(tf.reduce_mean(tf.abs(self.u_xxx_pred)) )
        elif self.loss_name == 'l1l2':
            if self.penalty_order == 1:
                self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) + \
                            self.alpha*(tf.reduce_mean(tf.abs(self.u_x_pred))  )
            elif self.penalty_order == 2:
                self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) + \
                            self.alpha*( tf.reduce_mean(tf.abs(self.u_xx_pred)) )
            elif self.penalty_order == 3:
                self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                            self.alpha*(tf.reduce_mean(tf.abs(self.u_xxx_pred)) )
        elif self.loss_name == 'l1':
            self.loss = tf.reduce_mean( tf.abs(self.u_tf - self.u_pred) ) 
        elif self.loss_name == 'l2':
            self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) 


        self.error_sum = tf.reduce_sum(tf.abs(self.u_tf - self.u_pred))

        self.u_absolute_mean = np.mean(np.abs(self.u))
        self.use_LBFGS = use_LBFGS
        if use_LBFGS is True:
            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            if self.activation == 'sine':
                W = self.siren_init(size=[layers[l], layers[l+1]])
            else:
                W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def siren_init(self,size):
        # initialization in https://arxiv.org/pdf/2006.09661.pdf
        in_dim  = size[0]
        out_dim = size[1]
        return tf.Variable(tf.random.uniform([in_dim, out_dim], minval=-(6./in_dim)**0.5, maxval=(6./in_dim)**0.5), dtype=tf.float32)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
            if self.activation == 'tanh':
                H = tf.tanh(H)
            elif self.activation == 'sine':
                # wo=30 suggested in https://arxiv.org/pdf/2006.09661.pdf
                H =tf.sin(30.*H)
            if self.batch_norm == True:
                # Batch Normalize
                fc_mean, fc_var = tf.nn.moments(
                    H,
                    axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
                )
                scale = tf.Variable(tf.ones([self.layers[l+1]]))
                shift = tf.Variable(tf.zeros([self.layers[l+1]]))
                epsilon = 0.001
 
                # apply moving average for mean and var when train on batch
                ema = tf.train.ExponentialMovingAverage(decay=0.5)
                def mean_var_with_update():
                    ema_apply_op = ema.apply([fc_mean, fc_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(fc_mean), tf.identity(fc_var)
                mean, var = tf.cond(self.is_training,  
                    mean_var_with_update,
                    lambda: (
                        ema.average(fc_mean), 
                        ema.average(fc_var)
                        )    
                    )   
                H = tf.nn.batch_normalization(H, mean, var, shift, scale, epsilon)
        
            # dropout
            if self.keep_prob < 1.:
                H = tf.nn.dropout(H, rate=self.keep_prob)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def build_grads(self,max_order=3):
        if max_order>= 1:
            self.u_x_pred   = tf.gradients(self.u_pred,   self.x_tf)[0]
        if max_order >= 2:
            self.u_xx_pred  = tf.gradients(self.u_x_pred, self.x_tf)[0]
        if max_order >= 3:
            self.u_xxx_pred = tf.gradients(self.u_xx_pred,self.x_tf)[0]

    def callback(self, loss):
        print('Loss: %.3e' % (loss))
    
    def train(self, nIter=50000,minibatch_size=100000,patience=100): 
        """
        patience...............int, patience for early stopping
        """
        start_time = time.time()
        n_batches  = int(np.ceil(len(self.x)/minibatch_size))
        best_loss = np.inf
        best_step = 0
        best_weights = None
        best_biases = None
        for it in range(nIter):
            loss_sum = 0.
            e_sum = 0.
            for i in range(n_batches):
                start = i*minibatch_size
                if i == n_batches-1:
                    end = len(self.x)
                else:
                    end = (i+1)*minibatch_size
                tf_dict = {self.x_tf: self.x[start:end], self.u_tf:self.u[start:end],self.is_training:True}
                self.sess.run(self.train_op_Adam, tf_dict)
            
            for i in range(n_batches):
                start = i*minibatch_size
                if i == n_batches-1:
                    end = len(self.x)
                else:
                    end = (i+1)*minibatch_size
                tf_dict = {self.x_tf: self.x[start:end], self.u_tf:self.u[start:end],self.is_training:False}
                loss_value, ae  = self.sess.run([self.loss, self.error_sum], tf_dict)
                loss_sum += loss_value*(end-start)
                e_sum += ae
            mean_loss = loss_sum / len(self.x)
            mean_ae = e_sum / len(self.x)
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Relative error: %.3e, Time: %.2f' % 
                      (it, mean_loss, mean_ae/self.u_absolute_mean, elapsed))
                if not self.log_file == None:
                    outf = open(self.log_file, 'a')
                    outf.write(('It: %d, Loss: %.3e, Relative error: %.3e, Time: %.2f\n' % 
                      (it, mean_loss, mean_ae/self.u_absolute_mean, elapsed)))
                    outf.close()
                start_time = time.time()
            # early stopping
            if mean_loss < best_loss:
                #print ('checkpoint at epoch %d' % it)
                best_step = it
                best_loss = mean_loss
                best_weights = self.sess.run(self.weights)
                best_biases = self.sess.run(self.biases)
            if (it-best_step) > patience:
                print ('early stopping at %d' % it) 
                break
        # restore best weights and bias
        for i in range(len(best_weights)):
            assign_op1 = self.weights[i].assign(best_weights[i])
            assign_op2 = self.biases[i].assign(best_biases[i])
            self.sess.run([assign_op1,assign_op2])

        if self.use_LBFGS is True:
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss,],
                                 loss_callback = self.callback)
            
    def predict(self, x_star):
        x = np.expand_dims(np.array(x_star,dtype=np.float32).flatten(),1) 
        tf_dict = {self.x_tf: x,self.is_training:False}
        if not (hasattr(self, 'u_pred') and hasattr(self,'u_x_pred')  and hasattr(self,'u_xx_pred')):
            self.build_grads(max_order=2)
        u_star = self.sess.run(self.u_pred, tf_dict)
        u_x_star = self.sess.run(self.u_x_pred, tf_dict)
        u_xx_star = self.sess.run(self.u_xx_pred, tf_dict)
        return u_star, u_x_star, u_xx_star
    
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self,x):
        self._x = np.expand_dims(np.array(x,dtype=np.float32).flatten(),1)
    
    @property
    def u(self):
        return self._u
    @u.setter
    def u(self,u):
        self._u = np.expand_dims(np.array(u,dtype=np.float32).flatten(),1)

#      
#    N_train = 5000
#    
#    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
#    
    
#    # Training
#    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
#    model.train(200000)
    
