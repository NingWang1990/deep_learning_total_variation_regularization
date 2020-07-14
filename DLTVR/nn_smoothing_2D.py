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
tf.set_random_seed(1234)

class NNSmoothing2D:
    # Initialize the class
    def __init__(self, x, y, t, u, hidden_layers=[20,20,20,20,20,20], alpha=0.1,loss='l1', log_file=None, penalty_order=2, use_LBFGS=False):
        """
        log_file..............None or string. If None, no log to write out
                                              If string, write out log into this file.
        use_LBFGS.............Boolean. If True, run LBFGS optimizer after Adam optimizer
        loss..................str, l1 or l2
        penalty_order.........int, to specify which order the penalty is applied to.
        """
        
        layers = [3,] + hidden_layers + [1,]
        self.x = x
        self.y = y
        self.t = t
        self.u = u
        X = np.concatenate([self.x, self.y,self.t], 1)
        self.lb = X.min(0)
        self.ub = X.max(0)
        self.log_file = log_file
        self.layers = layers
        if not loss in ['l1', 'l2']:
            raise ValueError('loss can only be l1 or l2')
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
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        # model
        self.u_pred = self.neural_net(tf.concat([self.x_tf,self.y_tf,self.t_tf], 1), self.weights, self.biases)
        self.build_grads(max_order=self.penalty_order)
        
        if self.loss_name == 'l2':
            if self.penalty_order == 1:
                self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) + \
                            self.alpha*( tf.reduce_mean(tf.square(self.u_t_pred)) + \
                            tf.reduce_mean(tf.square(self.u_x_pred)) +  \
                            tf.reduce_mean(tf.square(self.u_y_pred)) )
            elif self.penalty_order == 2:
                self.loss = tf.reduce_mean( tf.square(self.u_tf - self.u_pred) ) + \
                            self.alpha*( tf.reduce_mean(tf.square(self.u_tt_pred)) + \
                            tf.reduce_mean(tf.square(self.u_xx_pred)) +  \
                            tf.reduce_mean(tf.square(self.u_xy_pred)) + \
                            tf.reduce_mean(tf.square(self.u_yy_pred)) )
            elif self.penalty_order == 3:
                self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                            self.alpha*(tf.reduce_mean(tf.square(self.u_tt_pred)) + \
                            tf.reduce_mean(tf.square(self.u_xxx_pred)) +  \
                            tf.reduce_mean(tf.square(self.u_xxy_pred)) + \
                            tf.reduce_mean(tf.square(self.u_xyy_pred)) + \
                            tf.reduce_mean(tf.square(self.u_yyy_pred)) )
        elif self.loss_name == 'l1':
            if self.penalty_order == 1:
                self.loss = tf.reduce_mean( tf.abs(self.u_tf - self.u_pred) ) + \
                            self.alpha*( tf.reduce_mean(tf.abs(self.u_t_pred)) + \
                            tf.reduce_mean(tf.abs(self.u_x_pred)) +  \
                            tf.reduce_mean(tf.abs(self.u_y_pred)) )
            elif self.penalty_order == 2:
                self.loss = tf.reduce_mean( tf.abs(self.u_tf - self.u_pred) ) + \
                            self.alpha*( tf.reduce_mean(tf.abs(self.u_tt_pred)) + \
                            tf.reduce_mean(tf.abs(self.u_xx_pred)) +  \
                            tf.reduce_mean(tf.abs(self.u_xy_pred)) + \
                            tf.reduce_mean(tf.abs(self.u_yy_pred)) )
            elif self.penalty_order == 3: 
                self.loss = tf.reduce_mean(tf.abs(self.u_tf - self.u_pred)) + \
                            self.alpha*(tf.reduce_mean(tf.abs(self.u_tt_pred)) + \
                            tf.reduce_mean(tf.abs(self.u_xxx_pred)) +  \
                            tf.reduce_mean(tf.abs(self.u_xxy_pred)) + \
                            tf.reduce_mean(tf.abs(self.u_xyy_pred)) + \
                            tf.reduce_mean(tf.abs(self.u_yyy_pred)) )
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
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def build_grads(self,max_order=3):
        if max_order>= 1:
            self.u_t_pred   = tf.gradients(self.u_pred,   self.t_tf)[0]
            self.u_x_pred   = tf.gradients(self.u_pred,   self.x_tf)[0]
            self.u_y_pred   = tf.gradients(self.u_pred,   self.y_tf)[0]
        if max_order >= 2:
            self.u_tt_pred  = tf.gradients(self.u_t_pred, self.t_tf)[0]
            self.u_xx_pred  = tf.gradients(self.u_x_pred, self.x_tf)[0]
            self.u_xy_pred  = tf.gradients(self.u_x_pred, self.y_tf)[0]
            self.u_yy_pred  = tf.gradients(self.u_y_pred, self.y_tf)[0]
        if max_order >= 3:
            self.u_xxx_pred = tf.gradients(self.u_xx_pred,self.x_tf)[0]
            self.u_xxy_pred = tf.gradients(self.u_xx_pred,self.y_tf)[0]
            self.u_xyy_pred = tf.gradients(self.u_xy_pred,self.y_tf)[0]
            self.u_yyy_pred = tf.gradients(self.u_yy_pred,self.y_tf)[0]

    def callback(self, loss):
        print('Loss: %.3e' % (loss))
    
    def train(self, nIter=50000,minibatch_size=100000,patience=100): 
        """
        patience...............int, patience for early stopping
        """
        start_time = time.time()
        n_batches  = int(np.round(len(self.x)/minibatch_size))
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
                tf_dict = {self.x_tf: self.x[start:end], self.y_tf:self.y[start:end], self.t_tf:self.t[start:end], self.u_tf:self.u[start:end]}
                self.sess.run(self.train_op_Adam, tf_dict)
            for i in range(n_batches):
                start = i*minibatch_size
                if i == n_batches-1:
                    end = len(self.x)
                else:
                    end = (i+1)*minibatch_size
                tf_dict = {self.x_tf: self.x[start:end], self.y_tf:self.y[start:end], self.t_tf:self.t[start:end], self.u_tf:self.u[start:end]}
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
            
    def predict(self, x_star, y_star, t_star):
        x = np.expand_dims(np.array(x_star,dtype=np.float32).flatten(),1) 
        y = np.expand_dims(np.array(y_star,dtype=np.float32).flatten(),1) 
        t = np.expand_dims(np.array(t_star,dtype=np.float32).flatten(),1) 
        tf_dict = {self.x_tf: x, self.y_tf: y, self.t_tf: t}
        if not (hasattr(self, 'u_pred') and hasattr(self,'u_t_pred') and hasattr(self,'u_x_pred') and \
                hasattr(self, 'u_y_pred') and hasattr(self,'u_xx_pred') and hasattr(self,'u_xy_pred') and \
                hasattr(self, 'u_yy_pred')):
            self.build_grads(max_order=2)
        u_star = self.sess.run(self.u_pred, tf_dict)
        u_t_star = self.sess.run(self.u_t_pred, tf_dict)
        u_x_star = self.sess.run(self.u_x_pred, tf_dict)
        u_y_star = self.sess.run(self.u_y_pred, tf_dict)
        u_xx_star = self.sess.run(self.u_xx_pred, tf_dict)
        u_xy_star = self.sess.run(self.u_xy_pred, tf_dict)
        u_yy_star = self.sess.run(self.u_yy_pred, tf_dict)

        return u_star, u_t_star, u_x_star,u_y_star, u_xx_star,u_xy_star,u_yy_star
    
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self,x):
        self._x = np.expand_dims(np.array(x,dtype=np.float32).flatten(),1)
    
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self,y):
        self._y = np.expand_dims(np.array(y,dtype=np.float32).flatten(),1)

    @property
    def t(self):
        return self._t
    @t.setter
    def t(self,t):
        self._t = np.expand_dims(np.array(t,dtype=np.float32).flatten(),1)

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
    
