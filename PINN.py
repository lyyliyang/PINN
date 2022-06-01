import numpy as np
import tensorflow as tf

def normalization(x,value):
    x1=x-np.mean(x,axis=0)
    # scale=np.max(X1,axis=0)/np.pi
    x2=x1/np.std(x1,axis=0)*value
    
    return x2

def rebuild_date(y,x,Ncs,index):
    y_cs=y
    for i in range(Ncs):
        y_cs=np.hstack([y_cs,np.cos(x[:,index[i]:index[i]+1]),np.sin(x[:,index[i]:index[i]+1])])
    return y_cs

def divide_data(x,y_cs,per_train):
    
    permutation = np.random.choice(range(x.shape[ 0 ]),
    x.shape[ 0 ], replace = False)
    size_train = np.int32(np.round(x.shape[ 0 ] * per_train))
    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]
    
    x_train = x[ index_train, : ]
    y_train = y_cs[ index_train,: ]
    x_test = x[ index_test, : ]
    y_test = y_cs[ index_test,: ]
    
    return x_train,y_train,x_test,y_test

class PINN():
    def __init__(self,PINN_params,sess,function=tf.abs,PINN_scope_name='PINN'):
        self.PINN_params=PINN_params
        self.PINN_scope_name=PINN_scope_name
        self.sess=sess
        self.index=PINN_params['index']
        self.x_dim=PINN_params['x_dim']
        self.y_dim=PINN_params['y_dim']
        self.function=function
        self.Ncs=PINN_params['Ncs']
        self.SC_params=PINN_params['SC_params']
        self.dim=np.concatenate([[self.x_dim],PINN_params['hidden_dim'],[2*(self.y_dim+self.Ncs)]])
        self.Input=tf.placeholder(tf.float32,[None,self.x_dim],'input')
        self.Label=tf.placeholder(tf.float32,[None,self.y_dim+2*self.Ncs],'label')
        
        with tf.variable_scope(self.PINN_scope_name): 
            self.PINN_w=[]
            self.PINN_b=[]
            for i, size in enumerate(self.dim[1:]):
                self.PINN_w.append(tf.get_variable("PINN_w_{}".format(i), [self.dim[i], size],dtype=tf.float32,initializer=tf.random_normal_initializer(0.,.01)))
                self.PINN_b.append(tf.get_variable("PINN_b_{}".format(i), [1, size],dtype=tf.float32,initializer=tf.random_normal_initializer(0., .01)))
        self.PINN_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.PINN_scope_name)       
        
        self.net,self.sigma=self.build_PINN(self.Input)
        self.prediction=self.net[-1]
        self.RMSE=tf.sqrt(tf.losses.mean_squared_error(self.prediction[:,:self.y_dim], 
                                                       self.Label[:,:self.y_dim]))
        
        dist0=tf.contrib.distributions.Normal(loc=self.net[-1][:,:self.x_dim],scale=tf.exp(self.net[-1][:,-self.y_dim:]))
        LL0=-dist0.log_prob(self.Label[:,:self.y_dim])
        loss = tf.reduce_mean(tf.concat([tf.squared_difference(self.net[-1][:,self.y_dim:self.y_dim+self.Ncs*2], 
                                                               self.Label[:,self.y_dim:self.y_dim+self.Ncs*2]),
                                         LL0],axis=1))


        optimizer = tf.train.AdamOptimizer(PINN_params['lr'])
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=PINN_params['clip_norm'])
    
        self.train = optimizer.apply_gradients(zip(grads, tvars))
        

        
        dist=tf.contrib.distributions.Normal(loc=self.prediction[:,:self.y_dim],
                                             scale=self.sigma+function(self.prediction[:,-self.y_dim:]))

        self.LL=tf.reduce_mean(dist.log_prob(self.Label[:,:self.y_dim]))
        
        
    def build_PINN(self,Input):
        PINN_net=[]
        for i in range(len(self.dim)-2):
            if i==0:
                PINN_net.append(tf.nn.relu(tf.matmul(Input, self.PINN_w[i])+self.PINN_b[i],name='PINN_net_{}'.format(i)))
            else:
                PINN_net.append(tf.nn.relu(tf.matmul(PINN_net[i-1], self.PINN_w[i])+self.PINN_b[i],name='PINN_net_{}'.format(i)))
        PINN_net.append((tf.matmul(PINN_net[len(self.dim)-3], self.PINN_w[len(self.dim)-2])+self.PINN_b[len(self.dim)-2]))
        
        
        SC2=0
        for i in range(self.Ncs):
            SC2+=tf.square(tf.cos((Input[:,self.index[i]:self.index[i]+1]))-(PINN_net[-1][:,self.y_dim+i*2:self.y_dim+i*2+1]))\
                            +tf.square(tf.sin((Input[:,self.index[i]:self.index[i]+1]))-(PINN_net[-1][:,self.y_dim+i*2+1:self.y_dim+i*2+2]))

        SC2=tf.reshape(SC2,[-1,1])
        sigma=self.SC_params*tf.sqrt(SC2)

        return PINN_net,sigma
    def PINN_train(self,x_train,y_train):
        _,RSME,LL=self.sess.run([self.train,self.RMSE,self.LL],feed_dict={self.Input:x_train,self.Label:y_train})
        return RSME,LL
    def test(self,x_test,y_test):
        RMSE,LL,m,v=self.sess.run([self.RMSE,self.LL,self.prediction,self.sigma],feed_dict={self.Input:x_test,self.Label:y_test})
        return RMSE,LL,m,v
        
    