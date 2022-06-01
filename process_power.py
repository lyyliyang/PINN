import os
import pandas as pd
from utils import loc_data
import pandas as pd
from utils import loc_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


RMSE_A=[]
LL_A=[]
import os
import time
RSME_file=open('./data/'+os.path.basename(__file__)[:-3]+'_RMSE.txt',mode='w+')
PLL_file=open('./data/'+os.path.basename(__file__)[:-3]+'_PLL.txt',mode='w+')
time_file=open('./data/'+os.path.basename(__file__)[:-3]+'_time.txt',mode='w+')
for _ in range(20):

    tf.reset_default_graph()
    
    df = pd.read_excel(loc_data("Combined Cycle Power Plant/CCPP/Folds5x2_pp.ods"))
    
    
    
    data=np.array(df)
    
    x_dim=4
    y_dim=1
    Ncs=3

    x = data[:,:x_dim]

    y = data[:,-y_dim:]-np.mean(data[:,-y_dim:],axis=0)
    
    y_cs=y/np.std(y)
    
    x1=x-np.mean(x,axis=0)

    x2=x1/np.std(x1,axis=0)*np.pi/2
    index=[0,1,2,3,4,5,6,7]
    x=x2
    for i in range(Ncs):
        y_cs=np.hstack([y_cs,np.cos(x2[:,index[i]:index[i]+1]),np.sin(x2[:,index[i]:index[i]+1])])
        
    hy_par=np.sqrt(np.var(y_cs,axis=0)[:y_dim]/np.sum(np.var(y_cs,axis=0)[y_dim:]))
    permutation = np.random.choice(range(x.shape[ 0 ]),
    x.shape[ 0 ], replace = False)
    size_train = np.int32(np.round(x.shape[ 0 ] * 0.9))
    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]
    x_train = x[ index_train, : ]
    y_train = y_cs[ index_train,: ]
    x_test = x[ index_test, : ]
    y_test = y_cs[ index_test,: ]        
    start_time=time.perf_counter()
    s=tf.placeholder(tf.float32,[None,x_dim],'s')
    label=tf.placeholder(tf.float32,[None,y_dim+Ncs*2],'label')

    
    w1 = tf.get_variable('w1_s', [x_dim, 1000],initializer=tf.random_normal_initializer(0., .1/np.sqrt(x_dim)))
    b1 = tf.get_variable('b1', [1, 1000],initializer=tf.random_normal_initializer(0., .1/np.sqrt(x_dim)))
    w2 = tf.get_variable('w2_s', [1000, 1000],initializer=tf.random_normal_initializer(0., .1/np.sqrt(1000)))  
    b2 = tf.get_variable('b2', [1, 1000],initializer=tf.random_normal_initializer(0.,.1/np.sqrt(1000)))
    w3 = tf.get_variable('w3_s', [1000, y_dim+Ncs*2+y_dim],initializer=tf.random_normal_initializer(0., .1/np.sqrt(1000)))  
    b3 = tf.get_variable('b3', [1, y_dim+Ncs*2+y_dim],initializer=tf.random_normal_initializer(0.,.1/np.sqrt(1000)))
    
    net1 = tf.nn.relu(tf.matmul(s, w1) + b1,name='net1')
    
    net2 = tf.nn.relu(tf.matmul(net1, w2) + b2,name='net2')
    
    net3= tf.matmul(net2, w3)+b3

    
    log_sigma=0
    
    s1=s
    
    for i in range(Ncs):
        
        log_sigma=log_sigma+tf.square(tf.cos((s1[:,index[i]:index[i]+1]))-(net3[:,y_dim+i*2:y_dim+i*2+1]))+tf.square(tf.sin((s1[:,index[i]:index[i]+1]))-(net3[:,y_dim+i*2+1:y_dim+i*2+2]))
            
    
    
    
    log_sigma=tf.reshape(log_sigma,[-1,1])

    sigma=hy_par*tf.sqrt(log_sigma)

    dist0=tf.contrib.distributions.Normal(loc=net3[:,:y_dim],scale=tf.exp(net3[:,-y_dim:]))
    LL0=-tf.log(dist0.prob(label[:,0:y_dim])+1e-6)
    loss = tf.reduce_mean(tf.concat([tf.squared_difference(net3[:,y_dim:y_dim+Ncs*2], label[:,y_dim:y_dim+Ncs*2]),LL0],axis=1))

                                     
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.001,global_step,2000,1)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=10)


    
    train = optimizer.apply_gradients(zip(grads, tvars))
    
    dist=tf.contrib.distributions.Normal(loc=net3[:,0:y_dim],scale=sigma+tf.exp(net3[:,-y_dim:]))
    LL=tf.log(dist.prob(label[:,:y_dim])/np.std(y)+1e-6)
    diff=tf.squared_difference(net3[:,:y_dim], label[:,:y_dim])*np.var(y)
    
    
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    
    
    l_all=[]
    index=np.int32(np.linspace(0,len(x_train)-1,len(x_train)))
    train_size=np.int32(len(x_train)/4)
    for i in range(500):
        np.random.shuffle(index)
        _,l,yy,var,ls,log_p,dy=sess.run([train,loss,net3,sigma,log_sigma,LL,diff],feed_dict={s:x_train[index[:train_size]],label:y_train[index[:train_size]]})
        l_all.append(l)
        if i%5==0:
            print('\r',end='')
            print('\rcur step',i,'loss',l,np.sqrt(np.mean(dy)),np.mean(log_p),end='')
    _,l,yy,var,ls,log_p,dy=sess.run([train,loss,net3,sigma,log_sigma,LL,diff],feed_dict={s:x_train,label:y_train})
    end_time=time.perf_counter()
    var1=var
    plt.figure(0)
    plt.plot(l_all)
    
    for i in range(y_dim):
        plt.figure(i+1)
        plt.plot(yy[:,i])
        plt.plot(y_train[:,i])
        plt.fill(np.concatenate([np.linspace(0,len(x_train),len(x_train)), np.linspace(len(x_train),0,len(x_train))]),
        np.concatenate([yy[:,i]+1.96*var1[:,i],
                                (yy[:,i]-1.96*var1[:,i])[::-1]]),
                  alpha=.5, fc='b', ec='None', label='95% confidence interval')
        
    yy,var,ls,log_p,dy=sess.run([net3,sigma,log_sigma,LL,diff],feed_dict={s:x_test,label:y_test})
        
    print('\r',np.sqrt(np.mean(dy)))
    print(np.mean(log_p))
    RMSE_A.append(np.sqrt(np.mean(dy)))
    LL_A.append(np.mean(log_p))
    RSME_file.write(str(np.sqrt(np.mean(dy)))+'\n')
    PLL_file.write(str(np.mean(log_p))+'\n')
    time_file.write(str(end_time-start_time)+'\n')
RSME_file.close()
PLL_file.close()
time_file.close()
print(np.mean(RMSE_A),'±',np.std(RMSE_A))
print(np.mean(LL_A),'±',np.std(LL_A))
    
    