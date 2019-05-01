# -*- coding: utf-8 -*-
"""
Created on Sun May 13 21:48:42 2018

@author: XMM
"""

import tensorflow as tf
import argparse
import pandas as pd
#from tensorflow.examples.tutorials.mnist import input_data
#import csv

#==============================================================================#
#MNIST数据集相关的常数  
input_node=100  #输入层的节点数。对于MNIST数据集，这个就等于图片的像素  
output_node=2  #输出层的节点数。这个等于类别的数目。因为MNIST数据集中需要区分的是0-9这10个数字，所以这里输出层的节点数为10  

#配置神经网络的参数  
layer1_node=500 #隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例  
                #这个隐藏层有500个节点  
batch_size=100  #一个训练batch中的训练数据个数。数字越小时，训练过程越接近，随机梯度下降；数字越大时，训练越接近梯度下降  
learning_rate_base = 0.8 #基础的学习率  
learning_rate_decay = 0.99 #学习率的衰减率  
regularization_rate = 0.0001 #描述模型复杂度的正则化在损失函数中的系数(防止过拟合)  
training_steps = 9001         #训练轮数  
moving_average_decay = 0.99  #滑动平均衰减率（控制模型更新速度阈值）  

#获取数据集    
#train_data,test_data = data.get_all()

#n_layer为第几层
def init_W_b(in_size,out_size,n_layer,regularizer = None):
    weightName = 'Weights%s' % n_layer
    biasesName = 'biases%s' % n_layer
    print(weightName)
    print(biasesName)
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1),dtype = tf.float32,name = weightName)
    biases = tf.Variable(tf.constant(0.1,shape=[out_size]),dtype = tf.float32,name = biasesName)
    
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(Weights))
    return Weights,biases
#    Wx_plus_b = tf.matmul(inputs,Weights) + biases
#    
#    if activation_function is None:
#        outputs = Wx_plus_b
#    else:
#        outputs = activation_function(Wx_plus_b)
#    return outputs

def inference(inputs,keep_prob,regularizer):
    #第一层神经网络
    weights1,biases1 = init_W_b(input_node, layer1_node, 1, regularizer)
    l1 = tf.nn.relu(tf.matmul(inputs,weights1)+biases1)
    l1 = tf.nn.dropout(l1,keep_prob)
    #第二层网络
    weights2,biases2 = init_W_b(layer1_node,output_node, 2, regularizer)
    prediction = tf.matmul(l1,weights2)+biases2
    prediction = tf.nn.dropout(l1,keep_prob)
    return prediction


def train(train_data,test_data):
    keep_prob = tf.placeholder(tf.float32)  #定义dropout保持结果的百分比
    xs = tf.placeholder(tf.float32,[None,input_node],name = 'xs')
    ys = tf.placeholder(tf.float32,[None,output_node],name = 'ys')
    
    #construct the network
#    weights1,biases1 = init_W_b(input_node,layer1_node,1)
#    weights2,biases2 = init_W_b(layer1_node,output_node,2)
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)  
    prediction = inference(xs,keep_prob,regularizer)
    #保存prediction至tf中
    tf.add_to_collection('pred_network',prediction)
    #==============================================================================#
    #定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量(trainable=False) .在使用Tensorflow训练神经网络时  
    #一般会将代表训练轮数的变量指定为不可训练的参数。  
    global_step = tf.Variable(0,trainable=False)    
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类  
    #给定训练轮数的变量可以加快训练早期变量的更新速度  
    variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)   
    #在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量(比如g  lobal_step)就不需要了  
    #tf.variables返回的就是图上集合  
    #GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定你trainable=False的参数  
    variables_averages_op=variable_averages.apply(tf.trainable_variables())  
    
    #==============================================================================#
    #compute loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=tf.argmax(ys, 1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)    
    #计算L2正则化损失函数  
  
    #计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项  
#    regularization = regularizer(weight1)+regularizer(weight2)    
    #总损失等于交叉熵损失和正则化损失的和  
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  
    
    #==============================================================================#
    learning_rate = tf.train.exponential_decay(  
        learning_rate_base,#基础的学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减  
        global_step,       #当前迭代的轮数  
        train_data.num_examples/batch_size, #过完所有的训练数据需要的迭代次数   
        learning_rate_decay)        #学习率衰减速度    
    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数。这里损失函数包含了  
    #交叉熵损失和L2正则化损失  
#    print(tf.train.GradientDescentOptimizer(learning_rate))  
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)  
    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数  
    #又要更新每一个参数的滑动平均值。为了一次完成多个操作，Tensorflow提供了  
    #tf.control_dependencies和tf.group两种机制下面两行程序和  
    #train_op=tf.group(train_step,variables_averages_op)是等价的  
    train_op=tf.group(train_step,variables_averages_op)  
    
    #==============================================================================#
    #计算准确率
    correct_prediction = tf.equal(tf.arg_max(prediction,1), tf.arg_max(ys,1))    
    #这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在  
    #这一组数据上的正确率  
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
    
    #声明tf.train.Saver类用于保存模型  
    saver = tf.train.Saver()  
    
    #==============================================================================#
    with tf.Session() as sess:  
        tf.global_variables_initializer().run()  
        #准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的  
        #条件和评判训练的效果  
        validate_feed={xs:train_data.input_data, ys:train_data.output_data,keep_prob:1}  
    #        print('训练数据量为：',sess.run(x))      
        #准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为  
        #模型优劣的最后评判标准  
        test_feed = {xs:test_data.input_data,ys:test_data.output_data,keep_prob:1}  
          
        #迭代地训练神经网络  
        for i in range(training_steps):  
            #每1000轮输出一次在验证数据集上的测试结果  
            if i%1000==0:  
                #将模型保存到这个文件下  
                save_path = saver.save(sess, 'DNN_model.ckpt')  
      
                #计算滑动平均模型在验证数据上的结果。因为MNIST数据集比较小，所以一次  
                #可以处理所有的验证数据。为了计算方便，本样例程序没有将验证数据划分为更小的batch  
                #当神经网络模型比较复杂或者验证数据比较大时，太大的batch  
                #会导致计算时间过长甚至发生内存溢出的错误  
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)  
                test_acc = sess.run(accuracy, feed_dict=test_feed)  
                print("After %d training step(s), validation accuracy using average model is %g, test accuracy using average model is %g" %(i,validate_acc,test_acc))  
      
            #产生这一轮使用的一个batch的训练数据，并运行训练过程  
    #            print('第',i,'次循环')
            x_batch,y_batch=train_data.next_batch(batch_size)  
    #            print('xs:',xs,'xs量的大小',len(xs))
    #            print('ys',ys)
    #            print('ys:',ys,'ys量的大小',len(ys))
            sess.run(train_op,feed_dict={xs:x_batch,ys:y_batch,keep_prob:1})  
    #            print('training',sess.run(train_op,feed_dict={x:xs,y_:ys}))
        #在训练结束之后，在测试数据上检测神经网络模型的最终正确率  
        test_acc=sess.run(accuracy,feed_dict=test_feed)  
        print("After %d training step(s), test accuracy using average model is %g "%(training_steps,test_acc)) 

def parse_args():

	parser = argparse.ArgumentParser(description="Run APR.")

	parser.add_argument('--input_trainFile', nargs='?', default='features/train_features_k25_cas50.csv',
	                    help='Input train cascadaes features file path')

	parser.add_argument('--input_testFile', nargs='?', default='features/validation1_features_k25_cas10.csv',
	                    help='Input test cascadaes features file path')

	parser.add_argument('--output', nargs='?', default='../dataset/features/result.csv',
	                    help='Features path')
	parser.add_argument('--k', type=int, default=25,
	                    help='Number of early adopters. Default is 25.')
	parser.add_argument('--myModel', type=int, default=0,
	                    help='1 denotes user MyModel featuers. 0 denotes APR features.')
	return parser.parse_args()

def read_features_csv(file, My_Model):
    #需要注意第一行数据未读取
    df = pd.read_csv(file)
    columns_size = df.columns.size
    if My_Model == 0:
        X = df.iloc[:,1: columns_size - args.k - 2]
    else:
        X = pd.concat([df.iloc[:,1: columns_size - 2* args.k -2] ,df.iloc[:, columns_size - args.k -2: -2]], axis=1)
    Y = df.iloc[:,-1]
#        print(X.head())
#        print(X,Y)
    return [X, Y]

def main(args):
    train_data = read_features_csv(args.input_trainFile, args.myModel)
    test_data = read_features_csv(args.input_testFile, args.myModel)
    columns  = train_data[0].shape[1]
    global input_node
    input_node = columns

    train(train_data, test_data)

if __name__ == '__main__':
    args = parse_args()
    main(args)