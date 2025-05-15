import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #隐藏提示警告

#加载图像数据
mapFile = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\HSIs\\HYDICE.mat'
dataFile = mapFile  # HYDICE专用，论文里的图像用下面一行
# dataFile = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_ori\\abu-urban-1_ori.mat'
datahigh = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_low\\HYDICE_low1.mat'
datares = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_low_res\\HYDICE_low.mat'
data = scio.loadmat(dataFile)
data_map = scio.loadmat(mapFile)
data_l = scio.loadmat(datahigh)
ori = data['data'] # HYDICE专用，论文里的图像用下面一行
# ori = data['ori_new']
map = data_map['map']
low_ori = data_l['data_low']
bands = ori.shape[2]
width = ori.shape[1]
hight = ori.shape[0]
are = width*hight
low = np.reshape(low_ori,(are,bands))
# low_ori = np.reshape(ori,(are,bands))      ###原图直接训练，验证低频的好处
# low = (low_ori - low_ori.min())/(low_ori.max() - low_ori.min())

#训练网络参数
learning_rate = 0.0001
display_step = 1
num = 1300
n_input = bands

tf.set_random_seed(33)
#传入网络的数据
X = tf.placeholder("float", [None, n_input])

# 用字典的方式存储各隐藏层的参数
n_hidden_1 = 94 # 第一编码层神经元个数
n_hidden_2 = 47 # 第二编码层神经元个数
n_hidden_3 = 24
n_hidden_4 = 3
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
weights = {
 'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
 'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
 'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
 'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3],)),
 'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
 'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
 'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
 'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
 'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
 'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
 'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
 'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
 'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
 'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
 'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}

# 每一层结构都是 xW + b
# 构建编码器
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))
    return layer_4

# 构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    return layer_4

# 构建模型
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# 预测
y_pred = decoder_op
y_true = X

# 定义代价函数和优化器
fenzi = tf.reduce_sum(tf.multiply(y_pred,y_true),1,keep_dims=True)
fenmu_1 = tf.reduce_sum(tf.multiply(y_pred,y_pred),1,keep_dims=True)
fenmu_2 = tf.reduce_sum(tf.multiply(y_true,y_true),1,keep_dims=True)
fenmu_11 = tf.sqrt(fenmu_1)
fenmu_22 = tf.sqrt(fenmu_2)
fenmu = tf.multiply(fenmu_1,fenmu_2)
sam = tf.acos(tf.divide(fenzi,fenmu))
loss3 = tf.divide(tf.reduce_mean(sam),np.pi)    #光谱角
loss1 = tf.reduce_mean(tf.abs(y_true - y_pred))  # L1范数
loss2 = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  #L2范数
a = 0.01
r0 = 0.02
term2_fenzi = tf.constant(1.) - r0
r1 = tf.reduce_mean(encoder_op,0)
term2_fenmu = tf.constant(1.) - r1
kl1 = tf.multiply(r0,tf.log(tf.div(r0,r1)))
kl2 = tf.multiply(term2_fenzi,tf.log(tf.div(term2_fenzi,term2_fenmu)))
kl = kl1 + kl2
kl_loss = tf.reduce_sum(kl)
loss = loss2 + a*kl_loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

saver = tf.train.Saver()
model_path = "./low_HYDICE"
#在会话中执行神经网络
# time_start=time.time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
loss_val = np.zeros(shape=[1, num])
loss_num = np.zeros(shape=[1, num])
for i in range(num):
    _, c, encode_decode = sess.run([optimizer, loss, y_pred], feed_dict={X: low})   # 运行了两个函数，用两个变量保存
    if i % display_step == 0:
        print("number:", '%04d' % (i), "cost=", "{:.9f}".format(c))
        loss_val[:,i] = c
        loss_num[:,i] = i
saver.save(sess,model_path)
print("Optimization Finished!")

##loss曲线
plt.figure()
loss_val = np.squeeze(loss_val)
loss_num = np.squeeze(loss_num)
plt.plot(loss_num,loss_val)

#编码器输出
encode = sess.run(encoder_op, feed_dict={X: low})
feature = np.reshape(encode,(hight,width,n_hidden_4))
# time_end=time.time()
# print('totally cost',time_end-time_start)
print(feature.min(),feature.max())
scio.savemat(datares,{'low_res':feature})

f, a = plt.subplots(1, 3,figsize=(9,3))
for x in a.ravel():
    x.axis('off')
for b in range(3):
    a[b].imshow(feature[:, :, b], cmap=plt.cm.gray)
plt.show()


#简化绘图过程plt.figure()，fig.add
# f, a = plt.subplots(1, 4, figsize=(8, 2))
# for x in a.ravel():
#     x.axis('off')
# a[0].imshow(result[:, :, 6],cmap=plt.cm.gray)
# # a[1].imshow(low[:, :, 6],cmap=plt.cm.gray)
# # a[2].imshow(high[:, :, 6],cmap=plt.cm.gray)
# a[1].imshow(map,cmap=plt.cm.gray)
# plt.show()