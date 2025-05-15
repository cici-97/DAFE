import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #隐藏提示警告

#加载图像数据
dataFile = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_ori\\abu-urban-5_ori.mat'
data = scio.loadmat(dataFile)
ori = data['ori_new']
bands = ori.shape[2]
width = ori.shape[1]
hight = ori.shape[0]
are = width * hight
datahigh = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_low\\abu-urban-5_low1.mat'
datares = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_low_res\\abu-urban-5_low.mat'
data_l = scio.loadmat(datahigh)
low_ori = data_l['data_low']
low = np.reshape(low_ori,(are,bands))


#训练网络参数
learning_rate = 0.0001
display_step = 1
num = 1300
n_input = bands

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
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
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

time_start=time.time()
saver = tf.train.Saver()
model_path = "./model_low1"
#在会话中执行神经网络
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
# loss_val = np.zeros(shape=[1, num])
# loss_num = np.zeros(shape=[1, num])
# for i in range(num):
#     _, c, encode_decode = sess.run([optimizer, loss, y_pred], feed_dict={X: low})   # 运行了两个函数，用两个变量保存
#     if i % display_step == 0:
#         print("number:", '%04d' % (i), "cost=", "{:.9f}".format(c))
#         loss_val[:,i] = c
#         loss_num[:,i] = i
saver.restore(sess,model_path)
print("Model restored.")

#编码器输出
encode = sess.run(encoder_op, feed_dict={X:low})
feature = np.reshape(encode,(hight,width,3))
time_end=time.time()
print(time_end-time_start)
print(feature.min(),feature.max())
scio.savemat(datares,{'low_res':feature})

tf.reset_default_graph()

# #重构输出
# encode_decode_low = sess.run(y_pred, feed_dict={X: low})
# max1 = encode_decode_low.max()
# min1 = encode_decode_low.min()
# encode_decode_low_1 = (encode_decode_low - min1)/(max1 - min1)
# diff = np.reshape(encode_decode_low_1,(hight,width,bands))
# scio.savemat(datares, {'low_res': diff})
# diff_add = np.zeros(shape=[hight, width])
# for b in range (bands):
#     ori_img = (diff[:,:,b])
#     diff_add = diff_add + ori_img


# f, a = plt.subplots(1, 4,figsize=(16,4))
# for x in a.ravel():
#     x.axis('off')
# for b in range(3):
#     a[b].imshow(feature[:, :, b], cmap=plt.cm.gray)
# # a[3].imshow(diff_add,cmap=plt.cm.gray)   ##重构结果
# plt.show()
