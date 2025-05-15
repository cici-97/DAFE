import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  #隐藏提示警告
time_start=time.time()

#加载图像数据
mapFile = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\HSIs\\HYDICE.mat'
dataFile = mapFile  # HYDICE专用，论文里的图像用下面一行
# dataFile = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_ori\\abu-urban-1_ori.mat'
datahigh = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_high\\HYDICE_high1.mat'
datares = 'D:\\图像所\\毕设-专利-论文\\DFAE代码整理\\data_high_res\\HYDICE_high.mat'
data = scio.loadmat(dataFile)
data_map = scio.loadmat(mapFile)
data_h = scio.loadmat(datahigh)
ori = data['data'] # HYDICE专用，论文里的图像用下面一行
# ori = data['ori_new']
map = data_map['map']
high_ori = data_h['data_high']
bands = ori.shape[2]
width = ori.shape[1]
hight = ori.shape[0]
are = width*hight
high = np.reshape(high_ori,(are,bands))
# high_ori = np.reshape(high_ori,(are,bands))      ###原图直接训练，验证低频的好处
# high = (high_ori- high_ori.min())/(high_ori.max() - high_ori.min())

#训练网络参数
learning_rate = 0.0002
display_step = 1
num = 1300
n_input = bands

tf.set_random_seed(1)
#传入网络的数据
X = tf.placeholder("float", [None, n_input])

# 用字典的方式存储各隐藏层的参数
n_hidden_1 = 60 # 第一编码层神经元个数
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
weights = {
 'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
 'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
 'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
 'decoder_b1': tf.Variable(tf.random_normal([n_input])),
}

# 每一层结构都是 xW + b
# 构建编码器
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    return layer_1

# 构建解码器
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    return layer_1


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
a = 0.0001
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
model_path = "./high_HYDICE"
#在会话中执行神经网络
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
loss_val = np.zeros(shape=[1, num])
loss_num = np.zeros(shape=[1, num])
for i in range(num):
    _, c_high, encode_decode = sess.run([optimizer, loss, y_pred], feed_dict={X: high}) # 运行了两个函数，用两个变量保存
    if i % display_step == 0:
        print("number:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c_high))
        loss_val[:,i] = c_high
        loss_num[:,i] = i
saver.save(sess,model_path)
print("Optimization Finished!")

##loss曲线
plt.figure()
loss_val = np.squeeze(loss_val)
loss_num = np.squeeze(loss_num)
plt.plot(loss_num,loss_val)
plt.show()

#重构输出
encode_decode_high = sess.run(y_pred, feed_dict={X: high})
max1 = encode_decode_high.max()
min1 = encode_decode_high.min()
encode_decode_high_1 = (encode_decode_high - min1)/(max1 - min1)

diff_spectrum_high = abs(high - encode_decode_high)
# diff_spectrum_high = high - encode_decode_high
max0 = diff_spectrum_high.max()
min0 = diff_spectrum_high.min()
diff_spectrum_high_1 = (diff_spectrum_high - min0)/(max0 - min0)

#画图转换
diff_high_ori = np.reshape(diff_spectrum_high_1,(hight,width,bands))
diff = np.reshape(encode_decode_high_1,(hight,width,bands))
time_end=time.time()
print('time cost',time_end-time_start)

diff_add_high_ori = np.zeros(shape=[hight, width])
for b in range (bands):
    ori_img_high_ori = (diff_high_ori[:,:,b])
    diff_add_high_ori = diff_add_high_ori + ori_img_high_ori
scio.savemat(datares, {'high_res': diff_add_high_ori})

#重构输出
diff_add = np.zeros(shape=[hight, width])
for b in range (bands):
    ori_img = (diff[:,:,b])
    diff_add = diff_add + ori_img

ori_high = np.zeros(shape=[hight, width])
for b in range (bands):
    high_tem = (high_ori[:,:,b])
    # high_tem = (ori[:, :, b])
    ori_high = ori_high + high_tem
max2 = ori_high.max()
min2 = ori_high.min()
ori_1 = (ori_high - min2)/(max2 - min2)

#简化绘图过程plt.figure()，fig.add
f, a = plt.subplots(1, 4, figsize=(16, 4))
for x in a.ravel():
    x.axis('off')
a[0].imshow(ori_1,cmap=plt.cm.gray)    ##高频域原图
a[1].imshow(diff_add,cmap=plt.cm.gray)   ##重构结果
a[2].imshow(diff_add_high_ori,cmap=plt.cm.gray)   ##重构误差结果
a[3].imshow(map,cmap=plt.cm.gray)
plt.show()