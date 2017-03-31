import tensorflow as tf
import utils
from helper import *
import numpy as alice_input
from glob import glob
import os
from tensorflow.contrib.layers import convolution2d
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops.nn import sigmoid_cross_entropy_with_logits as cross_entropy
from tensorflow.contrib.layers import batch_norm as BatchNorm
#tf.merge_all_summaries = tf.summary.merge_all
#tf.train.SummaryWriter = tf.summary.FileWriter
class Model:
    def __init__(self, sess, conf, N, batch_size, learning_rate, x_weidu = 28, y_weidu = 28, rgb_weidu = 3, shape = (28, 28, 3)):
        '''
        sess:tensorflow的Session()会话
        N:明文的长度
        batch_size:生成样例的多少
        x_weidu:图片的长
        y_weidu:图片的宽
        rgb_weidu:1为单色，3为rgb三色
        '''
        self.sess = sess
        self.conf = conf
        self.P = utils.generate_data(batch_size, N)
        self.x_weidu = x_weidu
        self.y_weidu = y_weidu
        self.rgb = rgb_weidu
        self.batch_size = batch_size
        self.data_images = tf.placeholder(tf.float32, [self.batch_size] + list(shape))
        alice_image = tf.reshape(self.data_images, [batch_size, -1])
        alice_input = tf.concat([self.P, alice_image], 1)

        #Alice结构
        image_length = self.x_weidu * self.y_weidu * self.rgb
        alice_fc = fc_layer(alice_input, shape = (image_length + N, 2 * image_length), name = 'alice/alice_fc')
        alice_fc = tf.reshape(alice_fc, [batch_size, 2 * image_length, 1])
        alice_conv1 = conv_layer(alice_fc, filter_shape = [4,1,2], stride = 1, sigmoid = True, name = 'alice/alice_conv1')
        alice_conv2 = conv_layer(alice_conv1, filter_shape = [2,2,4], stride = 2, sigmoid = True, name = 'alice/alice_conv2')
        alice_conv3 = conv_layer(alice_conv2, filter_shape = [1,4,4], stride = 1, sigmoid = True, name = 'alice/alice_conv3')
        alice_conv4 = conv_layer(alice_conv3, filter_shape = [1,4,1], stride = 1, sigmoid = False, name = 'alice/alice_conv4')


        self.bob_input = tf.reshape(alice_conv4, [-1, self.x_weidu, self.y_weidu, self.rgb])
        

        #Bob网络结构
        bob_conv1 = convolution2d(self.bob_input, 64, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv1')

        bob_conv2 = convolution2d(bob_conv1, 64 * 2, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv2')

        bob_conv3 = convolution2d(bob_conv2, 64 * 4, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv3')

        bob_conv4 = convolution2d(bob_conv3, 64 * 8,kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'bob/conv4')

        bob_conv4 = tf.reshape(bob_conv4, [batch_size, -1])
        bob_fc = fully_connected(bob_conv4, N, activation_fn = tf.nn.tanh, normalizer_fn = BatchNorm,
        weights_initializer=tf.random_normal_initializer(stddev=1.0))
        #Bob_loss = tf.reduce_mean(utils.Distance(bob_fc, self.P, [1]))

        #Eve网络
        eve_real = self.discriminator_stego_nn(self.data_images, batch_size)
        eve_fake = self.discriminator_stego_nn(self.bob_input, batch_size)

        #Bob损失函数
        self.Bob_loss = tf.reduce_mean(utils.Distance(bob_fc, self.P, [1]))

        #Eve的损失函数
        Eve_fake_loss = tf.reduce_mean(cross_entropy(logits = eve_fake, labels = tf.zeros_like(eve_fake)))
        Eve_real_loss = tf.reduce_mean(cross_entropy(logits = eve_real, labels = tf.ones_like(eve_real)))
        self.Eve_loss = Eve_fake_loss + Eve_real_loss

        Alice_C_loss = tf.reduce_mean(utils.Distance(self.bob_input, self.data_images, [1,2,3]))
        self.Alice_loss = self.conf.alphaA * Alice_C_loss + self.conf.alphaB * self.Bob_loss + self.conf.alphaC * self.Eve_loss

        #定义优化器
        optimizer1 = tf.train.AdamOptimizer(self.conf.learning_rate)
        optimizer2 = tf.train.AdamOptimizer(self.conf.learning_rate)
        optimizer3 = tf.train.AdamOptimizer(self.conf.learning_rate)
        
        #获取变量列表
        self.Alice_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alice/")
        self.Bob_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'bob/')
        self.Eve_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eve/')

        #定义trainning step
        self.alice_step = optimizer1.minimize(self.Alice_loss, var_list= self.Alice_vars)
        self.bob_step = optimizer2.minimize(self.Bob_loss, var_list= self.Bob_vars)
        self.eve_step = optimizer3.minimize(self.Eve_loss, var_list= self.Eve_vars)

        #定义Saver
        self.alice_saver = tf.train.Saver(self.Alice_vars)
        self.bob_saver = tf.train.Saver(self.Bob_vars)
        self.eve_saver = tf.train.Saver(self.Eve_vars)

        self.Bob_bit_error = utils.calculate_bit_error(self.P, bob_fc, [1])
        self.Alice_bit_error = utils.calculate_bit_error(self.data_images, self.bob_input, [1,2,3])
        self.Eve_fake_error = tf.reduce_mean(tf.nn.sigmoid(eve_fake))
        self.Eve_real_error = tf.reduce_mean(tf.nn.sigmoid(eve_real))
        print("初始化")
    
    def train(self, epochs):
        data_images_path = glob(os.path.join(self.conf.pic_dict, "*.%s" % self.conf.img_format))
        if(len(data_images_path) == 0):
            print("No Images here: %s" % self.conf.pic_dict)
            exit(1)
        data = [utils.imread(path) for path in data_images_path]

        data = [utils.transform(image) for image in data]

        #merged = tf.merge_all_summaries()
        #train_weiter = tf.train.SummaryWriter('./logs_sgan', self.sess.graph)
        #tf.summary.scalar("bob_input", self.bob_input)
        #merged_summary_op = tf.summary.merge_all()
        #summary_writer = tf.summary.FileWriter('./logs', self.sess.graph)
        tf.initialize_all_variables().run()
        bob_results = []
        alice_results = []

        while(len(data) < self.batch_size):
            data.append(data)
        for i in range(epochs):
            self.sess.run(self.alice_step, feed_dict = {self.data_images: data[ 0: self.batch_size]})
            self.sess.run(self.alice_step, feed_dict = {self.data_images: data[ 0: self.batch_size]})
            self.sess.run(self.alice_step, feed_dict = {self.data_images: data[ 0: self.batch_size]})
            self.sess.run(self.bob_step, feed_dict= {self.data_images: data[0 : self.batch_size]})
            self.sess.run(self.eve_step, feed_dict= {self.data_images: data[0 : self.batch_size]})
            if i % 100 == 0:
                bit_error, alice_error, eve_real, eve_fake = self.sess.run([self.Bob_bit_error, self.Alice_bit_error, self.Eve_real_error, self.Eve_fake_error], 
                feed_dict= {self.data_images: data[0 : self.batch_size]})
                print("step {}, bob bit error {}, alice bit error {}, Eve real {}, Eve fake {}".format(i, bit_error, alice_error, eve_real, eve_fake))
                bob_results.append(bit_error)
                alice_results.append(alice_error)
                #summary_str = self.sess.run(merged_summary_op, feed_dict = {self.data_images: data[ 0: self.batch_size]})
                #summary_writer.add_summary(summary_str, i)
            if (i > 45000) and (i % 100 == 0):
                c_output = self.sess.run(self.bob_input, feed_dict= {self.data_images: data[0 : self.batch_size]})
                c_output = utils.inverse_transform(c_output)
                utils.save_images(c_output, i, self.conf.save_pic_dict)
        #保存图片
        c_output = self.sess.run(self.bob_input, feed_dict= {self.data_images: data[0 : self.batch_size]})
        return bob_results, alice_results
                
            


### Eve的网络结构
    def discriminator_stego_nn(self, img, batch_size):
        eve_conv1 = convolution2d(img, 64, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv1')

        eve_conv2 = convolution2d(eve_conv1, 64 * 2, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv2')

        eve_conv3 = convolution2d(eve_conv2, 64 * 4,kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv3')

        eve_conv4 = convolution2d(eve_conv2, 64* 8, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/conv4')

        eve_conv4 = tf.reshape(eve_conv4, [batch_size, -1])

        #eve_fc = fully_connected(eve_conv4, 1, activation_fn = tf.nn.sigmoid, normalizer_fn = BatchNorm,
        #weights_initializer=tf.random_normal_initializer(stddev=1.0))
        eve_fc = fully_connected(eve_conv4, 1, normalizer_fn = BatchNorm, 
        weights_initializer=tf.random_normal_initializer(stddev=1.0))
        return eve_fc

        




