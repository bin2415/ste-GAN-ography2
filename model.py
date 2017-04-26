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

class batch_norm(object):
    """代码参考了http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.), trainable=True)
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02), trainable=True)

                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')

                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


class Model:
    def __init__(self, sess, conf, N, batch_size, learning_rate, x_weidu = 32, y_weidu = 32, rgb_weidu = 3, shape = (32, 32, 3)):
        '''
        sess:tensorflow的Session()会话
        N:明文的长度
        batch_size:生成样例的多少
        x_weidu:图片的长
        y_weidu:图片的宽Eve_real_error
        rgb_weidu:1为单色，3为rgb三色
        '''
        self.sess = sess
        self.conf = conf
        #self.P = utils.generate_data(batch_size, N)
        self.x_weidu = x_weidu
        self.y_weidu = y_weidu
        self.rgb = rgb_weidu
        self.batch_size = batch_size
        self.data_images = tf.placeholder(tf.float32, [self.batch_size] + list(shape))
        self.data_input = tf.placeholder(tf.float32, [self.batch_size, N])
        self.N = N
        alice_image = tf.reshape(self.data_images, [batch_size, -1])
        alice_input = tf.concat([self.data_images, alice_image], 1)

        drop_rate = tf.constant(0.5, dtype = tf.float32)

        self.g_bn0 = batch_norm(name = 'alice/bn0')
        self.g_bn1 = batch_norm(name = 'alice/bn1')
        self.g_bn2 = batch_norm(name = 'alice/bn2')
        self.g_bn3 = batch_norm(name = 'alice/bn3')
        self.g_bn4 = batch_norm(name = 'alice/bn4')
        self.g_bn5 = batch_norm(name = 'alice/bn5')

        #Alice结构
        image_length = self.x_weidu * self.y_weidu * self.rgb
        alice_fc = fc_layer(alice_input, shape = (image_length + N, image_length*8), name = 'alice/alice_fc')
        #alice_fc = tf.reshape(alice_fc, [batch_size, 2 * image_length, 1])
        #alice_conv1 = conv_layer(alice_fc, filter_shape = [4,1,2], stride = 1, sigmoid = True, name = 'alice/alice_conv1')
        #alice_conv2 = conv_layer(alice_conv1, filter_shape = [2,2,4], stride = 2, sigmoid = True, name = 'alice/alice_conv2')
        #alice_conv2 = tf.nn.dropout(alice_conv2, drop_rate)
        #alice_conv3 = conv_layer(alice_conv2, filter_shape = [1,4,4], stride = 1, sigmoid = True, name = 'alice/alice_conv3')
        #alice_conv4 = conv_layer(alice_conv3, filter_shape = [1,4,1], stride = 1, sigmoid = False, name = 'alice/alice_conv4')
        alice_fc = tf.reshape(alice_fc, [-1, self.x_weidu, self.y_weidu, self.rgb*8])
        alice_fc = self.g_bn0(alice_fc, train = True)
        aclie_fc = tf.nn.relu(alice_fc)

        alice_conv1 = self.conv2d_transpose(alice_fc, [self.batch_size, self.x_weidu*2, self.y_weidu*2, self.rgb * 4], name = 'alice/conv1')
        alice_conv1 = self.g_bn1(alice_conv1, train = True)
        alice_conv1 = tf.nn.relu(alice_conv1)

        alice_conv2 = self.conv2d_transpose(alice_conv1, [self.batch_size, self.x_weidu * 4, self.y_weidu * 4, self.rgb * 8], name = 'alice/conv2')
        alice_conv2 = self.g_bn2(alice_conv2, train = True)
        alice_conv2 = tf.nn.relu(alice_conv2)

        #alice_conv3 = self.conv2d_transpose(alice_conv2, [self.batch_size, self.x_weidu * 8, self.y_weidu * 8, self.rgb * 16], name = 'alice/conv3')
        #alice_conv3 = self.g_bn3(alice_conv3, train = True)
        #alice_conv3 = tf.nn.relu(alice_conv3)

        alice_conv4 = self.conv2d(alice_conv2, self.rgb * 8, name = 'alice/conv4')
        alice_conv4 = self.g_bn4(alice_conv4, train = True)
        alice_conv4 = tf.nn.relu(alice_conv4)

        alice_conv5 = self.conv2d(alice_conv4, self.rgb * 4, name = 'alice/conv5')
        alice_conv5 = self.g_bn5(alice_conv5, train = True)
        alice_conv5 = tf.nn.relu(alice_conv5)

        alice_conv6 = self.conv2d(alice_conv5, self.rgb, d_h = 1, d_w = 1, name = 'alice/conv6')
        #alice_conv6 = self.g_bn3(alice_conv6, train = True)
        alice_conv6 = tf.nn.tanh(alice_conv6)

        #alice_conv7 = self.conv2d(alice_conv6, self.rgb, d_h = 1, d_w = 1, name = 'alice/conv7')
        #alice_conv7 = tf.nn.tanh(alice_conv7)






        #self.bob_input = tf.reshape(alice_conv4, [-1, self.x_weidu, self.y_weidu, self.rgb])
        self.bob_input = alice_conv6
        

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
        eve_real = self.discriminator_stego_nn(self.data_images, batch_size, 'real')
        eve_fake = self.discriminator_stego_nn(self.bob_input, batch_size, 'fake')

        #Bob损失函数
        self.Bob_loss = tf.reduce_mean(utils.Distance(bob_fc, self.P, [1]))

        #Eve的损失函数
        Eve_fake_loss = tf.reduce_mean(cross_entropy(logits = eve_fake, labels = tf.zeros_like(eve_fake)))
        Eve_real_loss = tf.reduce_mean(cross_entropy(logits = eve_real, labels = tf.ones_like(eve_real)))
        self.Eve_loss = Eve_fake_loss + Eve_real_loss

        Alice_C_loss = tf.reduce_mean(utils.Distance(self.bob_input, self.data_images, [1,2,3]))
        self.Alice_loss = self.conf.alphaA * Alice_C_loss + self.conf.alphaB * self.Bob_loss + self.conf.alphaC * self.Eve_loss

        #定义优化器
        optimizer1 = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        optimizer2 = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        optimizer3 = tf.train.AdamOptimizer(self.conf.learning_rate, beta1=self.conf.beta1)
        #optimizer4 = tf.train.AdamOptimizer(self.conf.learning_rate)
        
        #获取变量列表
        self.Alice_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "alice/")
        self.Bob_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'bob/')
        self.Eve_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eve/')
        print(self.Bob_vars)

        #定义trainning step
        self.alice_step = optimizer1.minimize(self.Alice_loss, var_list= self.Alice_vars)
        self.bob_step = optimizer2.minimize(self.Bob_loss, var_list= self.Bob_vars)
        self.eve_step = optimizer3.minimize(self.Eve_loss, var_list= self.Eve_vars)
        #self.alice_step_only = optimizer4.minimize(Alice_C_loss, var_list= self.Alice_vars)

        #定义Saver
        self.alice_saver = tf.train.Saver(self.Alice_vars)
        self.bob_saver = tf.train.Saver(self.Bob_vars)
        self.eve_saver = tf.train.Saver(self.Eve_vars)

        self.Bob_bit_error = utils.calculate_bit_error(self.P, bob_fc, [1])
        self.Alice_bit_error = utils.calculate_bit_error(self.data_images, self.bob_input, [1,2,3])
        self.Eve_fake_error = tf.reduce_mean(tf.nn.sigmoid(eve_fake))
        self.Eve_real_error = tf.reduce_mean(tf.nn.sigmoid(eve_real))

        #Saver
        self.alice_saver = tf.train.Saver(self.Alice_vars)
        self.bob_saver = tf.train.Saver(self.Bob_vars)
        self.eve_saver = tf.train.Saver(self.Eve_vars)

        self.saver = tf.train.Saver()

        #self.g_bn0 = batch_norm(name = 'alice/bn0')
        #self.g_bn1 = batch_norm(name = 'alice/bn1')
        #self.g_bn2 = batch_norm(name = 'alice/bn2')
        #self.g_bn3 = batch_norm(name = 'alice/bn3')

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
        self.sess.run(tf.global_variables_initializer())
        bob_results = []
        alice_results = []

        while(len(data) < self.batch_size):
            data.append(data)
        
        if len(data) > 1024:
            data = data[0 : 1024]

        lens = len(data)
        input_data = np.random.random_integers(0,1,size = (4096, N))
        startInputIndex = 0
        for i in range(epochs):
            startIndex = (i * self.batch_size) % lens
            endIndex = startIndex + self.batch_size
            if endIndex > lens:
                dataTrain = data[lens-self.batch_size:lens]
            else:
                dataTrain = data[startIndex : endIndex]
            if startInputIndex >= 4096:
                startInputIndex = startInputIndex - 4096
            input_data1 = input_data[startInputIndex : startInputIndex + self.batch_size]
            #if i >=0 and i <= 30000:
                ##self.sess.run(self.alice_step_only, feed_dict = {self.data_images: data[ 0: self.batch_size]})
            #self.sess.run(self.alice_step_only, feed_dict = {self.data_images: data[ 0: self.batch_size]})
            #self.sess.run(self.alice_step, feed_dict = {self.data_images: dataTrain})
            #self.sess.run(self.alice_step, feed_dict = {self.data_images: dataTrain})
            self.sess.run(self.alice_step, feed_dict = {self.data_images: dataTrain, self.data_input:input_data1})
            #if i > 30000:
            #    self.sess.run(self.bob_step, feed_dict= {self.data_images: data[0 : self.batch_size]})
            #    self.sess.run(self.eve_step, feed_dict= {self.data_images: data[0 : self.batch_size]})
            self.sess.run(self.bob_step, feed_dict= {self.data_images: dataTrain, self.data_input:input_data1})
            #self.sess.run(self.eve_step, feed_dict= {self.data_images: data[0 : self.batch_size]})
            self.sess.run(self.eve_step, feed_dict= {self.data_images: dataTrain, self.data_input:input_data1})
            #self.sess.run(self.alice_step, feed_dict = {self.data_images: data[ 0: self.batch_size]})
            if i % 100 == 0:
                bit_error, alice_error, eve_real, eve_fake = self.sess.run([self.Bob_bit_error, self.Alice_bit_error, self.Eve_real_error, self.Eve_fake_error], 
                feed_dict= {self.data_images: dataTrain, self.data_input:input_data1})
                print("step {}, bob bit error {}, alice bit error {}, Eve real {}, Eve fake {}".format(i, bit_error, alice_error, eve_real, eve_fake))
                bob_results.append(bit_error)
                alice_results.append(alice_error)
                #summary_str = self.sess.run(merged_summary_op, feed_dict = {self.data_images: data[ 0: self.batch_size]})
                #summary_writer.add_summary(summary_str, i)
            if (i > 48000) and (i % 100 == 0):
                c_output = self.sess.run(self.bob_input, feed_dict= {self.data_images: dataTrain, self.data_input:input_data1})
                c_output = utils.inverse_transform(c_output)
                utils.save_images(c_output, i/100, self.conf.save_pic_dict)
        #保存图片
        #c_output = self.sess.run(self.bob_input, feed_dict= {self.data_images: da})
        return bob_results, alice_results

    def test(self):
        data_images_path = glob(os.path.join(self.conf.pic_dict, "*.%s" % self.conf.img_format))
        if(len(data_images_path) == 0):
            print("No Images here: %s" % self.conf.pic_dict)
            exit(1)
        data = [utils.imread(path) for path in data_images_path]
        data = [utils.transform(image) for image in data]

            #tf.initialize_all_variables().run()
        testDataStart = 1024
        testDataEnd = len(data)
        i = 0
        while testDataStart <= testDataEnd:
            if testDataStart >= testDataEnd - self.batch_size:
                testData = data[testDataEnd-self.batch_size : testDataEnd]
            else:
                testData = data[testDataStart : testDataStart + self.batch_size]
            testDataStart += self.batch_size
            bit_error, alice_error, eve_real, eve_fake = self.sess.run([self.Bob_bit_error, self.Alice_bit_error, self.Eve_real_error, self.Eve_fake_error], 
            feed_dict= {self.data_images: testData})
            print("step {}, bob bit error {}, alice bit error {}, Eve real {}, Eve fake {}".format(i, bit_error, alice_error, eve_real, eve_fake))
            
                
    def variable_init(self):
        self.sess.run(tf.global_variables_initializer())


### Eve的网络结构
    def discriminator_stego_nn(self, img, batch_size, name):
        eve_input = self.image_processing_layer(img)
        eve_conv1 = convolution2d(eve_input, 64, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/' + name + '/conv1')

        eve_conv2 = convolution2d(eve_conv1, 64 * 2, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/' + name + '/conv2')

        eve_conv3 = convolution2d(eve_conv2, 64 * 4,kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/' + name + '/conv3')

        eve_conv4 = convolution2d(eve_conv3, 64* 8, kernel_size = [5, 5], stride = [2,2],
        activation_fn= tf.nn.relu, normalizer_fn = BatchNorm, scope = 'eve/' + name + '/conv4')

        eve_conv4 = tf.reshape(eve_conv4, [batch_size, -1])

        #eve_fc = fully_connected(eve_conv4, 1, activation_fn = tf.nn.sigmoid, normalizer_fn = BatchNorm,
        #weights_initializer=tf.random_normal_initializer(stddev=1.0))
        eve_fc = fully_connected(eve_conv4, 1, normalizer_fn = BatchNorm, 
        weights_initializer=tf.random_normal_initializer(stddev=1.0))
        return eve_fc
    

    '''
    保存模型
    '''
    def save(self, save_path):
        '''
        arguments:
        save_path: string
        要保存的模型的地址
        '''
        #self.alice_saver.save(self.sess, save_path + '/alice_model.ckpt')
        #self.bob_saver.save(self.sess, save_path + '/bob_model.ckpt')
        self.saver.save(self.sess, save_path + '/save.ckpt')
    
    #先对图片进行处理
    def image_processing_layer(self, X):
        K = 1 / 12. * tf.constant(
            [
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]
            ], dtype= tf.float32
        )
        #kernel = tf.pack([K, K, K])
        #kernel = tf.pack([kernel, kernel, kernel])
        kernel = tf.stack([K, K, K])
        kernel = tf.stack([kernel, kernel, kernel])

        return tf.nn.conv2d(X, tf.transpose(kernel, [2, 3, 0, 1]), [1, 1, 1, 1], padding='SAME')
    
    def restore_alice(self, restore_path):
        self.alice_saver.restore(self.sess, restore_path)
    
    def restore_bob(self, restore_path):
        self.bob_saver.restore(self.sess, restore_path)

    def restore_eve(self, restore_path):
        self.eve_saver.restore(self.sess, restore_path)

    def restore_saver(self, restore_path):
        self.saver.restore(self.sess, restore_path)


    #反卷积网络
    def conv2d_transpose(self, input_, output_shape, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "deconv2d"):
        with tf.variable_scope(name):
            #filter: [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer= tf.random_normal_initializer(stddev = stddev), trainable=True
            )
            return tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides = [1, d_h, d_w, 1])
    
    #卷积网络
    def conv2d(self, input_, output_channel, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "conv2d"):
        with tf.variable_scope(name):
            #filter: [height, width, in_channels, output_channels]
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_channel],
                                initializer= tf.random_normal_initializer(stddev = stddev), trainable=True
            )
            return tf.nn.conv2d(input_, w, strides = [1, d_h, d_w, 1], padding = 'SAME')
'''
    def conv2d2(self, input_, output_channel, k_h = 5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.2, name = "deconv2d"):
        with tf.variable_scope(name):
            #filter: [height, width, in_channels, output_channels]
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_channel],
                                initializer= tf.random_normal_initializer(stddev = stddev)
            )
            return tf.nn.conv2d(input_, w, strides = [1, d_h, d_w, 1], padding = 'SAME')'''
        


        
        



        




