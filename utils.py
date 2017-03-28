#!/usr/bin/python
# -*- coding: utf-8 -*- 
'''
  @Author: binpang 
  @Date: 2017-03-19 09:28:28 
  @Last Modified by:   binpang 
  @Last Modified time: 2017-03-19 09:28:28 
'''

import tensorflow as tf
import scipy.misc
import numpy as np
import os

def generate_data(batch_size, length):
      P = 2 * tf.random_uniform([batch_size, length], minval = 0, maxval = 2, dtype = tf.int32) -1
      return tf.to_float(P)

def bias_variable(shape, value, name):
      initial = tf.constant(value, shape = shape)
      b = tf.Variable(initial, name = name)
      return b




def weight_variable(shape, std, name):
      initial = tf.truncated_normal(shape, stddev = std)   #生成正太分布数据，通常作为权重的初始值
      W = tf.Variable(initial, name = name)
      return W

''' 从特定路径读取图片'''
def imread(path):
      img = scipy.misc.imread(path)
      img = img.astype(np.float)
      img_shape = list(img.shape)
      #转化为3维数组
      if len(img_shape) == 2:
            img_shape.append(1)
            img = img.reshape(img_shape)
      return img

#保存图片
def imsave(images, size, path):
      h, w = images.shape[1], images.shape[2]
      img = np.zeros((h * size[0], w * size[1], 3))

      for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            img[j*h:j*h+h, i*w:i*w+w, :] = image
      return scipy.misc.imsave(path, img)

#将图片转化为一维数组
def convertImg2Arr(image):
      img_shape = list(np.array(image).shape)
      result = list()
      for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                  for k in range(img_shape[2]):
                        result.append(image[i][j][k])
      return result

#将一维数组转化为图片
def convertArr2Img(list, width, height, rgb):
      image = np.ones((width, height, rgb), dtype = int16)
      for i in range(width):
            for j in range(height):
                  for k in range(rgb):
                        image[i][j][k] = list[i*j*k]
      return image

'''
求图片或明文的距离
@param weidu: 如果是[1],则是求文字的距离，如果是[1,2, 3]则是求图片的距离
'''
def Distance(P1, P2, weidu):
      return tf.reduce_sum(tf.abs(P1 - P2), weidu)

def calculate_bit_error(P1, P2, weidu):
      boolean_error = tf.cast(tf.not_equal(tf.sign(P1), tf.sign(P2)), tf.float32)
      return tf.reduce_mean(tf.reduce_sum(boolean_error, weidu))


def save_images(images, i, folder):
    for idx, image in enumerate(inverse_transform(images)):
        scipy.misc.imsave(os.path.join(folder, '%s.png' % (i * idx)), image)

#保存图片的时候需要处理
def inverse_transform(images):
    return (images + 1.) / 2.

#将图片每一像素的值转为-1到1之间
def transform(images):
      return np.array(images) / 127.5 - 1.0
      



  

