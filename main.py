#!/usr/bin/python
# -*- coding: utf-8 -*- 
'''
  @Author: binpang 
  @Date: 2017-03-19 14:26:31 
  @Last Modified by:   binpang 
  @Last Modified time: 2017-03-19 14:26:31 
'''
import os
from time import time

import numpy as np 
import tensorflow as tf
import logger
import utils
from model import Model
import sys
#sys.setrecursionlimit(1000)

flags = tf.app.flags
flags.DEFINE_float("alphaA", 0.6, "alphaA的值")
flags.DEFINE_float("alphaB", 0.2, "alphaB的值")
flags.DEFINE_float("alphaC", 0.2, "alphaC的值")
flags.DEFINE_float("learning_rate", 0.0008, "学习速率")
flags.DEFINE_string("pic_dict", "./pictures", "存放的图片的位置")
flags.DEFINE_string("save_pic_dict", "/savedPictures", "保存的图片位置")
flags.DEFINE_string("img_format", "jpg", "处理的图片格式")
flags.DEFINE_integer("batch_size", 16, "训练的样本数量")
flags.DEFINE_integer("plain_nums", 32, "明文的长度")
flags.DEFINE_integer("training", 1, "一共训练多少次")
flags.DEFINE_integer("training_epochs", 50000, "训练轮数")

FLAGS = flags.FLAGS
for i in range(FLAGS.training):
      tf.reset_default_graph()
      logger.log("training begin")
      with tf.Session() as sess:
          model = Model(sess, FLAGS, FLAGS.plain_nums, FLAGS.batch_size, FLAGS.learning_rate)
          model.train(50000)
      #model = Model(FLAGS, FLAGS.plain_nums, FLAGS.batch_size, FLAGS.learning_rate)
      #print("training {0} begining".format(i))
      #bob_results = model.train(FLAGS.training_epochs)
      #alice_processed_results = model.bob_input  #Alice最终加工生成的图片
      #alice_processed_results = alice_processed_results.eval()
      #alice_processed_results = utils.inverse_transform(alice_processed_results)
      #utils.save_images(alice_processed_results, i, FLAGS.save_pic_dict)  #存放图片
      

      






