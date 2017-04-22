#!/usr/bin/python
# --*-- coding: utf-8 --*--
'''用来进行模型的测试'''

import os
from time import time

import numpy as np 
import tensorflow as tf 
import logger
import utils
from model import Model
import sys

flags = tf.app.flags
flags.DEFINE_float("alphaA", 0.6, "alphaA的值")
flags.DEFINE_float("alphaB", 1.8, "alphaB的值")
flags.DEFINE_float("alphaC", -0.2, "alphaC的值")
flags.DEFINE_float("learning_rate", 0.0002, "学习速率")
flags.DEFINE_string("pic_dict", "./pictures", "存放的图片的位置")
flags.DEFINE_string("save_pic_dict", "./savedPictures", "保存的图片位置")
flags.DEFINE_string('save_model_dict',"./savedModel", "存放的模型的位置")
flags.DEFINE_string("img_format", "jpg", "处理的图片格式")
flags.DEFINE_integer("batch_size", 32, "训练的样本数量")
flags.DEFINE_integer("plain_nums", 16, "明文的长度")
flags.DEFINE_integer("training", 1, "一共训练多少次")
flags.DEFINE_integer("training_epochs", 50000, "训练轮数")

FLAGS = flags.FLAGS
logger.log("Testing begin")
with tf.Session() as sess:
    model = Model(sess, FLAGS, FLAGS.plain_nums, FLAGS.batch_size, FLAGS.learning_rate)
    model.variable_init()
    model.restore_alice(FLAGS.save_model_dict + '/alice_model.ckpt')
    model.restore_alice(FLAGS.save_model_dict + '/bob_model.ckpt')
    model.restore_alice(FLAGS.save_model_dict + '/eve_model.ckpt')
    model.test()
