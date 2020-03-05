
from model_train import T_CNN
from utils import (
  imsave,
  prepare_data
)
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 400, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 112, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 112, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 112 ,"The size of label to produce [230]")
flags.DEFINE_integer("label_width", 112, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS


pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  
  with tf.Session() as sess:
    srcnn = T_CNN(sess, 
                  image_height=FLAGS.image_height,
                  image_width=FLAGS.image_width, 
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir
                  )

    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
