## Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 #######
## Project: https://li-chongyi.github.io/proj_benchmark.html 
############################################################################################################################################################################

from model import T_CNN
from utils import *
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 120, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 112, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 112, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 112, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 112, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  filenames = os.listdir('test_real')
  data_dir = os.path.join(os.getcwd(), 'test_real')
  data = glob.glob(os.path.join(data_dir, "*.png"))
  test_data_list = data + glob.glob(os.path.join(data_dir, "*.jpg"))+glob.glob(os.path.join(data_dir, "*.bmp"))+glob.glob(os.path.join(data_dir, "*.jpeg"))

  filenames1 = os.listdir('wb_real')
  data_dir1 = os.path.join(os.getcwd(), 'wb_real')
  data1 = glob.glob(os.path.join(data_dir1, "*.png"))
  test_data_list1 = data1 + glob.glob(os.path.join(data_dir1, "*.jpg"))+glob.glob(os.path.join(data_dir1, "*.bmp"))+glob.glob(os.path.join(data_dir1, "*.jpeg"))

  filenames2 = os.listdir('ce_real')
  data_dir2 = os.path.join(os.getcwd(), 'ce_real')
  data2 = glob.glob(os.path.join(data_dir2, "*.png"))
  test_data_list2 = data2 + glob.glob(os.path.join(data_dir2, "*.jpg"))+glob.glob(os.path.join(data_dir2, "*.bmp"))+glob.glob(os.path.join(data_dir2, "*.jpeg"))

  filenames3 = os.listdir('gc_real')
  data_dir3 = os.path.join(os.getcwd(), 'wb_real')
  data3 = glob.glob(os.path.join(data_dir3, "*.png"))
  test_data_list3 = data3 + glob.glob(os.path.join(data_dir3, "*.jpg"))+glob.glob(os.path.join(data_dir3, "*.bmp"))+glob.glob(os.path.join(data_dir3, "*.jpeg"))

  for ide in range(0,len(test_data_list)):
    image_test =  get_image(test_data_list[ide],is_grayscale=False)
    wb_test =  get_image(test_data_list1[ide],is_grayscale=False)
    ce_test =  get_image(test_data_list2[ide],is_grayscale=False)
    gc_test =  get_image(test_data_list3[ide],is_grayscale=False)
    shape = image_test.shape
    tf.reset_default_graph()
    with tf.Session() as sess:
      # with tf.device('/cpu:0'):
        srcnn = T_CNN(sess, 
                  image_height=shape[0],
                  image_width=shape[1],  
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  c_depth_dim=FLAGS.c_depth_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  test_image_name = test_data_list[ide],
                  test_wb_name = test_data_list1[ide],
                  test_ce_name = test_data_list2[ide],
                  test_gc_name = test_data_list3[ide],
                  id = ide
                  )

        srcnn.train(FLAGS)
        sess.close()
    tf.get_default_graph().finalize()
if __name__ == '__main__':
  tf.app.run()
