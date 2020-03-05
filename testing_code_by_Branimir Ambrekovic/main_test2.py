## Chongyi Li, Chunle Guo, Wenqi Ren, Runmin Cong, Junhui Hou, Sam Kwong, Dacheng Tao , "An Underwater Image Enhancement Benchmark Dataset and Beyond" IEEE TIP 2019 #######
## Project: https://li-chongyi.github.io/proj_benchmark.html
############################################################################################################################################################################

from model import T_CNN
from utils import *
import numpy as np
from absl import flags
import tensorflow as tf

import pprint
import os

#flags = tf.app.flags
# flags = argparse.ArgumentParser()
# flags.add_argument("epoch", type=int, default=120)
# flags.add_argument("batch_size", type=int, default=1)
# flags.add_argument("image_height", type=int, default=112)
# flags.add_argument("image_width", type=int, default=112)
# flags.add_argument("label_height", type=int, default=112)
# flags.add_argument("label_width", type=int, default=112)
# flags.add_argument("learning_rate", type=float, default=0.001)
# flags.add_argument("beta1", type=float, default=0.5)
# flags.add_argument("c_dim", type=int, default=3)
# flags.add_argument("c_depth_dim", type=int, default=1)
# flags.add_argument("checkpoint_dir",type=str, default= "checkpoint")
# flags.add_argument("sample_dir", type=str, default="sample")
# flags.add_argument("test_data_dir", type=str, default="test")
# flags.add_argument("is_train", type=bool, default=False)

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
#FLAGS, unparsed = flags.parse_known_args()


pp = pprint.PrettyPrinter()

def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
      os.makedirs(FLAGS.sample_dir)
    filenames = os.listdir('test_real')
    data_dir = os.path.join(os.getcwd(), 'test_real')
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    test_data_list = data + glob.glob(os.path.join(data_dir, "*.png"))+glob.glob(os.path.join(data_dir, "*.bmp"))+glob.glob(os.path.join(data_dir, "*.jpeg"))
    print(data_dir,test_data_list)

  # filenames1 = os.listdir('wb_real')
  # data_dir1 = os.path.join(os.getcwd(), 'wb_real')
  # data1 = glob.glob(os.path.join(data_dir1, "*.png"))
  # test_data_list1 = data1 + glob.glob(os.path.join(data_dir1, "*.jpg"))+glob.glob(os.path.join(data_dir1, "*.bmp"))+glob.glob(os.path.join(data_dir1, "*.jpeg"))

 # filenames2 = os.listdir('ce_real')
 # data_dir2 = os.path.join(os.getcwd(), 'ce_real')
 # data2 = glob.glob(os.path.join(data_dir2, "*.png"))
 # test_data_list2 = data2 + glob.glob(os.path.join(data_dir2, "*.jpg"))+glob.glob(os.path.join(data_dir2, "*.bmp"))+glob.glob(os.path.join(data_dir2, "*.jpeg"))

 # filenames3 = os.listdir('gc_real')
  # data_dir3 = os.path.join(os.getcwd(), 'wb_real')
 # data3 = glob.glob(os.path.join(data_dir3, "*.png"))
 # test_data_list3 = data3 + glob.glob(os.path.join(data_dir3, "*.jpg"))+glob.glob(os.path.join(data_dir3, "*.bmp"))+glob.glob(os.path.join(data_dir3, "*.jpeg"))

    for ide in range(0,len(test_data_list)):
        image_test =  cv.imread(test_data_list[ide],1)
        # wb_test =  get_image(test_data_list1[ide],is_grayscale=False)
        # ce_test =  get_image(test_data_list1[ide],is_grayscale=False)
        # gc_test =  get_image(test_data_list1[ide],is_grayscale=False)
        shape = image_test.shape
        print(ide)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
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
                      #test_wb_name = test_data_list1[ide],
                      #test_ce_name = test_data_list1[ide],
                      #test_gc_name = test_data_list1[ide],
                      id = ide
                      )
            print("Loop1")
            srcnn.train(FLAGS)
            sess.close()
        tf.compat.v1.get_default_graph().finalize()

if __name__ == '__main__':
  tf.compat.v1.app.run()
