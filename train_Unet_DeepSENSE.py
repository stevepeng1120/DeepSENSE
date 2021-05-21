"""
U-Net implementation in TensorFlow for DeepSENSE

Y = f(X)

X: input sensitivity (159, 159, 32)
Y: output sensitivity (159, 159, 32)

Loss function: minimize MSE

Notes:

Original Paper:
"""
import time
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import re

import scipy.misc
import math
import scipy.io
from matplotlib import pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import ipdb
import nibabel as nib

def image_augmentation3D(X, Y, mask):
  """Returns (maybe) augmented images
  (1) Random in-plane rotation and translation

  Args:
    X (4-D Tensor): image tensor of (H, W, Z, 32)
    Y (4-D Tensor): image tensor of (H, W, Z, 32)
    mask (3-D Tensor): mask image tensor of (H, W, Z)

  Returns:
    X
    Y
    mask
  """
  concat_image = tf.concat([X, Y], axis=-1)
  # XY rotation
  tmp_angle = tf.random_uniform([1], minval=-math.pi/18, maxval=math.pi/18) #  -10~10 degree for CNN2, -1~1 degree for CNN1
  concat_image = tf.transpose(concat_image, perm=[2, 0, 1, 3])
  concat_image = tf.contrib.image.rotate(concat_image, tmp_angle, interpolation='BILINEAR')  
  mask = tf.contrib.image.rotate(mask, tmp_angle, interpolation='NEAREST')
  # YZ
  concat_image = tf.transpose(concat_image, perm=[1, 2, 0, 3])
  mask = tf.transpose(mask, perm=[1, 2, 0])
  # ZX
  concat_image = tf.transpose(concat_image, perm=[1, 2, 0, 3])
  mask = tf.transpose(mask, perm=[1, 2, 0])

  # XY translation
  dxy = tf.random_uniform([2], -3, 3, dtype=tf.int32) # -3~3 for CNN2, -1~1 for CNN1
  dxy = tf.cast(dxy, tf.float32)
  concat_image = tf.transpose(concat_image, perm=[1, 2, 0, 3])
  concat_image = tf.contrib.image.translate(concat_image, dxy, interpolation='BILINEAR')
  mask = tf.transpose(mask, perm=[1, 2, 0])
  mask = tf.contrib.image.translate(mask, dxy, interpolation='NEAREST')

  # permute back to X,Y,Z,C
  concat_image = tf.transpose(concat_image, perm=[1, 2, 0, 3])
  X = concat_image[:, :, :, 0:32]
  Y = concat_image[:, :, :, 32:64]
  
  return X, Y, mask

def get_X_Y_mask(queue, augmentation=True):
  """
  Input pipeline:
    Queue -> CSV -> FileRead -> Decode bin

  (1) Queue contains a CSV filename
  (2) Text Reader opens the CSV
    CSV file contains 3 columns
    ["path/to/X.bin", "path/to/Y.bin", "path/to/mask.bin"]
  (3) File Reader opens all the files
  (4) Decode bin to tensors

  Notes:
    height, width = 159, 159

  Returns
    X, Y, mask
  """
  print "Start get images"
  text_reader = tf.TextLineReader(skip_header_lines=0)
  _, csv_content = text_reader.read(queue)
  print "Finish read csv_content"

  X_path, Y_path, mask_path = tf.decode_csv(csv_content, record_defaults=[[""], [""], [""]]) # X, Y, mask
  print "Finish decode csv_content"
  
  X_bytes = tf.read_file(X_path)
  Y_bytes = tf.read_file(Y_path)
  mask_bytes = tf.read_file(mask_path)
  print "Finish read image and mask bytes"
  
  X = tf.decode_raw(X_bytes, np.float32)
  X = tf.reshape(X, [32, 36, 159, 159])
  X = tf.cast(X, tf.float32)# cast to another type
  X = tf.transpose(X,perm=[3,2,1,0])
  X = tf.reshape(X, [159, 159, 36, 32])

  Y = tf.decode_raw(Y_bytes, np.float32)
  Y = tf.reshape(Y, [32, 36, 159, 159])
  Y = tf.cast(Y, tf.float32)# cast to another type
  Y = tf.transpose(Y,perm=[3,2,1,0])
  Y = tf.reshape(Y, [159, 159, 36, 32])
  
  mask = tf.decode_raw(mask_bytes, np.float32)
  mask = tf.reshape(mask, [36, 159, 159])
  mask = tf.cast(mask, tf.float32)# cast to another type
  mask = tf.transpose(mask,perm=[2,1,0])
  mask = tf.reshape(mask, [159, 159, 36])
  
  if augmentation:
	X, Y, mask = image_augmentation3D(X, Y, mask)

  print "Finish loading dataset!!"
  return X, Y, mask, Y_path

def conv(input_, n_filters, training, flags, name, pool=False, activation=tf.nn.leaky_relu):
  """{Conv -> BN -> RELU}

  Args:
    input_ (4-D Tensor): (batch_size, H, W, C)
    n_filters (list): number of filters [int, int]
    training (1-D Tensor): Boolean Tensor
    name (str): name postfix
    pool (bool): If True, MaxPool2D
    activation: Activaion functions

  Returns:
    net: output of the Convolution operations
  """

  with tf.variable_scope("layer{}".format(name)):
    for i, F in enumerate(n_filters):
	  net = tf.layers.conv3d(input_, F, kernel_size = [3, 3, 3], activation=None, strides=(1, 1, 1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg), name="conv_{}".format(i + 1))
	  net = tf.layers.dropout(net, 0.1, training=training)
	  net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
	  net = activation(net, alpha=0.1, name="relu{}_{}".format(name, i + 1))
	  #net = net + input_
    if pool is False:
      return net

def conv_down(input_, n_filters, training, flags, name, pool=False, activation=tf.nn.leaky_relu):
  """{Conv -> BN -> RELU}

  Args:
    input_ (4-D Tensor): (batch_size, H, W, C)
    n_filters (list): number of filters [int, int]
    training (1-D Tensor): Boolean Tensor
    name (str): name postfix
    pool (bool): If True, MaxPool2D
    activation: Activaion functions

  Returns:
    net: output of the Convolution operations
  """
  with tf.variable_scope("layer{}".format(name)):
    for i, F in enumerate(n_filters):
      net = tf.layers.conv3d(input_, F, kernel_size = [3, 3, 8], activation=None, strides=(2, 2, 1), padding='valid', kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg), name="conv_{}".format(i + 1))
      net = tf.layers.dropout(net, 0.1, training=training)
      net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
      net = activation(net, alpha=0.1, name="relu{}_{}".format(name, i + 1))

    if pool is False:
      return net
#    pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name)) # no pooling
#    return net, pool

def deconv_concat(inputA, input_B, n_filters, training, flags, name, activation=tf.nn.leaky_relu):
  """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (5-D Tensor): (N, H, W, Z, C)
        input_B (5-D Tensor): (N, 2*H, 2*H, Z, C2)
        name (str): name of the concat operation
    Returns: output (5-D Tensor): (N, 2*H, 2*W, C + C2)"""
  with tf.variable_scope("layer{}".format(name)):
    for i, F in enumerate(n_filters):
      net = tf.layers.conv3d_transpose(inputA, F, kernel_size = (3,3,8), activation = None, strides=(2,2,1), padding='valid', use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg), name="deconv_{}".format(i + 1))
      net = tf.layers.dropout(net, 0.1, training=training)
      net = tf.contrib.layers.bias_add(net)
      net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
      net = activation(net, alpha=0.1, name="relu{}_{}".format(name, i + 1))
    return tf.concat([net, input_B], axis=-1, name="concat_{}".format(i+1)) # concatenation
        
def make_unet(X, training, flags):# flags=None
  """Build a U-Net architecture

  Args:
    X (5-D Tensor): (N, H, W, Z, C)
    training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

  Returns:
    output (5-D Tensor): (N, H, W, Z, C)
      Same shape as the `input` tensor

  Notes:
  """
  print('size of {}: {}_{}_{}'.format('X', X.shape[1], X.shape[2], X.shape[3]))
  
  conv1 = conv(X, [32], training, flags, name=1)
  print('size of {}: {}_{}_{}'.format('conv1', conv1.shape[1], conv1.shape[2], conv1.shape[3]))
  conv2 = conv_down(conv1, [64], training, flags, name=2)
  print('size of {}: {}_{}_{}'.format('conv2', conv2.shape[1], conv2.shape[2], conv2.shape[3]))
  
  conv3 = conv(conv2, [64], training, flags, name=3)
  print('size of {}: {}_{}_{}'.format('conv3', conv3.shape[1], conv3.shape[2], conv3.shape[3]))
  conv4 = conv_down(conv3, [128], training, flags, name=4)
  print('size of {}: {}_{}_{}'.format('conv4', conv4.shape[1], conv4.shape[2], conv4.shape[3]))
  
  conv5 = conv(conv4, [128], training, flags, name=5)
  print('size of {}: {}_{}_{}'.format('conv5', conv5.shape[1], conv5.shape[2], conv5.shape[3]))
  conv6 = conv_down(conv5, [256], training, flags, name=6)
  print('size of {}: {}_{}_{}'.format('conv6', conv6.shape[1], conv6.shape[2], conv6.shape[3]))

  conv7 = conv(conv6, [256], training, flags, name=7)
  print('size of {}: {}_{}_{}'.format('conv7', conv7.shape[1], conv7.shape[2], conv7.shape[3]))
  conv8 = conv_down(conv7, [512], training, flags, name=8)
  print('size of {}: {}_{}_{}'.format('conv8', conv8.shape[1], conv8.shape[2], conv8.shape[3]))
  
  dconv7 = deconv_concat(conv8, conv7, [256], training, flags, name=9)
  print('size of {}: {}_{}_{}'.format('dconv7', dconv7.shape[1], dconv7.shape[2], dconv7.shape[3]))
  dconv6 = conv(dconv7, [256], training, flags, name=10)
  print('size of {}: {}_{}_{}'.format('dconv6', dconv6.shape[1], dconv6.shape[2], dconv6.shape[3]))
  
  dconv5 = deconv_concat(dconv6, conv5, [128], training, flags, name=11)
  print('size of {}: {}_{}_{}'.format('dconv5', dconv5.shape[1], dconv5.shape[2], dconv5.shape[3]))
  dconv4 = conv(dconv5, [128], training, flags, name=12)
  print('size of {}: {}_{}_{}'.format('dconv4', dconv4.shape[1], dconv4.shape[2], dconv4.shape[3]))
  
  dconv3 = deconv_concat(dconv4, conv3, [64], training, flags, name=13)
  print('size of {}: {}_{}_{}'.format('dconv3', dconv3.shape[1], dconv3.shape[2], dconv3.shape[3]))
  dconv2 = conv(dconv3, [64], training, flags, name=14)
  print('size of {}: {}_{}_{}'.format('dconv2', dconv2.shape[1], dconv2.shape[2], dconv2.shape[3]))
  
  dconv1 = deconv_concat(dconv2, conv1, [32], training, flags, name=15)
  print('size of {}: {}_{}_{}'.format('dconv1', dconv1.shape[1], dconv1.shape[2], dconv1.shape[3]))
  
  dconv0 = tf.layers.conv3d(
        dconv1,
        32,
        kernel_size = (3, 3, 3),
        activation = None,
        strides=(1, 1, 1),
        padding='same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        name="residual")
  print('size of {}: {}_{}_{}'.format('dconv0', dconv0.shape[1], dconv0.shape[2], dconv0.shape[3]))
  return tf.add(dconv0, X, name="final") # skip connection

def percent_error(y_pred, y_true, mask_):
  """
	  Args:
    y_pred (5-D array): (N, H, W, Z, 32)
    y_true (5-D array): (N, H, W, Z, 32)
    mask_ (5-D array): (N, H, W, Z, 1)
  """

  H, W, Z, _ = y_pred.get_shape().as_list()[1:]
  pred_flat = tf.reshape(y_pred, [-1, H * W* Z*32])
  true_flat = tf.reshape(y_true, [-1, H * W* Z*32])
  m_flat = tf.reshape(mask_, [-1, H, W, Z, 1])
  m_flat = tf.concat([m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat, m_flat], axis=-1);
  m_flat = tf.reshape(m_flat, [-1, H * W* Z*32])

  pred_flat = pred_flat * m_flat
  true_flat = true_flat * m_flat

  numerator = tf.norm(pred_flat - true_flat) 
  denominator = tf.norm(true_flat)

  return numerator / denominator
  
def MSE_(y_pred, y_true, mask_):
  """Returns a MSE score
  Args:
    y_pred (5-D array): (N, H, W, Z, 1)
    y_true (5-D array): (N, H, W, Z, 1)

  Returns:
    float: MSE Score
  """
  H, W, Z, _ = y_pred.get_shape().as_list()[1:]
  mask2 = tf.reshape(mask_, [-1, H, W, Z, 1])
  mask2 = tf.concat([mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2, mask2], axis=-1);
  sub = tf.subtract(y_pred, y_true)
  sub_mask = tf.multiply(sub, mask2) 
  return tf.nn.l2_loss(sub_mask)
  
def make_train_op(y_pred, y_true, mask_, x, rho, rho_low, ACS_mask):
  """Returns a training operation

  Loss function = MSE_

  Args:
    y_pred (5-D Tensor): (N, H, W, Z, 1)
    y_true (5-D Tensor): (N, H, W, Z, 1)

  Returns:
    train_op: minimize operation
  """
  loss1 = MSE_(y_pred,  y_true, mask_)
  
  global_step = tf.train.get_or_create_global_step()
  # rate = tf.train.exponential_decay(0.005, global_step, 1, 0.99975)
  optim = tf.train.AdamOptimizer() 
  return optim.minimize(loss1, global_step=global_step)


def read_flags():
  """Returns flags"""

  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs",
            default=1,
            type=int,
            help="Number of epochs (default: 1)")

  parser.add_argument("--batch-size",
            default=4,
            type=int,
            help="Batch size (default: 4)")

  parser.add_argument("--logdir",
            default="output/logdir",
            help="Tensorboard log directory (default: logdir)")

  parser.add_argument("--ckdir",
            default="output/models",
            help="Checkpoint directory (default: models)")

  parser.add_argument("--test",
            default=False,
            type=bool,
            help="It's obvious")
            
  parser.add_argument(
      "--debug",
      type=bool,
      nargs="?",
      const=True,
      default=False,
      help="Use debugger to track down bad values during training. "
           "Mutually exclusive with the --tensorboard_debug_address flag.")

  parser.add_argument("--reg",
            default=0.0,
            type=float,
            help="L2 Regularization")
            
  parser.add_argument(
      "--tensorboard_debug_address",
      type=bool,
      nargs="?",
      const=True,
      default=False,
      help="Use tensorboard_debug_address to track down bad values during training. "
           "Mutually exclusive with the --debug flag.")

  flags = parser.parse_args()
  return flags

def get_batch():
  return 0

def main(flags):
  test_file_name = "testing_data_cross1.csv" #  file contain testing data path
  train_file_name = "training_data_cross1.csv" # file contain training data path 

  train = pd.read_csv(train_file_name)
  n_train = train.shape[0] # find the number of training data

  test = pd.read_csv(test_file_name)
  n_test = test.shape[0]

  current_time = time.strftime("%m/%d/%H/%M/%S")
  train_logdir = os.path.join(flags.logdir, "train", current_time)
  test_logdir = os.path.join(flags.logdir, "test", current_time)

  tf.reset_default_graph()
  X = tf.placeholder(tf.float32, shape=[None, 159, 159, 36, 32], name="X") 
  mask_ = tf.placeholder(tf.float32, shape=[None, 159, 159, 36, 1], name="mask") 
  y = tf.placeholder(tf.float32, shape=[None, 159, 159, 36, 32], name="y")
  mode = tf.placeholder(tf.bool, name="mode")

  pred = make_unet(X, mode, flags)
  
  pred_residual = (pred - X)
  tf.add_to_collection("inputs", X)
  tf.add_to_collection("inputs", mask_)
  tf.add_to_collection("inputs", mode)
  tf.add_to_collection("outputs", pred)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

  with tf.control_dependencies(update_ops):
    train_op = make_train_op(pred, y, mask_, X)

  MSE_op = MSE_(pred, y, mask_)
  MSE_op = tf.Print(MSE_op, [MSE_op]) # print the MSE loss
  tf.summary.scalar("MSE", MSE_op)
  
  relative_error_op = percent_error(pred, y, mask_)
  relative_error_op = tf.Print(relative_error_op, [relative_error_op])
  tf.summary.scalar("Relative_error", relative_error_op)

  train_csv = tf.train.string_input_producer([train_file_name])
  test_csv = tf.train.string_input_producer([test_file_name])
 
  train_X, train_Y, train_mask, train_path = get_X_Y_mask(train_csv, augmentation=True)
  test_X, test_Y , test_mask, test_path = get_X_Y_mask(test_csv, augmentation=False)

  X_batch_op,  y_batch_op, mask_batch_op, path_batch_op = tf.train.shuffle_batch([train_X, train_Y, train_mask, train_path],
                          batch_size=flags.batch_size,
                          capacity=flags.batch_size * 30,
                          min_after_dequeue=flags.batch_size*5,
                          allow_smaller_final_batch=True)

  X_test_op, y_test_op, mask_test_op, path_test_op = tf.train.batch([test_X, test_Y, test_mask, test_path],
                      batch_size=flags.batch_size,
                      capacity=flags.batch_size * 10,
                      allow_smaller_final_batch=False)
                      
  summary_op = tf.summary.merge_all()
  sess = tf.Session()
  
  if flags.debug:
	sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  elif flags.tensorboard_debug_address:
	sess = tf_debug.TensorBoardDebugWrapperSession(sess, flags.tensorboard_debug_address)
	
  train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
  test_summary_writer = tf.summary.FileWriter(test_logdir)

  init = tf.global_variables_initializer()
  sess.run(init)
  
  saver = tf.train.Saver()

  if os.path.exists(flags.ckdir) and tf.train.checkpoint_exists(flags.ckdir):
		latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
		if latest_check_point != None:
		  saver.restore(sess, latest_check_point)
  else:
		try:
			os.rmdir(flags.ckdir)
		except IOError: # FileNotFoundError not existing in Python 2.7
			pass
		os.mkdir(flags.ckdir)

  try:
      global_step = tf.train.get_global_step(sess.graph)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      errors = []
      for epoch in range(flags.epochs):
        if flags.test == True: # or epoch % 100 == 99: # testing while training
          print "Begin Testing!!"
          #########################################################################################    Testing
          output_volume = np.zeros([159,159,36,32]) # Nx, Ny, Nz, Ncoil, Ns
          print "Finish Creating output matrix!!"
          print "Number of test dataset: {}".format(n_test)
          for num_b in range(0, n_test+1, flags.batch_size):
			print "num_b: {}".format(num_b)
			X_batch, y_batch, mask_batch, path_batch = sess.run([X_test_op, y_test_op, mask_test_op, path_test_op])
			print "Finish Loading test data in num_b {}".format(num_b)
			X_batch = np.reshape(X_batch, (-1,159,159,36,32))
			y_batch = np.reshape(y_batch, (-1,159,159,36,32))
			mask_batch = np.reshape(mask_batch, (-1,159,159,36,1))
			if num_b%1 ==0:
				input_X, true_y, predicted_y, predicted_residual = sess.run([X, y, pred, pred_residual],
				feed_dict={X: X_batch,
				           y: y_batch,
				           mode: True})
				for ii in xrange(flags.batch_size):
					index = re.findall(r'\d+',path_batch[ii]) # find subject index number
					print "subject {}".format(index[0])
					sub_num = int(index[0])
					output_volume[:,:,:,:] = predicted_y[ii,:,:,:,:]
					img = nib.Nifti1Image(output_volume, np.eye(4))
					nib.save(img, os.path.join('output','nii/test_Sensitivity_prediction_'+str(sub_num)+'_3D.nii.gz'))# save output prediction
          
        else:
          for num_b in range(0, n_train, flags.batch_size):
            X_batch,  y_batch, mask_batch, path_batch = sess.run([X_batch_op, y_batch_op, mask_batch_op, path_batch_op])
            X_batch = np.reshape(X_batch, (-1,159,159,36,32))
            y_batch = np.reshape(y_batch, (-1,159,159,36,32))
            mask_batch = np.reshape(mask_batch, (-1,159,159,36,1))
            _, step_mse, step_summary, global_step_value = sess.run([train_op, MSE_op, summary_op, global_step], 
            feed_dict={X: X_batch, 
                      y: y_batch, 
                      mask_: mask_batch,
                      mode: True})
            train_summary_writer.add_summary(step_summary, global_step_value)
            
        if not os.path.exists(flags.ckdir):
           os.mkdir(flags.ckdir)
        if not flags.test:
          pass
          if epoch % 500 == 499:
            saver.save(sess, "{}/model.ckpt".format(flags.ckdir))
 
  finally:
      coord.request_stop()
      coord.join(threads)
      if not flags.test:
        pass
        saver.save(sess, "{}/model.ckpt".format(flags.ckdir))
 
if __name__ == '__main__':
  flags = read_flags()
  main(flags)
