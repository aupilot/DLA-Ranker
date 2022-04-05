from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# NUM_RETYPE =  15

# GRID_SIZE = 24
# GRID_VOXELS = GRID_SIZE * GRID_SIZE * GRID_SIZE
# NB_TYPE = 169


def _weight_variable(name, shape):
  return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.01))


def _bias_variable(name, shape):
  return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))


class ScoringModel:
  def __init__(self,
               num_retype=15,
               GRID_SIZE=24,
               NB_TYPE=169,
               batch_norm=True,
               validation='softplus',
               final_activation='sigmoid'
               ):

    self.num_retype = num_retype
    self.GRID_SIZE = GRID_SIZE
    self.GRID_VOXELS = self.GRID_SIZE * self.GRID_SIZE * self.GRID_SIZE
    self.NB_TYPE = NB_TYPE
    self.batch_norm = batch_norm
    self.validation = validation
    self.final_activation = final_activation

  def get_pred(self,
               maps,
               isTraining):

    print('Creating model...')



    input_data = tf.reshape(maps, [-1, self.NB_TYPE, self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE])

    # First step reducing data dimensionality

    retyped = self.retype_layer(input_data, self.num_retype, self.NB_TYPE, name='retype')

    # First convolution

    CONV1_OUT = 20

    out_conv_1 = self.conv_layer(retyped, [3, 3, 3, self.num_retype, CONV1_OUT], name='CONV_1')

    # batch norm and activation

    out_conv_1 = self.activation_normalization_layer(out_conv_1, self.batch_norm, self.validation, isTraining, name = 'act_norm_1')

    # Second convolution

    CONV2_OUT = 30

    out_conv_2 = self.conv_layer(out_conv_1, [4, 4, 4, CONV1_OUT, CONV2_OUT], name='CONV_2')


    # Batch norm and activation

    out_conv_2 = self.activation_normalization_layer(out_conv_2, self.batch_norm, self.validation, isTraining, name = 'act_norm_2')

    # Third convolution

    CONV3_OUT = 20

    out_conv_3 = self.conv_layer(out_conv_2, [4, 4, 4, CONV2_OUT, CONV3_OUT], name='CONV_3')

    out_conv_3 = self.activation_normalization_layer(out_conv_3, self.batch_norm, self.validation, isTraining, name = 'act_norm_3')

    # pooling and flattening

    POOL_SIZE = 4

    prev_layer = tf.nn.avg_pool3d(
      out_conv_3,
      [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
      [1, POOL_SIZE, POOL_SIZE, POOL_SIZE, 1],
      padding='VALID')

    NB_DIMOUT = 4 * 4 * 4 * CONV3_OUT

    flat0 = tf.reshape(prev_layer, [-1, NB_DIMOUT])

    # Fully connected layer 1

    LINEAR1_OUT = 64

    out_l1 = self.fc_layer(flat0, NB_DIMOUT, LINEAR1_OUT, bias=True, name='linear_1', num_retype=self.num_retype)

    out_l1 = self.activation_normalization_layer(out_l1, self.batch_norm, self.validation, isTraining, name = 'act_norm_4')

    out = self.fc_layer(out_l1, LINEAR1_OUT, 1, False, 'Linear_2', self.num_retype)

    out = tf.squeeze(out)

    if self.final_activation == 'tanh':
      return tf.add(tf.tanh(out) * 0.5, 0.5, name="main_output")
    else:
      return tf.sigmoid(out, name="main_output")

  def retype_layer(self, prev_layer, num_retype_, input_, name='retype'):

    retyper = _weight_variable(name + "_" + str(num_retype_), [input_, num_retype_])

    with tf.name_scope(name):
      tf.summary.histogram(name, retyper)

    prev_layer = tf.transpose(prev_layer, perm=[0, 2, 3, 4, 1])
    map_shape = tf.gather(tf.shape(prev_layer), [0, 1, 2, 3])  # Extract the first three dimensions

    map_shape = tf.concat([map_shape, [self.num_retype]], axis=0)
    prev_layer = tf.reshape(prev_layer, [-1, self.NB_TYPE])
    prev_layer = tf.matmul(prev_layer, retyper)

    return tf.reshape(prev_layer, map_shape)

  def conv_layer(self, prev_layer, kernel_size, name='CONV'):

    kernelConv = _weight_variable("weights_" + name + "_" + str(self.num_retype), kernel_size)
    prev_layer = tf.nn.conv3d(prev_layer, kernelConv, [1, 1, 1, 1, 1], padding='VALID', name = name)
    biasConv = _bias_variable("biases_" + name + "_" + str(kernel_size[3]), kernel_size[-1])

    with tf.name_scope(name):
      tf.summary.histogram("weights_" + name, kernelConv)
      tf.summary.histogram("biases_" + name, biasConv)

    return prev_layer + biasConv;


  def activation_normalization_layer(self, input_vector, batch_norm, validation, isTraining, name='act_norm_'):
    if batch_norm:
      input_vector = tf.layers.batch_normalization(input_vector, training=isTraining, name = name)

    if validation == 'softplus':
      return tf.nn.softplus(input_vector, name="softplus")
    else:
      return tf.nn.elu(input_vector, name="elu")


  def fc_layer(self, input_vector, input_size, output_size, bias, name, num_retype):

    weightsLinear = _weight_variable("weights_" + name + "_" + str(num_retype), [input_size, output_size])

    prev_layer = tf.matmul(input_vector, weightsLinear)
    if bias:
      biasLinear = _bias_variable("biases_" + name + "_" + str(num_retype), [output_size])

    with tf.name_scope(name):
      tf.summary.histogram("weights_" + name, weightsLinear)
      if bias:
        tf.summary.histogram("biases_" + name, biasLinear)
    if bias:
      return prev_layer + biasLinear
    else:
      return prev_layer

  def compute_loss(self, scores, cad_score):
    return tf.square(scores - cad_score, name='loss')

  def train(self, loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay = 0.999)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')

    return train_op