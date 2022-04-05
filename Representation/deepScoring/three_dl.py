from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import numpy
import subprocess
import os
import random
from pathlib import Path

from six.moves import xrange
import tensorflow as tf

import load_data
import model
import config
import stat_casp

# Basic model parameters as external flags.
FLAGS = None
log_file = None


def placeholder_inputs(batch_size):
  maps_placeholder = tf.placeholder(tf.float32, shape=(batch_size, model.GRID_VOXELS * model.NB_TYPE),
                                    name='main_input')
  ground_truth = tf.placeholder(tf.float32, shape=(batch_size))
  is_training = tf.placeholder(tf.bool, name='is_training')
  return maps_placeholder, ground_truth, is_training


def fill_feed_dict(data_set, maps_pl, gt_pl, is_training, training=True, batch_size_=50, selec_res=-1):
  maps_feed, gt_feed = data_set.next_batch(batch_size_, shuffle=True, select_residue=selec_res)
  feed_dict = {
    maps_pl: maps_feed,
    gt_pl: (gt_feed[:, 2] / 1000000),
    is_training: training
  }
  return feed_dict


def run_training(test=None, strategy="active", nb_prot_batch=1, selec_res = 83):
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # testfname = FLAGS.test_file
    # testDataset = load_data.read_data_set(testfname, dtype=dtypes.float16, seed = 1)

    # initialize a map generator variable with the configuration of the config file
    conf = config.load_config()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Generate placeholders for the maps and labels.
    maps_placeholder, ground_truth_placeholder, is_training = placeholder_inputs(None)

    # Build a Graph that computes predictions from the inference model.

    n_retype = 15
    logits, _, _, weightsLinear2 = model.scoringModel(n_retype, maps_placeholder, is_training, batch_norm=True,
                                                      validation='elu', final_activation='tanh')
    loss = model.loss(logits, ground_truth_placeholder)

    train_op = model.training(loss, FLAGS.learning_rate)

    variable_summaries(logits)

    variable_summaries(ground_truth_placeholder)

    with tf.name_scope('summaries'):
      tf.summary.scalar('loss', tf.reduce_mean(1000 * loss))

    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(conf.TENSORBOARD_PATH)
    writer.add_graph(sess.graph)

    # Create a saver for writing training checkpoints.
    print("Create saver ...")
    saver = tf.train.Saver(tf.global_variables())

    # And then after everything is built:

    # Run the Op to initialize the variables.

    if (os.path.isdir(FLAGS.restore)):

      print('Restore existing model: %s' % (FLAGS.restore))
      print('Latest checkpoint: %s' % (tf.train.latest_checkpoint(FLAGS.restore)))
      saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(FLAGS.restore) + '.meta')
      saver.restore(sess, tf.train.latest_checkpoint(FLAGS.restore))
      print("Model restored!")

    else:
      # Add the variable initializer Op.
      print("Initializing...")
      init = tf.global_variables_initializer()
      sess.run(init)
      print("Initialized!")

    trainDataset = None
    slidingloss = 0

    decayLoss = 0.999
    dln = 1

    # counts the number of updates of the weights
    nb_updates = 0
    lim = 0
    percentile_range = 1

    init = True

    for step in xrange(FLAGS.max_steps):
      # choose a file randomly

      for i_ in range(nb_prot_batch):
        pdbfile = None
        native = False
        while (pdbfile == None):

          caspNumber, casp_choice = choose_casp()

          # file in models
          if strategy == 'active':
            # increases the number of protein progressively starting with the  ones wich means is closest the the mean of the dataset

            if init or lim > 500:

              if init:
                if os.path.exists(conf.TEMP_META_PATH + FLAGS.data_variability + '_means.npy'):
                  print('Loading CASP stats...')
                  f = open(conf.TEMP_META_PATH + FLAGS.data_variability + '_means.npy', 'rb')
                  means_casp = numpy.load(f)
                  f.close()
                  print('Done !')
                else:
                  print('Computing CASP stats...')
                  means_casp = stat_casp.stat_full(conf.DATA_FILE_PATH, casp_choice)
                  f = open(conf.TEMP_META_PATH + FLAGS.data_variability + '_means.npy', 'wb')
                  print('Saving CASP stats...')
                  means_casp = numpy.save(f, means_casp)
                  f.close()
                  print('Done !')

              init = False
              a = numpy.percentile(means_casp, 100 - percentile_range)
              # b = numpy.percentile(means_casp, 50 + percentile_range)

              # print('Range : ', percentile_range)
              # print(a)
              # print(b)

              if percentile_range * 1.09 < 60:
                percentile_range = percentile_range * 1.09
              else:
                percentile_range = 60

              lim = 0

            mean_file = 0
            nb_file_ignored = -1
            loop = True

            start_time = time.time()
            while loop:
              nb_file_ignored += 1

              model_dir = random.choice(os.listdir(conf.DATA_FILE_PATH + caspNumber + "/MODELS"))
              model_file = random.choice(os.listdir(conf.DATA_FILE_PATH + caspNumber + "/MODELS/" + model_dir))
              if model_file[-4:] == ".sco":
                model_file = model_file[:-4]
              if os.path.isfile(conf.DATA_FILE_PATH + caspNumber + "/MODELS/" + model_dir + "/" + model_file + '.sco'):

                mean_file = numpy.mean(stat_casp.stat_file(
                  conf.DATA_FILE_PATH + caspNumber + "/MODELS/" + model_dir + "/" + model_file + '.sco'))
              else:
                mean_file = 0
              if a <= mean_file:
                loop = False
                print(mean_file)

            duration = time.time() - start_time
            print("Time to find file : %.3f - files ignored : " % duration, nb_file_ignored)
            print('Range : ', percentile_range / 1.09)
          else:
            model_dir = random.choice(os.listdir(conf.DATA_FILE_PATH + caspNumber + "/MODELS"))
            model_file = random.choice(os.listdir(conf.DATA_FILE_PATH + caspNumber + "/MODELS/" + model_dir))

          if model_file[-4:] == ".sco":
            model_file = model_file[:-4]

          score_file = Path(conf.DATA_FILE_PATH + caspNumber + "/MODELS/" + model_dir + "/" + model_file + ".sco")

          if score_file.is_file():
            pdbfile = conf.DATA_FILE_PATH + caspNumber + "/MODELS/" + model_dir + "/" + model_file
            native = False
            print(pdbfile)

        # create a density map of it

        start_time = time.time()
        if native:
          subprocess.call(
            [conf.ROTAMERS_EXE_PATH, "--mode", "map", "-i", pdbfile, "--native", "-m", "24", "-v", "0.8", "-o",
             conf.TEMP_PATH + '_' + str(i_) + '.bin'])
        else:
          subprocess.call([conf.ROTAMERS_EXE_PATH, "--mode", "map", "-i", pdbfile, "-m", "24", "-v", "0.8", "-o",
                           conf.TEMP_PATH + '_' + str(i_) + '.bin'])
        duration = time.time() - start_time
        print("Mapping duration : %.3f" % duration)
        start_time = time.time()
        trainDataset = load_data.read_data_set(conf.TEMP_PATH + '_' + str(i_) + '.bin', shuffle=True, prop=1)
        duration = time.time() - start_time
        print("Loading duration : %.3f" % duration)

        if trainDataset is None:
          continue;

        if i_ == 0:
          trainDataset_full = trainDataset
          print('New data set created with extracted maps')
        else:
          trainDataset_full.append(trainDataset)
          print('Extracted maps added to current data set')

      if selec_res == -1:
        for i in xrange(1 + trainDataset_full.num_res // FLAGS.batch_size):
          # Fill a feed dictionary with the actual set of maps and labels
          # for this particular training step.
          feed_dict = fill_feed_dict(trainDataset_full,
                                     maps_placeholder,
                                     ground_truth_placeholder,
                                     is_training, FLAGS.dropout, batch_size_=FLAGS.batch_size)  # 0.5

          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          start_time = time.time()
          if numpy.sum(feed_dict[ground_truth_placeholder]) > 1:
            dln = dln * decayLoss
            _, loss_value, res, s = sess.run([train_op, loss, logits, merged],
                                             feed_dict=feed_dict)
            nb_updates += 1
            lim += 1
            slidingloss = decayLoss * slidingloss + (1 - decayLoss) * numpy.mean(loss_value)

            print('Res - mean : %.4f - std : %.6f' % (numpy.mean(res), numpy.std(res)))
            print('Tru - mean : %.4f - std : %.6f' % (
              numpy.mean(feed_dict[ground_truth_placeholder]), numpy.std(feed_dict[ground_truth_placeholder])))

            writer.add_summary(s, nb_updates)

            # print(loss_value)
            # print(numpy.sum(loss_value))
            duration = time.time() - start_time
            print('Step %d: loss = %.4f corr =  (%.3f sec)' % (step, 1000 * slidingloss / (1 - dln), duration))
            # print(numpy.corrcoef(res, feed_dict[ground_truth_placeholder] ))

      else:
        epoch_before_batch = trainDataset_full._epochs_completed
        while epoch_before_batch == trainDataset_full._epochs_completed:
          # Fill a feed dictionary with the actual set of maps and labels
          # for this particular training step.
          feed_dict = fill_feed_dict(trainDataset_full,
                                     maps_placeholder,
                                     ground_truth_placeholder,
                                     is_training, FLAGS.dropout, batch_size_=FLAGS.batch_size, selec_res = selec_res)  # 0.5

          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          start_time = time.time()
          if numpy.sum(feed_dict[ground_truth_placeholder]) > 1:
            dln = dln * decayLoss
            _, loss_value, res, s = sess.run([train_op, loss, logits, merged],
                                             feed_dict=feed_dict)
            nb_updates += 1
            lim += 1
            slidingloss = decayLoss * slidingloss + (1 - decayLoss) * numpy.mean(loss_value)

            print('Res - mean : %.4f - std : %.6f' % (numpy.mean(res), numpy.std(res)))
            print('Tru - mean : %.4f - std : %.6f' % (
              numpy.mean(feed_dict[ground_truth_placeholder]), numpy.std(feed_dict[ground_truth_placeholder])))

            writer.add_summary(s, nb_updates)

            # print(loss_value)
            # print(numpy.sum(loss_value))
            duration = time.time() - start_time
            print('Step %d: loss = %.4f corr =  (%.3f sec)' % (step, 1000 * slidingloss / (1 - dln), duration))
            # print(numpy.corrcoef(res, feed_dict[ground_truth_placeholder] ))

      # Write the summaries and print an overview fairly often.
      if step % 10 == 0:
        save_path = saver.save(sess, conf.SAVE_PATH)
        print("Model saved in path: %s" % save_path)

      print('\n')
  # print('Done')
  # var = sess.run(weightsLinear2)
  # print(var)


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean, std = tf.nn.moments(var, axes=[0])
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('std', std)


def choose_casp():
  # chose a casp file in the data directory randomly depending on the different options you have in the data directory 

  if FLAGS.data_variability == 'file_one':
    casp_choice = ["dummy1"]
    return random.choice(casp_choice), casp_choice
  elif FLAGS.data_variability == 'files_same':
    casp_choice = ["dummy4"]
    return random.choice(casp_choice), casp_choice
  elif FLAGS.data_variability == 'files_different':
    casp_choice = ["dummy2"]
    caspNumber = random.choice(casp_choice), casp_choice
  elif FLAGS.data_variability == 'protein':
    casp_choice = ["dummy3"]
    return random.choice(casp_choice), casp_choice
  elif FLAGS.data_variability == 'protein_few':
    casp_choice = ["dummy5"]
    return random.choice(casp_choice), casp_choice
  elif FLAGS.data_variability == 'all':
    casp_choice = ["CASP7", "CASP8", "CASP9", "CASP10", "CASP11"]
    return random.choice(casp_choice), casp_choice
  elif FLAGS.data_variability == 'casp':
    casp_choice = ["CASP11"]
    return random.choice(casp_choice), casp_choice


def main(_):
  run_training()
  # conf = config.load_config()
  # means_casp12 = stat_casp.stat_casp('/home/benoitch/data/CASP12/MODELS/')
  # eval_model.run_eval('/home/benoitch/Temp/Model', '/home/benoitch/data/', numpy.percentile(means_casp12, 40), FLAGS.learning_rate, batch_s = 40)
  # eval_model.prepare_plot('/home/benoitch/save/nn_3/Model/', '/home/benoitch/data/')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.0001,
    help='Initial learning rate.'
  )
  parser.add_argument(
    '--dropout',
    type=float,
    default=0.5,
    help='Dropout rate.'
  )
  parser.add_argument(
    '--max_steps',
    type=int,
    default=25000,
    help='Number of steps to run trainer.'
  )
  parser.add_argument(
    '--conv_layers_size',
    type=int,
    nargs='*',
    dest='conv_size_list',  # store in 'list'.
    default=[],
    help='Number of output in each convolutional layer.'
  )
  parser.add_argument(
    '--full_layers_size',
    type=int,
    nargs='*',
    dest='full_size_list',  # store in 'list'.
    default=[],
    help='Number of units in each hidden layer.'
  )
  parser.add_argument(
    '--groupSize',
    type=int,
    default=12,
    help='Size of the group representing the same data.'
  )
  parser.add_argument(
    '--elementSize',
    type=int,
    default=1,
    help='Number of group of density map representing the same structure.'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=10,
    help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
    '--log_dir',
    type=str,
    default='/tmp/tensorflow/mnist/logs/fully_connected_feed2',
    help='Directory to put the log data.'
  )
  parser.add_argument(
    '--log_file',
    type=str,
    default='tmp.log',
    help='File to put human readable logs.'
  )
  parser.add_argument(
    '--restore',
    type=str,
    default='',  # '/tmp/tensorflow/mnist/logs/fully_connected_feed_',
    help='path to restore model from'
  )
  parser.add_argument(
    '--evaluate',
    default=False,
    help='train or evaluate the data'
  )
  parser.add_argument(
    '--training_data_path',
    default='',
    help='path to the training data directory'
  )
  parser.add_argument(
    '--test_file',
    default='',
    help='path to the test data file'
  )
  parser.add_argument(
    '--data_variability',
    default='all',
    help='Define the variability of data, could be only one version of a protein (file_one), a few versions of the same protein (files_same) or different proteins (files_different) one protein (protein), a few proteins (protein_few), a casp directory (casp) or all the data (all)'
  )
  FLAGS, unparsed = parser.parse_known_args()
  log_file = open(FLAGS.log_file, "w")
  print(FLAGS)
  print(unparsed)
  print(FLAGS, file=log_file)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
