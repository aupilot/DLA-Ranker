from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import os
import subprocess
import numpy as np
import argparse
import pickle


import config
import model_v2_routed as model_v2
import load_data
import stat_casp

"""
General notes :
# I suggest to name the tests following this convention : start with '_' then the id of the test (integer) '28'
  If you stop the training and want to start it again just name ad '_' and an integer equals to the number of time you've
  restarted the training.
  eg. Initial training : test = '_28'
        restarded      : test = '_28_1'
        restarted again: test = '_28_2' etc...
    
  To plot the loss curves just moove the loss files, into a specific directory, change the compare_losses.py variable 
  so that it points to that directory and then call:
  python3 compare_losses.py -t [list of ids that you want to plot, eg: 0 12 28] 
"""

conf = config.load_config()
FLAGS = None
log_file = None

class Strategy(object):
    """
    A strategy is an object that is used to provide the model with data during the training phase.
    Its role is to choose files from the database according to a training strategy (for instance: choose the file
    among the top 1% (in terms of ground truth cad score), every 200 steps increase this range by a factor 1.05)
    and it then gives it to the mapper.
    """

    def __init__(self, casp_possibilities, range_init = 1, update_every = 200, update_speed = 1.09, range_max = 65, use_ls_file = None):
        """
        :param casp_possibilities: String that is given to choose_casp function and that will then determined the directories
                                   from where the data is taken (eg. 'all' -> CASP7 to CASP11, see choose_casp())
        :param range_init: For strategy1 initial range. See Strategy1().
        :param update_every: Number of batch to wait before updating the range
        :param update_speed: Range updated by this factor
        :param range_max: range updates stop after reaching this level
        :param use_ls_file: For strategy feeding_order_strategy() : path to feeding order file.
        """

        # loads a distributions of the gt scores for the data set. if it exists
        if os.path.exists(conf.TEMP_META_PATH + casp_possibilities + '_means.npy'):
            print('Loading CASP stats...')
            f = open(conf.TEMP_META_PATH + casp_possibilities + '_means.npy', 'rb')
            self.means_casp = np.load(f)
            f.close()
            print('Done !')
        # if it does not, computes it with stat_casp file. And then saves it
        else:
            print('Computing CASP stats...')
            _, casp_list = self.choose_casp(casp_possibilities)
            self.means_casp = stat_casp.stat_full(conf.DATA_FILE_PATH, casp_list)
            f = open(conf.TEMP_META_PATH + casp_possibilities + '_means.npy', 'wb')
            print('Saving CASP stats...')
            np.save(f, self.means_casp)
            f.close()
            print('Done !')

        self.casp_choice = casp_possibilities
        self.range = range_init
        self.update = update_every
        self.count = 0
        self.update_factor = update_speed
        self.lim_inf = range_max
        self.use_ls_file = use_ls_file
        # indix of the line
        self.file_i = 0
         # For feeding_order_strategy, loads the file and then reads its line : each line is the path to a protein
        if use_ls_file is None:
            self.ls_file = []
        else:
            f = open(use_ls_file,'r')
            self.ls_file = f.readlines()

        for i,l in enumerate(self.ls_file):
            self.ls_file[i] = l[:-1]

        print(self.ls_file)
        f.close()

    def update_range(self):
        """
        Updates the range variable every self.update times it is called, up to self.lim_inf
        :return:
        """
        self.count += 1
        if self.count > self.update:
            self.count = 0
        if self.range*self.update_factor < self.lim_inf:
            self.range *= self.update_factor
        else:
            self.range = self.lim_inf

    def choose_file(self):
        """
        Choose a file randomly, check if this file suits the strategy. Once it finds one that does returns is
        :return: path to the chosen file
        """
        pdb = None
        nb_file_ignored = -1

        while pdb is None:
            nb_file_ignored += 1

            caspNumber, casp_choice = self.choose_casp(self.casp_choice)

            prot_path = conf.DATA_FILE_PATH + caspNumber + '/MODELS/'

            prot_dir = random.choice(os.listdir(prot_path))
            prot_file = random.choice(os.listdir(prot_path + prot_dir))

            if prot_file[-4:] == ".sco":
                prot_file = prot_file[:-4]

            prot_dir_path = prot_path + prot_dir + '/'
            prot_file_path = prot_dir_path + prot_file

            score_file = prot_file_path + ".sco"

            if os.path.exists(score_file):
                pdb, sco = self.check_file(prot_file_path)
            else:
                continue

            if pdb is None:
                continue
        print('Chose file %s according to strategy.' % pdb)
        print('Nb files ignored : ', nb_file_ignored)
        print('File quality : ', sco)
        return pdb

    def check_file(self, model_file_path):
        """
        Check if one file correspond to the object's strategy
        :param model_file_path: path to a file (pdb format)
        :return: String if the file fits the strategy, None otherwise.
        """
        # returns none if the file chosen doesn't correspond match the training strategy at that point of time
        # if it does returns the file itself

        sco_file = np.mean(stat_casp.stat_file(model_file_path + '.sco'))

        if self.is_ok(sco_file):
            return model_file_path, sco_file
        print('None')
        return None, None

    def choose_casp(self, var):
        """
        Randomly chooses a CASP directory according to the variability keyword
        This directory must have one MODEL/ subdir, with subdirs for every protein (eg. MODELS/T0854/) in which are saved
        the proteins files in pdb format
        :param var: Keyword
        :return: String, name of the chosen directory
        """
        # chose a casp file in the data directory randomly depending on the different options you have in the data directory
        if var == 'file_one':
            casp_choice = ["dummy1"]
            return random.choice(casp_choice), casp_choice
        elif var == 'files_same':
            casp_choice = ["dummy4"]
            return random.choice(casp_choice), casp_choice
        elif var == 'files_different':
            casp_choice = ["dummy2"]
            return random.choice(casp_choice), casp_choice
        elif var == 'protein':
            casp_choice = ["dummy3"]
            return random.choice(casp_choice), casp_choice
        elif var == 'protein_few':
            casp_choice = ["dummy5"]
            return random.choice(casp_choice), casp_choice
        elif var == 'all':
            # casp_choice = ["CASP7", "CASP8", "CASP9", "CASP10", "CASP11"]
            casp_choice = ["CASP7", "CASP8", "CASP9", "CASP10"]
            return random.choice(casp_choice), casp_choice
        elif var == 'one_casp':
            casp_choice = ["CASP11"]
        return random.choice(casp_choice), casp_choice

class Passive(Strategy):
    """
    No discrimination between the files.
    """
    def is_ok(self, sco_file):
        return True

class Active_1(Strategy):
    """
    Selects the file among the top self.range % (ground truth cad score) of the data set.
    Every self.update batch, updates the range by a factor self.update_factor up to self.lim_inf
    """
    def is_ok(self, sco_file):
        a = np.percentile(self.means_casp, 100 - self.range)
        if sco_file > a:
            return True
        return False

class Active_2(Strategy):
    """
    Selects the file among the top self.lim_inf %. Constant range.
    """
    def is_ok(self, sco_file):
        a = np.percentile(self.means_casp, 100 - self.lim_inf)

        if sco_file > a:
            return True
        return False

class feeding_order_strategy(Strategy):
    """
    Selects the file following the order specified in the feeding_order file specified at the creation of the object
    """
    def choose_file(self):
        model_file_path = self.ls_file[self.file_i]
        print(model_file_path)
        self.file_i += 1
        if self.file_i == len(self.ls_file):
            print('Run out of files')
            return 'out_of_file'
        return model_file_path

    def is_ok(self, sco_file):
        return True

def placeholder_inputs(batch_size, coarse=False):
    """
    Creates placeholder for the input data during the training of the model
    :param batch_size: integer
    :param coarse: True if mapping coarse grained
    :return: Placesholders
    """
    nbDimTypes  = FLAGS.types
    if coarse :
        nbDimTypes *=6
    maps_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.voxels * FLAGS.voxels * FLAGS.voxels * nbDimTypes),
                                      name='main_input')
    ground_truth = tf.placeholder(tf.float32, shape=batch_size, name = 'gt_pl')
    is_training = tf.placeholder(tf.bool, name='is_training')
    meta_placeholder = tf.placeholder(tf.float32, shape=(batch_size,16), name='meta_pl')

    return maps_placeholder, ground_truth, is_training, meta_placeholder



def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean, std = tf.nn.moments(var, axes=[0])
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('std', std)

def fill_feed_dict(data_set, maps_pl, meta_pl, gt_pl, is_training, training=True, batch_size_=50):
    """
    :param data_set:    |
    :param maps_pl:     |
    :param meta_pl:     | Data placeholders
    :param gt_pl:       |
    :param is_training: |
    :param training: True if the model is training
    :param batch_size_: integer
    :return: dictionnary to give to sess.run(.., feed_dict = )
    """
    maps_feed, gt_feed = data_set.next_batch(batch_size_, shuffle=True)

    feed_dict = {
        maps_pl: maps_feed,
        meta_pl: gt_feed,
        gt_pl: (gt_feed[:, 2] / 1000000),
        is_training: training
    }


    return feed_dict

def train_model(feeding_strategy, restore=None, conv_param=None, max_step=20000, batch_size=10, learning_rate=0.001, nb_branches=1, nb_voxel=24, size_voxel=0.8, router='base', test='', coarse='', useMeta = 0):
    """
    Trains the model
    :param feeding_strategy: Strategy object
    :param restore: Path to the model to restore
    :param conv_param: Conv_params object from model_v2_routed
    :param max_step: Number of steps to train the model
    :param batch_size: integer
    :param learning_rate: float
    :param nb_branches: Number of branches for the model
    :param nb_voxel: Number of voxel for mapping
    :param size_voxel: Size in angstrom of one voxel for mapping
    :param router: Type of router (eg. 'advanced', 'routed_res)
    :param test: string, test identifier, used as appendix for the file's name created by the script
    :param coarse: True if coarse grained mapping
    :return: ...
    """
    with tf.Graph().as_default():
        sess = tf.Session()

        # if the model is not restored, it is build using routed_model_v2
        if restore is None:
            print('Model needs to be build...')
            coarseParam = ( coarse == 'coarse' )

            scoring_nn = model_v2.ScoringModel(conv_param, num_retype=15, GRID_SIZE=nb_voxel, activation='elu', final_activation='tanh', nb_branches=nb_branches, router=router, coarse = coarseParam, types = FLAGS.types, useMeta = useMeta)

            maps_placeholder, ground_truth_placeholder, is_training, meta_placeholder = placeholder_inputs(None, coarse=coarseParam)

            logits = scoring_nn.get_pred(maps_placeholder, meta_placeholder, is_training)

            loss = scoring_nn.compute_loss(logits, ground_truth_placeholder)

            train_op = scoring_nn.train(loss, learning_rate)

            # adding all the summaries
            variable_summaries(logits)
            variable_summaries(ground_truth_placeholder)
            with tf.name_scope('summaries'):
                tf.summary.scalar('loss', tf.reduce_mean(1000 * loss))

            merged = tf.summary.merge_all()

            print(merged.name)

            writer = tf.summary.FileWriter(conf.TENSORBOARD_PATH)
            writer.add_graph(sess.graph)

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.global_variables())

            # initializing variables
            init = tf.global_variables_initializer()
            sess.run(init)

        else:

            print('Restore existing model: %s' % (restore))
            print('Latest checkpoint: %s' % (tf.train.latest_checkpoint(restore)))

            saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(restore) + '.meta')

            saver.restore(sess, tf.train.latest_checkpoint(restore))

            graph = tf.get_default_graph()

            meta_placeholder = graph.get_tensor_by_name('meta_pl:0')
            maps_placeholder = graph.get_tensor_by_name('main_input:0')
            ground_truth_placeholder = graph.get_tensor_by_name('gt_pl:0')
            is_training = graph.get_tensor_by_name('is_training:0')
            logits = graph.get_tensor_by_name("main_output:0")
            loss = graph.get_tensor_by_name("loss:0")
            train_op = graph.get_tensor_by_name('train_op:0')
            merged = graph.get_tensor_by_name("Merge/MergeSummary:0")

            writer = tf.summary.FileWriter(conf.TENSORBOARD_PATH)

        if not router == 'routed_res':
            graph = tf.get_default_graph()
            router = graph.get_tensor_by_name("out_router_1:0")

        updates = 0
        slidingloss = 0
        decayLoss = 0.999
        dln = 1
        losses = []

        while updates < max_step:
            trainDataset = None
            while trainDataset is None:
                # Choosing a file
                pdb_file = feeding_strategy.choose_file()


                if pdb_file == 'out_of_file':

                    # this means we've reach the end of the feeding_order file
                    updates = max_step + 10
                    trainDataset = 'out_of_file'
                    continue
                print('Mapping pdb file...')
                mapcommand = [conf.ROTAMERS_EXE_PATH, "--mode", "map", "-i", pdb_file, "-m", str(nb_voxel), "-t", str(FLAGS.types), "-v",
                            str(size_voxel), "-o",  # v 0.8
                            conf.TEMP_PATH + str(test) + '.bin']
                if FLAGS.orient == 0:
                    mapcommand.append("--orient")
                    mapcommand.append("0")
                if FLAGS.skipNeighb != 0:
                    mapcommand.append("--skip_neighb")
                    mapcommand.append(str(FLAGS.skipNeighb))
                if coarse == 'coarse':
                    mapcommand.append("--coarse")
                subprocess.call(mapcommand)
                print('Loading file...')

                trainDataset = load_data.read_data_set(conf.TEMP_PATH + test + '.bin', shuffle=True)

                if trainDataset is None:
                    print('Mapping or loading went wrong ; choosing another file...')
                    continue
            if pdb_file == 'out_of_file':
                # this means we've reach the end of the feeding_order file, breaks the while loop
                continue
            # If we're not already following a feeding order_file : adds the chosen file to the list
            if feeding_strategy.use_ls_file is None:
                f = open(conf.LS_TRAINING_FILE + 'feeding_order', 'a')
                f.write(pdb_file + '\n')

            # Split the protein into different batches
            for i in range(1 + trainDataset.num_res // batch_size):
                try :
                    feed_dict = fill_feed_dict(trainDataset,
                                   maps_placeholder,
                                   meta_placeholder,
                                   ground_truth_placeholder,
                                   is_training, batch_size_= batch_size)

                    if not np.sum(feed_dict[ground_truth_placeholder]) > 1:
                        print('Not relevant batch : ignore')
                        continue

                    dln = dln * decayLoss
        
                    # print(np.array(sess.run([tocheck], feed_dict=feed_dict)).shape)
                    print(maps_placeholder.shape)
                    _, loss_value, result, s = sess.run([train_op, loss, logits, merged], feed_dict=feed_dict)



                except ValueError as e:
                    print(e)
                    print(ValueError)
                    print("Error appened in this batch")
                    continue

                if updates % 100 == 0:
                    if router == 'routed_res':
                        a = sess.run(meta_placeholder, feed_dict=feed_dict)
                        print(a[:,1])
                    else:
                        r = sess.run(router, feed_dict=feed_dict)
                        print(r)

                updates += 1
                feeding_strategy.update_range()
                slidingloss = decayLoss * slidingloss + (1 - decayLoss) * np.mean(loss_value)

                print('Res - mean : %.4f - std : %.6f' % (np.mean(result), np.std(result)))
                print('Tru - mean : %.4f - std : %.6f' % (np.mean(feed_dict[ground_truth_placeholder]), np.std(feed_dict[ground_truth_placeholder])))

                writer.add_summary(s, updates)

                print('Step %d: loss = %.4f' % (updates, 1000 * slidingloss / (1 - dln)))

                losses.append(1000 * slidingloss / (1 - dln))

                if updates % 100 == 0:
                    if not os.path.exists(conf.SAVE_DIR + test):
                        os.makedirs(conf.SAVE_DIR + test)

                    save_path = saver.save(sess, conf.SAVE_DIR + test + '/model.ckpt')
                    print("Model saved in path: %s" % save_path)

                    f__ = open(conf.LS_TRAINING_FILE + 'losses'+ test + '.npy', 'wb')
                    np.save(f__,losses)

                    f__.close()
                if updates % 5000 == 0:

                    if not os.path.exists(conf.SAVE_DIR + test +'_5000' ):
                        os.makedirs(conf.SAVE_DIR + test + '_5000' )

                    save_path = saver.save(sess, conf.SAVE_DIR + test + '_5000' + '/model_'+ str(updates)+'.ckpt')
                    print("Model saved in path: %s" % save_path)
                    print('\n\n')

def main():
    # use_ls_file = conf.LS_TRAINING_FILE

    with open(FLAGS.conv, 'rb') as input:
      conv_params = pickle.load(input)

    strategy = Passive('all',range_init = 3, update_every = 200, update_speed = 1.1, range_max = 90, use_ls_file=FLAGS.feeding_order)
    if not FLAGS.feeding_order is None:
        strategy = feeding_order_strategy('all', range_init=3, update_every=200, update_speed=1.1, range_max=90,
                           use_ls_file=FLAGS.feeding_order)

    train_model(strategy, restore=FLAGS.restore,conv_param=conv_params, max_step=FLAGS.steps, batch_size=FLAGS.batch_size,
              learning_rate=FLAGS.learning_rate, nb_branches=FLAGS.branches, nb_voxel=FLAGS.voxels,
              size_voxel=FLAGS.size, router=FLAGS.router, test = FLAGS.test, coarse=FLAGS.map, useMeta = FLAGS.useMeta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--steps',
        type=int,
        default=102000,
        help='Number of steps of training'
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=40,
        help='Size of one batch'
    )
    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        default=0.0001,
        help='learning rate'
    )
    parser.add_argument(
        '--branches',
        type=int,
        default=22,
        help='number of branches'
    )
    parser.add_argument(
        '--types',
        type=int,
        default=167,
        help='number of atom types'
    )
    parser.add_argument(
        '-v',
        '--voxels',
        type=int,
        default=24,
        help='number of voxel in the map'
    )
    parser.add_argument(
        '--size',
        type=float,
        default=0.8,
        help='Size of a vovel in the map'
    )
    parser.add_argument(
        '-t',
        '--test',
        type=str,
        default='',
        help='id of the test'
    )
    parser.add_argument(
        '-c',
        '--conv',
        type=str,
        help='path to the file where the conv setting are saved'
    )
    parser.add_argument(
        '-f',
        '--feeding_order',
        default=None,
        type=str,
        help='path to the file where the feeding order is saved'
    )
    parser.add_argument(
        '--restore',
        default=None,
        type=str,
        help='path to model'
    )
    parser.add_argument(
        '--orient',
        default=1,
        type=int,
        help='orientation 0 or 1 (1 is orientation)'
    )
    parser.add_argument(
        '--skipNeighb',
        default=0,
        type=int,
        help='number of neighbours to skip (default is zero)'
    )
    parser.add_argument(
        '--useMeta',
        default=0,
        type=int,
        help='useMeta information such as secondary structure and surface area (0 is no information, 1 is sec struct, 2 is surface area 3 is both)'
    )
    parser.add_argument(
        '--router',
        default='base',
        type=str,
        help='Define the kind of router you want (base, routed_res, advanced)'
    )
    parser.add_argument(
        '--map',
        default='',
        type=str,
        help='Set to coarse to have a coarse map'
    )

    FLAGS = parser.parse_args()
    main()


