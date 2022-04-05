from numpy.core.multiarray import ndarray

import config
import subprocess
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import tensorflow as tf

import load_data
import model
import stat_casp
import argparse

CONF = config.load_config()
FLAGS = None

class model_prot():
    """
    Class that gathers informations about a protein models and that is used to
    score it in diffrent ways
    """
    def __init__(self, path_to_model, path_to_target=None):
        """
        :param path_to_model: Path to the generated protein file
        :param path_to_target: path to the protein's target's file
        """
        self._mod = path_to_model
        self._targ = path_to_target
        self._gdt = None
        self._gt_score = None
        self._pred = None
        self._gt_score_out = None

    def get_gdt(self):
        """
        Computes and returns the gdt score of a protein, using a subprocess.
        :return: gdt scores
        """
        if self._targ is None:
            print('ERROR: add a target to the model using .add_target(path_to_target)\n--> Return None')
            return None

        # Compute the GDT_TS score of the model from the target
        if self._gdt is None:
            with open(CONF.TEMP_OUT_GDT, 'w') as f:
                subprocess.call([CONF.TMSCORE_EXE_PATH, self._mod, self._targ], stdout=f)

            with open(CONF.TEMP_OUT_GDT, 'r') as f:
                lines = f.readlines()
            for l in lines:
                _l = l.split(' ')
                if _l[0] == 'GDT-TS-score=':
                  self._gdt = float(_l[1])
        return self._gdt

    def get_prediction(self, sess, graph):
        """
        Computes and returns the prediction made by a beforehand loaded tensorflow model
        :param sess: Tensorflow Session
        :param graph: Model graph
        :return: predicted score
        """
        # mapping protein
        maps_placeholder = graph.get_tensor_by_name('main_input:0')
        nbTypes = int(maps_placeholder.shape[1].value/(24*24*24))
        subprocess.call(
          [CONF.ROTAMERS_EXE_PATH, "--mode", "map", "-i", self._mod, "--native", "-m", "24", "-v", "0.8","-t",str(int(nbTypes)), "-o",
           CONF.TEMP_PATH + '_pred.bin'])
        if not os.path.exists(CONF.TEMP_PATH + '_pred.bin'):
            print('# Mapping failed, ignoring protein')
            return -1
        predDataset = load_data.read_data_set(CONF.TEMP_PATH + '_pred.bin')
        os.remove(CONF.TEMP_PATH + '_pred.bin')

        # creating placeholder for input data and feed dict
        is_training = graph.get_tensor_by_name('is_training:0')
        logits = graph.get_tensor_by_name("main_output:0")
        preds = []
        # compute prediction res by res
        for i in range(predDataset.num_res):
            f_map = np.reshape(predDataset.maps[i], (1, model.GRID_VOXELS * nbTypes))
            training = False
            feed_dict = {maps_placeholder: f_map, is_training: training}
            pred = sess.run(logits,feed_dict=feed_dict)
            preds.append(pred)
        # returns the mean value
        return np.mean(preds)
        
        


    def get_gt_out_score(self):
        """
        if it is not already computed
        compute the ground truth score using data in .out file (overall score)
        :return: ground truth score
        """
        if self._gt_score is None:
            if os.path.exists(self._mod + '.out'):
                  with open(self._mod + '.out') as f:
                        line = f.readline()
                        if len(line) < 4:
                            return -1
                        self._gt_score = float(line.split(' ')[4])

                        if self._gt_score > 1:
                            print('Value error for file out')
            else:
                print('ERROR : .out file doesn\'t exist, cannot tell gt score - return None')
                return -1

        return self._gt_score

    def get_gt_score(self):
        """
        if it is not already computed
        compute the ground truth score using data in .sco file (res by res score)
        :return: ground truth score
        """
        if self._gt_score_out is None:
            if os.path.exists(self._mod + '.sco'):
                self._gt_score_out = np.mean(stat_casp.stat_file(self._mod + '.sco'))
            else:
                print('ERROR : '+self._mod +'.sco file doesn\'t exist, cannot tell gt score - return None')
                return None
        return self._gt_score_out

    def add_target(self, path_to_target):
        self._targ = path_to_target

def compute_gdt_dir(path_to_models_directory, save=False):
    """
    Compute gdt score for all proteins in a specified directory. If the scores where already computed and saved
    restore the previous computation.
    :param path_to_models_directory: Path to the directory where are located all the protein models from which we want
                                     to compute GDT score. There must be one directory per protein containing all the
                                     different models of this protein. These files must be under
                                     path_to_models_directory + '/MODELS/'
    :param save: True if saving the scores. It will be saved in path_to_models_directory + '/EVAL/'. It creates one file
                 per protein containing the scores of all its models.
    :return: dictionary of the scored keyed by the name of the server
    """
    save_path = path_to_models_directory[:-6] + path_to_models_directory[-6:-1] + '_gdt_ts.sco'
    save_path = save_path.replace('MODELS', 'EVAL')
    dict_sco = {}
    if os.path.exists(save_path):
        print('Restoring previous calculation from ' + save_path)
        with open(save_path, 'r') as f:
            lines = f.readlines()
        for l in lines:
            l = l.split(' ')
            # verify that the score is not an outlier
            if 0 <= float(l[1][:-1]) <= 1 :
                dict_sco[l[0]] = float(l[1][:-1])

    else:
        ls_file = sorted(os.listdir(path_to_models_directory))
        for f in ls_file:
              m = model_prot(path_to_models_directory + f)
              dict_sco[f] = m.get_gdt()

    if save:
        if not os.path.exists(save_path):
            if not os.path.exists(path_to_models_directory + '/EVAL/'):
                os.makedirs(path_to_models_directory + '/EVAL/')
            with open(save_path, 'a') as f:
                for f_ in enumerate(ls_file):
                    f.write(f_ + ' ' + str(dict_sco[f_]) + '\n')
    return dict_sco

def compute_pred_dir(path_to_models_directory, path_to_nn = None, save=False, type_save='', model='deepscoring'):
    """
    Computes predictions (from a specified model) for all the proteins in a directory
    :param path_to_models_directory: Path to the directory where are located all the protein models from which we want
                                     to compute GDT score. There must be one directory per protein containing all the
                                     different models of this protein. These files must be under
                                     path_to_models_directory + '/MODELS/'
    :param path_to_nn: If testing our tensorflow model, path to the directory where the model is saved
    :param save: True if saving the scores. It will be saved in path_to_nn + '/Eval/'. It creates one file
                 per protein, containing the scores of all its models.
    :param type_save: This string will be added at the end opf the names of all the score files created to identify them
    :param model: Name of the model from which to take the prediction (our own is deep scoring)
    :return: dictionary of the scored keyed by the name of the server
    """

    # If the model is not our own, it will look for files containing the scores of this model.
    # The files must be named and saved following this convention :
    # There must be one file per protein containing the scores of all the models of this protein.
    # The files name's are NameOfProteinFolder + '.' + model + '_scores' eg. T0759.3Dcnn_scores
    # Theses files are located in path_to_models_directory
    score_dict = {}
    if not model == 'deepscoring':
        path = path_to_models_directory[:-6] + '/' + path_to_models_directory[-6:-1] + '.' + model + '_scores'

        with open(path,'r') as f:
            lines = f.readlines()

        for l in lines:
            # Since each model saved its scores following a different convention, we sparse the score file differently
            # according to the model name sparse the file differently
            # for
            if model == 'sbrod':
                name = l.split(' ')[0].split('/')[-1]
                try:
                    sco = float(l.split('\t')[1])
                    score_dict[name] = sco
                except ValueError:
                    print('Value Error, ignoring')
                    score_dict[name] = None

            elif model == 'rw':
                name = l.split(' ')[0].split('/')[-1]
                try:
                    sco = -1*float(l[77:91])
                    score_dict[name] = sco
                except ValueError:
                    print('Value Error, ignoring')
                    score_dict[name] = None

            elif model == 'voro':
                name = l.split(' ')[0].split('/')[-1]
                try:
                    sco = float(l.split(' ')[3])
                    score_dict[name] = sco
                except ValueError:
                    print('Value Error, ignoring')
                    score_dict[name] = None
            elif model == '3Dcnn':
                name = l.split(' ')[0]
                try:
                    sco = float(l.split(' ')[-1])
                    score_dict[name] = sco
                except ValueError:
                    print('Value Error, ignoring')
                    score_dict[name] = None
            elif model == 'my3d':
                name = l.split(' ')[0]
                try:
                    sco = float(l.split(' ')[-1])
                    score_dict[name] = sco
                except ValueError:
                    print('Value Error, ignoring')
                    score_dict[name] = None


            elif model == 'ProQ3D':
                name = l.split(' ')[0]
                try:
                    sco = float(l.split(' ')[-1])
                    score_dict[name] = sco
                except ValueError:
                    print('Value Error, ignoring')
                    score_dict[name] = None
        return score_dict

    # Removes the '_cad' appendix in cas
    save_name = type_save.replace('_cad','')
    save_path = path_to_nn + '/Eval/' + save_name + '_' + path_to_models_directory[-6:-1] + '_pred.sco'

    print(save_path)

    # If we're evaluating one of our own model, first we check if the calculation was already made for this protein.
    # beware of respecting the naming convention : '_' + CASP number + 'stage_' + Stage Number (+ _cad if using cad measure)
    if os.path.exists(save_path):
        print('Restoring previous calculation from ' + save_path)
        with open(save_path, 'r') as fi:
            lines = fi.readlines()

        for l in lines:
              l = l.split(' ')
              score_dict[l[0]] = float(l[1][:-1])

    else:
        # first needs to restore the model
        sess = tf.Session()

        print('Restore existing model: %s' % path_to_nn)
        print('Latest checkpoint: %s' % (tf.train.latest_checkpoint(path_to_nn)))
        saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(path_to_nn) + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(path_to_nn))

        graph = tf.get_default_graph()

        ls_file = sorted(os.listdir(path_to_models_directory))

        tot = len(ls_file)//2
        i__ = 0
        for f in ls_file:
            if f[-4:] == '.sco' or f[-4:] == '.out':
                continue
            i__ += 1
            m = model_prot(path_to_models_directory + f)
            score = m.get_prediction(sess, graph)
            print(str(i__) + '/' + str(tot) + ' : ' + str(score))
            score_dict[f] = score

        if save:
            if not os.path.exists(save_path):
                print('saving predictions into : ' + save_path)
                with open(save_path, 'a') as fi:
                    i = 0
                    for f in ls_file:
                        if f[-4:] == '.sco' or f[-4:] == '.out':
                            continue
                        fi.write(f + ' ' + str(score_dict[f]) + '\n')
                        i+=1

    return score_dict

def compute_gt(path_to_models_directory):
    """
    Compute the ground truth score for all the models of a protein using their .sco file
    :param path_to_models_directory: Path to the protein directory where all the models are located
    :return: dictionary of the scored keyed by the name of the server
    """
    ls_file = sorted(os.listdir(path_to_models_directory))
    dict_score = {}
    for f in ls_file:
        if f[-4:] == '.sco' or f[-4:] == '.out':
                continue

        m = model_prot(path_to_models_directory + '/' +f)
        dict_score[f] = m.get_gt_score()
    return dict_score

def compute_gt_out(path_to_models_directory):
    """
    Compute the ground truth score for all the models of a protein using their .out file
    :param path_to_models_directory: Path to the protein directory where all the models are located
    :return: dictionary of the scored keyed by the name of the server
    """
    ls_file = sorted(os.listdir(path_to_models_directory))
    dict_score = {}
    for f in ls_file:
        if f[-4:] == '.sco' or f[-4:] == '.out':
            continue

        m = model_prot(path_to_models_directory + '/' +f)
        dict_score[f] = m.get_gt_out_score()
    return dict_score

def compute_loss_eval(path_to_eval_directory, path_to_nn, name, measure='gdt', model='deepscoring', avoid_saving=''):
    """
    Evaluate one model (either our model, or any other model providing we have his prediction saved).
    :param path_to_eval_directory: Path to the directory where the dataset used to evaluate the model is saved
                                   eg. CASP11_stage1/
                                   In this directory must be, on directory named EVAL/ to store the computations, one
                                   named MODELS/ where are located all the proteins files containing the models, and the files
                                   containing the models, and one named TARGETS/ where are located the targets files
    :param path_to_nn: Path to the directory where the model is saved
    :param name: name of the evaluation. This will impact the naming of the files where are stored the computations.
                 I named my using this convention : '_' + CASP number + 'stage_' + Stage Number (+ '_' + name of the model tested if not our own)(+ _cad if using cad measure)
                 eg. _11stage_2_voro_cad, _12stage_1
    :param measure: The reference measure (either gdt or cad score)
    :param model: Name of the model if not own eg. voro, sbrod, 3Dcnn...
    :param avoid_saving: 'avoid' if you dont want to save the final results into a file
    :return: Creates a file in path_to_nn + 'Eval/'  final_results... containing the evaluation
    """
    ls_targets = sorted(os.listdir(path_to_eval_directory + 'TARGETS/'))

    losses = []
    kens = []
    pers = []
    spes = []

    pred_all = []
    m_all = []

    for target in ls_targets:
        target = target[:5]
        if not os.path.exists(path_to_eval_directory + 'MODELS/' + target + '/'):
            print('Skipping because of missing data')
            continue
        if not model == 'deepscoring':
            if not os.path.exists(path_to_eval_directory + 'MODELS/' + target + '.' + model + '_scores'):
                print('Skipping because of missing data')
                continue

        print('--> For target : ' + target)
        if measure == 'cad':
            measure_dict = compute_gt(path_to_eval_directory + 'MODELS/' + target + '/')
        elif measure == 'out':
            measure_dict = compute_gt_out(path_to_eval_directory + 'MODELS/' + target + '/')
        else:
            measure_dict = compute_gdt_dir(path_to_eval_directory + 'MODELS/' + target + '/', save=True)

        pred_dict = compute_pred_dir(path_to_eval_directory + 'MODELS/' + target + '/', path_to_nn, save=True, type_save='loss_eval' + name, model=model)

        list_of_servers_ = sorted(os.listdir(path_to_eval_directory + 'MODELS/' + target + '/'))
        list_of_servers = []

        for ser in list_of_servers_:
            if ser[-3:] == 'out' or ser[-3:] == 'sco':
                continue
            list_of_servers.append(ser)


        if len(pred_dict) == 0:
            print('No prediction made for this protein')
            continue

        ls_pred_scores, ls_m_scores = [], []
        print(list_of_servers)
        for server in list_of_servers:
            pred = pred_dict.get(server)
            m = measure_dict.get(server)


            if (not pred is None) and (not m is None):

                ls_pred_scores.append(pred)
                ls_m_scores.append(m)

        if sum(ls_m_scores) == 0:
            print('> corrupted ground truth file : Ignoring model')
            continue

        rank_pred = scipy.stats.rankdata(ls_pred_scores, method='dense')
        i_max_pred = np.argmax(np.array(ls_pred_scores))
        if model == '3Dcnn' or model == 'my3d':
            #ls_pred_scores = [-s for s in ls_pred_scores]
            i_max_pred = np.argmin(np.array(ls_pred_scores))
            #i_max_pred = np.argmax(np.array(ls_pred_scores))

        rank_m = scipy.stats.rankdata(ls_m_scores, method='dense')

        i_max_m = np.argmax(np.array(ls_m_scores))


        for i_ in range(len(ls_pred_scores)):
            pred_all.append(ls_pred_scores[i_])
            m_all.append(ls_m_scores[i_])

        loss = np.abs(ls_m_scores[i_max_m] - ls_m_scores[i_max_pred])
        print('--> ' + target + ' loss : ', loss)
        per = scipy.stats.pearsonr(ls_pred_scores, ls_m_scores)[0]
        print('--> ' + target + ' pearson : ', per)
        spe = scipy.stats.spearmanr(ls_pred_scores, ls_m_scores)[0]
        print('--> ' + target + ' spearman : ', spe)
        ken = scipy.stats.kendalltau(rank_pred, rank_m)[0]
        print('--> ' + target + ' kendall : ', ken)

        losses.append(loss)
        kens.append(ken)
        pers.append(per)
        spes.append(spe)

        print('---> Current mean loss : ', np.mean(losses))
        print('----> Final std loss : ', np.std(losses))
        print('---> Current mean pearson : ', np.mean(pers))
        print('---> Current mean spearman : ', np.mean(spes))
        print('---> Current mean kendall : ', np.mean(kens))

    print('----> Final mean loss : ', np.mean(losses))
    print('----> Final confidence interval loss : ', np.std(losses)/np.sqrt(len(losses)))
    print('----> Final mean pearson : ', np.mean(pers))
    print('----> Final mean spearman : ', np.mean(spes))
    print('----> Final mean kendall : ', np.mean(kens))

    print('\n\n')

    print('Over all data :\n')
    print('Pearson : ', scipy.stats.pearsonr(pred_all, m_all)[0])
    print('Spearman : ', scipy.stats.spearmanr(pred_all, m_all)[0])
    rank_m = scipy.stats.rankdata(m_all, method='dense')
    rank_pred = scipy.stats.rankdata(pred_all, method='dense')
    print('Kendall : ', scipy.stats.kendalltau(rank_pred, rank_m)[0])

    save_path = path_to_nn + 'Eval/' + 'results_eval' + name

    if avoid_saving == 'avoid':
        return -1
    if not os.path.exists(save_path):
        print('Saving results into : ' + save_path)
        fi = open(save_path, 'a')
        fi.write('======= RESULTS =======\n')
        fi.write('> Final loss : ' + np.str(np.mean(losses)) + '\n')
        fi.write('> Final pearson : ' + np.str(np.mean(pers)) + '\n')
        fi.write('> Final spearman : ' + np.str(np.mean(spes)) + '\n')
        fi.write('> Final kendall : ' + np.str(np.mean(kens)) + '\n')
        fi.write('target - loss - pearson - spearman - kendall\n')
        for i, target in enumerate(ls_targets):
            target = target[:5]
            fi.write(target + ',' + np.str(losses[i]) + ',' + np.str(pers[i]) + ',' + np.str(spes[i]) + ',' + np.str(kens[i]) + '\n')
            fi.close()
    return np.mean(losses)

def main():

    # Name of the data directory must be CASPX_stageY

    save_name = '_' + FLAGS.data.split('/')[-2].split('_')[0][4:] + 'stage_' + FLAGS.data.split('/')[-2].split('_')[1][5:]
    if not FLAGS.type_model == 'deepscoring':
        save_name = save_name + '_' + FLAGS.type_model
    if not FLAGS.score == 'gdt':
        save_name = save_name + '_' + FLAGS.score
    print(save_name)

    compute_loss_eval(FLAGS.data, FLAGS.model, name =save_name, measure=FLAGS.score,model=FLAGS.type_model,avoid_saving=FLAGS.save)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-s',
    '--score',
    type=str,
    default='gdt',
    help='Give the measure we are refering too'
  )
  parser.add_argument(
    '-d',
    '--data',
    type=str,
    help='Path to the validation data'
  )
  parser.add_argument(
    '-m',
    '--model',
    type=str,
    help='Path to the model that is tested'
  )    
  parser.add_argument(
    '-n',
    '--name',
    type=str,
    default= '_1Xstage_X',
    help='Name of the test'
  )
  parser.add_argument(
    '--save',
    type=str,
    default= '',
    help='avoid avoids saving'
  )
  parser.add_argument(
    '-t',
    '--type_model',
    type=str,
    default= 'deepscoring',
    help='If testing a different kind of model'
  )
  FLAGS = parser.parse_args()
  main()


