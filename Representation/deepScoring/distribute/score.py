import config
import subprocess
import os
import numpy as np
import scipy.stats
import random

import tensorflow as tf

import load_data
import model
import argparse

CONF = config.load_config()
FLAGS = None

def predict(sess, maps_placeholder, is_training, logits, filenames, meta_pl = None,outSuffix = ".orn", mapOptions=[]):
    print(mapOptions)
    # mapping protein
    mapFilename = CONF.TEMP_PATH + 'map_'+str(os.getpid())+ '_pred.bin'
    subProcessList=[]
    k = 0
    nbTypes = int(maps_placeholder.shape[1].value/(24*24*24))
    for filename in filenames :
        print('# Scoring '+filename)
        #subprocess.call([CONF.MAP_GENERATOR_PATH, "--mode", "map", "-i", filename, "--native", "-m", "24", "-v", "0.8", "-o", mapFilename])
        subProcessList.append(subprocess.Popen([CONF.MAP_GENERATOR_PATH, "--mode", "map", "-i", filename, "--native", "-m", "24", "-v", "0.8", "-o", mapFilename+str(k)]+mapOptions))
        k+=1
    k = 0
    for filename in filenames :
        subProcessList[k].wait()
        if not os.path.exists(mapFilename+str(k)):
            print('# Mapping failed, ignoring protein')
            k+=1
            continue
        predDataset = load_data.read_data_set(mapFilename+str(k))
        if predDataset is None :
            k+=1
            continue
        os.remove(mapFilename+str(k))
        result_file = open(filename+outSuffix,"w")

        preds = []
        # compute prediction res by res
        for i in range(predDataset.num_res):
            f_map = np.reshape(predDataset.maps[i], (1, model.GRID_VOXELS * nbTypes))
            if meta_pl is not None : 
                f_meta = np.reshape(predDataset.meta[i], (1, 16))
                feed_dict = {maps_placeholder: f_map, meta_pl:f_meta, is_training: False}
            else :
                feed_dict = {maps_placeholder: f_map, is_training: False}
            pred = sess.run(logits,feed_dict=feed_dict)
            preds.append(pred)
            outline='RES {:4d} {:c} {:5.4f}'.format(predDataset.meta[i][0], predDataset.meta[i][1], pred)
            print(outline, file = result_file)
            #print(predDataset.meta[i][0]+)
            #print(pred)
        result_file.close()
        k+=1

def main():
    print(FLAGS.options)
    #exit()
    sess = tf.Session()
    print('Restore existing model: %s' % FLAGS.model)
    saver = tf.train.import_meta_graph(FLAGS.model + '.meta')
    saver.restore(sess, FLAGS.model)

    graph = tf.get_default_graph()

    # getting placeholder for input data and output
    maps_placeholder = graph.get_tensor_by_name('main_input:0')
    meta_placeholder = graph.get_tensor_by_name('meta_pl:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    logits = graph.get_tensor_by_name("main_output:0")
     

    if FLAGS.structure != None :
        predict(sess, maps_placeholder, is_training, logits, FLAGS.structure, meta_pl = meta_placeholder, outSuffix = FLAGS.suffix, mapOptions=FLAGS.options.split())
    if FLAGS.directory != None :
        bufferFiles = []
        for filename in os.listdir(FLAGS.directory):
            bufferFiles.append(FLAGS.directory+'/'+filename)
            if len(bufferFiles) == FLAGS.buffer :
                predict(sess, maps_placeholder, is_training, logits, bufferFiles, meta_pl = meta_placeholder, outSuffix = FLAGS.suffix, mapOptions=FLAGS.options.split())
                bufferFiles = []
        predict(sess, maps_placeholder, is_training, logits, bufferFiles, meta_pl = meta_placeholder, outSuffix = FLAGS.suffix, mapOptions=FLAGS.options.split())






if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-d',
    '--directory',
    type=str,
    help='Path to the validation data'
  )
  parser.add_argument(
    '-s',
    '--structure',
    type=str,
    help='Path to the structure to score (in pdb format)'
  ) 
  parser.add_argument(
    '-f',
    '--suffix',
    default=".orn",
    type=str,
    help='suffix to add to result file'
  ) 
  parser.add_argument(
    '-b',
    '--buffer',
    default="8",
    type=int,
    help='number of files to buffer '
  ) 
  parser.add_argument(
    '-o',
    '--options',
    default=["-m", "24", "-v", "0.8"],
    type=str,
    
    help='argument to map generator '
  )
  parser.add_argument(
    '-m',
    '--model',
    default=CONF.MODEL_PATH,
    type=str,
    
    help='argument to map generator '
  )
  FLAGS = parser.parse_args()
  #print(FLAGS.options)
  main()


