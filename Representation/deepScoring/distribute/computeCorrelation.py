import config
import subprocess
import os
import numpy as np
import scipy.stats
from scipy.stats.stats import pearsonr  
import random
import math
import tensorflow as tf

import load_data
import model
import argparse

from glob import glob

def getAverageScoreSco(filename) :
    scoreList=[]
    #print(filename)
    with open(filename) as fp:
        lines = fp.readlines()
        for l in lines :
            #print(l)
            for k in l.split(" "):
                #print(k)
                try:
                    scoreList.append(float(k))
                except ValueError:
                    continue
    if scoreList == [] :
        return -1
    return np.average(scoreList)
    
def getAverageScoreOut(filename) :
    scoreList=[]
    #print(filename)
    with open(filename) as fp:
        lines = fp.readlines()
        for l in lines :
            #print(l)#print(k)
            try:
                scoreList.append(float(l.split(" ")[4]))
            except ValueError:
                continue
    if scoreList == [] :
        return -1
    return np.average(scoreList)
    
    
def getAverageScore(filename) :
    scoreList=[]
    #print(filename)
    with open(filename) as fp:
        lines = fp.readlines()
        for l in lines :
            #print(l)
            #print(k)
            try:
                scoreList.append(float(l[10:]))
            except ValueError:
                continue
    if scoreList == [] :
        return -1
    return np.average(scoreList)
    
    
def getDictScore(filename) :
    scoreDict={}
    #print(filename)
    with open(filename) as fp:
        lines = fp.readlines()
        for l in lines :
            #print(l)
            #print(k)
            try:
                resId = int(l[3:8])
                score= float(l[10:])
                scoreDict[resId] = score
            except ValueError:
                continue
    return scoreDict
        
def getDictScoreSco(filename) :
    scoreDict={}
    #print(filename)
    with open(filename) as fp:
        lines = fp.readlines()
        for l in lines :
            #print(l)
            #print(k)
            try:
                resId = int(l.split("r<")[1].split(">")[0])
                score= float(l.split(" ")[-1])
                scoreDict[resId] = score
            except ValueError:
                continue
    return scoreDict

def main():
    gtfiles = glob(FLAGS.directory + '/**/*'+FLAGS.suffix1, recursive=True)
    scoresGT = []
    scoresRes = []
    diffScores = []
    for f in gtfiles :
        
        resFile = f[:-len(FLAGS.suffix1)]+FLAGS.suffix2
        if os.path.exists(resFile) :
            #scoreGT = getAverageScoreOut(f)
            #scoreRes = getAverageScore(resFile)
            scoreSco = getDictScoreSco(f)
            scoreRes = getDictScore(resFile)
            for key, value in scoreSco.items():
                if key in scoreRes :
                    scoresGT.append(scoreSco[key])
                    scoresRes.append(scoreRes[key])
                    diffScores.append((scoreSco[key] - scoreRes[key])*(scoreSco[key] - scoreRes[key]))
                
            #print(f)
            #print(scoreGT)
            #print(scoreRes)
            
            
    #if scoresGT == [] :
     #   return
    print(len(scoresGT))
    print(pearsonr(scoresGT, scoresRes))
    print(np.mean(diffScores)*1000)
    print(1000*(np.std(diffScores))/math.sqrt(len(diffScores)))


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
    '--suffix1',
    default=".cad",
    type=str,
    help='suffix to add to gt file'
  ) 
  parser.add_argument(
    '-f',
    '--suffix2',
    default=".orn",
    type=str,
    help='suffix to add to result file'
  ) 
  FLAGS = parser.parse_args()
  main()


