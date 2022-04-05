import numpy as np 
import matplotlib.pyplot as plt
import os
import argparse


# Path to the directory where the losses files are located
directory_path = '/home/benoitch/save/tests_cluster/'
files = sorted(os.listdir(directory_path))
losses = []
indix = []
FLAGS = None

line_size = 0.9

def get_list():
    """
    format the FLAGS.tests
    :return: list of int, list of the test ids to plot
    """
    ids = []
    for l in FLAGS.tests:
        id = ''
        for i in l:
            id += i
        ids.append(int(id))
    return ids

def recover_loss(s):
    """
    Exctract the loss, from a smoothed loss list
    :param s: 1D List containing smooth loss : list created during training of the model
    :return: List containing the raw loss
    """
    alpha = 0.999
    l = np.zeros(s.size)
    l[0] = s[0]/1000
    for n in range(l.size):
        if n == 0:
            continue

        l[n] = (1/(1-alpha))*((s[n]*(1-alpha**n)/1000) - (alpha*s[n-1]*(1-alpha**(n-1))/1000))
    return l

def smooth(l):
    """
    Smooth the loss
    :param l: 1D list of the raw loss values
    :return: 1D list of the smooth loss
    """
    s = np.zeros(l.size)
    slid = 0
    decay = 0.999
    dln = 1
    for i in range(l.size):
        dln = dln*decay
        slid = decay*slid + (1-decay)*l[i]
        s[i] = 1000* slid / (1 - dln)
    return s

  
def main(ids):
    for f in files:
        f = directory_path + f

        # This two conditions are used to plot reference loss (simply prediciting the mean of what it has seen so far)
        # (The files containing the reference loss are located in benoitch/save/tests_cluster/ and are called gt_losses.npy and gt_x_losses.npy)
        if f.split('/')[-1][:4] == 'gt_l':
            f__ = open(f, 'rb')
            loss_gt = np.load(f__)
        elif f.split('/')[-1][:4] == 'gt_x':
            f__ = open(f, 'rb')
            x_gt = np.load(f__)
        # then collecting the ids of the tests in the directory
        elif f[-3:] == 'npy':
            f__ = open(f, 'rb')
            loss = np.load(f__)
            losses.append(loss)
            # get the id of the file's origin network id is an int
            f = f[:-4]
            indix.append(int(f.split('/')[-1].split('_')[1])) # the id needs to be the 7th char of the file's name

    number = -1

    for i,l in enumerate(losses):

        # need to deal with losses from a same network training splited in two files
        if not indix[i] in ids:
            continue

        if number == -1:
            number = indix[i]
            y = recover_loss(l)
            x = range(l.size)
        elif indix[i] == number:
            y = np.append(y,recover_loss(l))
            x = np.append(x, [x[-1]  + n for n in range(l.size)])
        else:
            s = smooth(y)
            plt.plot(x,s, alpha = 0.8, linewidth=line_size)
            number = indix[i]
            y = recover_loss(l)
            x = range(l.size)

    s = smooth(y)
    plt.plot(x,s, alpha = 0.8, linewidth=line_size)

    plt.plot(x_gt,loss_gt, alpha = 0.5, color='grey', linewidth=line_size)
    plt.show()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--tests',
        type=list,
        required=True,
        nargs='+',
        help='List of tests ids to plot'
    )
    FLAGS = parser.parse_args()
    ids = get_list()
    main(ids)
