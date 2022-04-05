import numpy as np
import os



def stat_file(path_to_file):
    f = open(path_to_file, 'r')

    lines = f.readlines()
    scores = []

    for l in lines:
        [_, sco] = l.split(' ')
        if sco[-2:] == '\n':
            sco = sco[:-2]
        if sco == '0':
            scores.append(0)
        else:
            scores.append(float(sco))
    f.close()
    if len(scores) == 0:
        return [0]

    return np.mean(scores)

def stat_prot(path_to_prot):
 
    file_ls = os.listdir(path_to_prot)
    scores_p = np.array([])

    for f in file_ls:
        if f[-4:] == '.sco':
          sco_f = stat_file(path_to_prot + '/' + f)
          scores_p = np.append(scores_p, sco_f)
    if scores_p == np.array([]):
        return None
    return scores_p

def stat_casp(path_to_casp):

    file_ls = os.listdir(path_to_casp)
    scores_c = np.array([])

    for f in file_ls:
        stats_prot = stat_prot(path_to_casp + '/' + f)
        if not stats_prot is None:
            scores_c = np.append(scores_c, stats_prot)

    scores_c = np.reshape(scores_c,(-1,))

    return scores_c

def stat_full(path_to_data, casp_choice):

    file_ls = casp_choice
    scores_full = np.array([])

    for f in file_ls:
        print('Stats : ' + f)
        scores_c = stat_casp(path_to_data + '/' + f + '/MODELS/')
        scores_full = np.append(scores_full, scores_c)
        print('Done')
    scores_full = np.reshape(scores_full,(-1,))
    return scores_full


def main():
    print(stat_full('/home/benoitch/data/', ['CASP7','CASP8']))

if __name__ == '__main__':
    main()
