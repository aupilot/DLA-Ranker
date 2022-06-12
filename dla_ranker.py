import argparse
import gc
import glob
import os
import pathlib
import re
import shutil
import sys
from multiprocessing import Pool
from os import path, remove, listdir
from shutil import copyfile, rmtree, copy
import numpy as np
import pandas as pd
from subprocess import CalledProcessError, check_call, call
# from prody import
import subprocess
import pickle
from prody import parsePDB, writePDB
from sklearn.preprocessing import OneHotEncoder
# disable TF warnings
from tensorflow.python.keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

'''
Run DLA-Ranker on "confom_dict" decoys and copy to the good_decoys dir only these that have score better than acceptable_score
Ex:
python3 dla_ranker.py
'''

# acceptable_score = 0.06        # 0.06 - good indicator.

use_multiprocessing = False  # True не работает, because naccess uses the same file names for temp files
#TODO: fix multipro
max_cpus = 8

confom_dict = [
    "decoy.1.pdb",
    "decoy.2.pdb",
    "decoy.3.pdb",
    "decoy.4.pdb",
    "decoy.5.pdb",
    "decoy.6.pdb",
    # "decoy.7.pdb",
    # "decoy.8.pdb",
    ]

channel_receptor = 'HL'
channel_ligand   = 'A'

# TODO: replace with the correct path!
# dla_dir = "/home/kir/Apps/DLA-Ranker/"
dla_dir = "/opt/DLA-Ranker/"
decoy_dir = '/opt/var/decoys/'
map_dir = '/opt/var/map_dir/'
good_decoys = '/opt/var/good_decoys/'

maps_gen_exe = dla_dir + 'Representation/maps_generator'
v_dim = 24
# naccess_exe  = f'cd {map_dir}; {dla_dir}Naccess/naccess'
naccess = f'{dla_dir}Naccess/naccess'

# sys.path.insert(1, f"{dla_dir}Representation")
import Representation.load_data as load


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, fix_imports=True)
        f.close()


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def do_processing(cases, function, use_multiprocessing):

    if use_multiprocessing:
        with Pool() as pool:
            pool.map(function, cases)
    else:
        for args in cases:
            function(args)


def rimcoresup(rsa_rec, rsa_lig, rsa_complex):
    '''INPUT: file rsa da NACCESS.

       ###rASAm: relative ASA in monomer
       ###rASAc: relative ASA in complex

       ###Levy model
       ###deltarASA=rASAm-rASAc
       ###RIM      deltarASA > 0 and rASAc >= 25 and rASAm >= 25   ###corretto da rASAc > 25 and rASAm > 25
       ###CORE     deltarASA > 0 and rASAm >= 25 and rASAc < 25    ###corretto da rASAm > 25 and rASAc <= 25
       ###SUPPORT  deltarASA > 0 and rASAc < 25 and rASAm < 25

       OUTPUT:rim, core, support'''

    ASA1 = []
    resNUMasa1 = 0
    lines = [line.rstrip('\n') for line in open(rsa_rec)]
    for lineee in lines:
        a = re.split(' ', lineee)
        a = list(filter(None, a))
        if a[0] == 'RES' and len(a) == 14:
            resNUMasa1 = (resNUMasa1) + 1
            restype = a[1]
            chain = a[2]
            resnumb = a[3]
            resnumb = re.findall('\d+', resnumb)
            resnumb = resnumb[0]
            rASAm = a[5]
            ASA1.append((restype, chain, int(resnumb), rASAm, 'receptor'))
        elif a[0] == 'RES' and len(a) == 13:
            resNUMasa1 = (resNUMasa1) + 1
            restype = a[1]
            testchain = re.findall('\d+|\D+', a[2])
            if len(testchain) == 1:
                chain = ''
                resnumb = int(testchain[0])
            elif len(testchain) == 2:
                primoterm = testchain[0]
                if primoterm.isdigit():
                    chain = ''
                    resnumb = int(testchain[0])
                else:
                    chain = testchain[0]
                    resnumb = int(testchain[1])
                    # resnumb=re.findall('\d+', resnumb)
            # resnumb=resnumb[0]
            rASAm = a[4]
            ASA1.append((restype, chain, int(resnumb), rASAm, 'receptor'))

    ASA2 = []
    resNUMasa2 = 0
    lines = [line.rstrip('\n') for line in open(rsa_lig)]
    for lineee in lines:
        a = re.split(' ', lineee)
        a = list(filter(None, a))
        if a[0] == 'RES' and len(a) == 14:
            resNUMasa2 = (resNUMasa2) + 1
            restype = a[1]
            chain = a[2]
            resnumb = a[3]
            resnumb = re.findall('\d+', resnumb)
            resnumb = resnumb[0]
            rASAm = a[5]
            ASA2.append((restype, chain, int(resnumb), rASAm, 'ligand'))
        elif a[0] == 'RES' and len(a) == 13:
            resNUMasa2 = (resNUMasa2) + 1
            restype = a[1]
            testchain = re.findall('\d+|\D+', a[2])
            if len(testchain) == 1:
                chain = ''
                resnumb = int(testchain[0])
            elif len(testchain) == 2:
                primoterm = testchain[0]
                if primoterm.isdigit():
                    chain = ''
                    resnumb = int(testchain[0])
                else:
                    chain = testchain[0]
                    resnumb = int(testchain[1])
                    # resnumb=re.findall('\d+', resnumb)
            # resnumb=resnumb[0]
            rASAm = a[4]
            ASA2.append((restype, chain, int(resnumb), rASAm, 'ligand'))

    ASAfull = []
    resNUMasafull = 0
    lines = [line.rstrip('\n') for line in open(rsa_complex)]
    for lineee in lines:
        a = re.split(' ', lineee)
        a = list(filter(None, a))
        if a[0] == 'RES' and len(a) == 14:
            resNUMasafull = resNUMasafull + 1
            restype = a[1]
            chain = a[2]
            resnumb = a[3]
            resnumb = re.findall('\d+', resnumb)
            resnumb = resnumb[0]
            rASAm = a[5]
            if resNUMasafull <= len(ASA1):
                filename = 'receptor'
            else:
                filename = 'ligand'
            ASAfull.append((restype, chain, int(resnumb), rASAm, filename))
        elif a[0] == 'RES' and len(a) == 13:
            resNUMasafull = resNUMasafull + 1
            restype = a[1]
            testchain = re.findall('\d+|\D+', a[2])
            if len(testchain) == 1:
                chain = ''
                resnumb = int(testchain[0])
            elif len(testchain) == 2:
                primoterm = testchain[0]
                if primoterm.isdigit():
                    chain = ''
                    resnumb = int(testchain[0])
                else:
                    chain = testchain[0]
                    resnumb = int(testchain[1])
                    # resnumb=re.findall('\d+', resnumb)
            # resnumb=resnumb[0]
            rASAm = a[4]
            if resNUMasafull <= len(ASA1):
                filename = 'receptor'
            else:
                filename = 'ligand'
            ASAfull.append((restype, chain, int(resnumb), rASAm, filename))

    rim = []
    core = []
    support = []
    for elements in ASAfull:
        for x in ASA1:
            if elements[0:3] == x[0:3] and elements[4] == x[4]:
                rASAm = float(x[3])
                rASAc = float(elements[3])
                deltarASA = rASAm - rASAc
                if deltarASA > 0:
                    if rASAm < 25:  # and rASAc < 25:
                        support.append(x)
                    elif rASAm > 25:
                        if rASAc <= 25:
                            core.append(x)
                        else:
                            rim.append(x)

        for x in ASA2:
            if elements[0:3] == x[0:3] and elements[4] == x[4]:
                rASAm = float(x[3])
                rASAc = float(elements[3])
                deltarASA = rASAm - rASAc
                if deltarASA > 0:
                    if rASAm < 25:  # and rASAc < 25:
                        support.append(x)
                    elif rASAm > 25:
                        if rASAc <= 25:
                            core.append(x)
                        else:
                            rim.append(x)

    return rim, core, support


def run_nsaccess(arg):
    # we run naccess in a separate dir, because it creates tmp files. then we copy all the result files back to our working dir
    thr = arg[0]
    pdb = arg[1]
    dir = f"/tmp/naccess_{thr}"
    pathlib.Path(dir).mkdir(exist_ok=True)
    cmd = f'cd {dir} && {naccess} -r {dla_dir}Naccess/vdw.radii -s {dla_dir}Naccess/standard.data {pdb}'
    os.system(cmd)

    for file in glob.glob(dir + "/*"):
        # print(file)
        shutil.copy(file, map_dir)


def get_scr(rec, lig, com, name):
    # cmd = f'{naccess_exe} -r {dla_dir}Naccess/vdw.radii -s {dla_dir}Naccess/standard.data {com}'
    # os.system(cmd)
    # cmd = f'{naccess_exe} -r {dla_dir}Naccess/vdw.radii -s {dla_dir}Naccess/standard.data {rec}'
    # os.system(cmd)
    # cmd = f'{naccess_exe} -r {dla_dir}Naccess/vdw.radii -s {dla_dir}Naccess/standard.data {lig}'
    # os.system(cmd)

    with Pool() as pool:
        pool.map(run_nsaccess, [[0,com], [1,rec], [2,lig]])

    # ('GLN', 'B', '44', '55.7', 'receptor')
    # rim, core, support = rimcoresup(path.basename(rec.replace('pdb', 'rsa')), path.basename(lig.replace('pdb', 'rsa')),
    #                                 path.basename(com.replace('pdb', 'rsa')))
    rim, core, support = rimcoresup(rec.replace('pdb', 'rsa'), lig.replace('pdb', 'rsa'), com.replace('pdb', 'rsa'))
    outprimcoresup = open(name + '_rimcoresup.csv', 'w')

    for elementrim in rim:
        outprimcoresup.write(str((' '.join(map(str, elementrim))) + " R") + "\n")  # Rim
    for elementcore in core:
        outprimcoresup.write(str((' '.join(map(str, elementcore))) + " C") + "\n")  # Core
    for elementsup in support:
        outprimcoresup.write(str((' '.join(map(str, elementsup))) + " S") + "\n")  # Support

    outprimcoresup.close()

    remove(rec.replace('pdb', 'rsa'))
    remove(rec.replace('pdb', 'asa'))
    remove(rec.replace('pdb', 'log'))

    remove(lig.replace('pdb', 'rsa'))
    remove(lig.replace('pdb', 'asa'))
    remove(lig.replace('pdb', 'log'))

    remove(com.replace('pdb', 'rsa'))
    remove(com.replace('pdb', 'asa'))
    remove(com.replace('pdb', 'log'))


def mapcomplex(complex):

    file = decoy_dir + complex
    name = map_dir + complex.replace(".pdb","")

    rec = parsePDB(file, chain=channel_receptor)
    rec.setChids('R')   # two chains H+L are renamed to one R

    lig = parsePDB(file, chain=channel_ligand)
    lig.setChids('L')   # we keep the epitope as A

    writePDB(name + '_r.pdb', rec)
    writePDB(name + '_l.pdb', lig)
    writePDB(name + '_complex.pdb', rec + lig)

    get_scr(name + '_r.pdb', name + '_l.pdb', name + '_complex.pdb', name)

    rcs = pd.read_csv(name + '_rimcoresup.csv', header=None, sep=' ')
    rec_regions = rcs.loc[rcs[4] == 'receptor']
    rec_regions = pd.Series(rec_regions[5].values, index=rec_regions[2]).to_dict()
    lig_regions = rcs.loc[rcs[4] == 'ligand']
    lig_regions = pd.Series(lig_regions[5].values, index=lig_regions[2]).to_dict()

    res_num2name_map_rec = dict(zip(rec.getResnums(), rec.getResnames()))
    res_num2name_map_lig = dict(zip(lig.getResnums(), lig.getResnames()))
    res_num2coord_map_rec = dict(zip(rec.select('ca').getResnums(), rec.select('ca').getCoords()))
    res_num2coord_map_lig = dict(zip(lig.select('ca').getResnums(), lig.select('ca').getCoords()))

    L1 = list(set(rec.getResnums()))
    res_ind_map_rec = dict([(x, inx) for inx, x in enumerate(sorted(L1))])
    L1 = list(set(lig.getResnums()))
    res_ind_map_lig = dict([(x, inx + len(res_ind_map_rec)) for inx, x in enumerate(sorted(L1))])

    res_inter_rec = [(res_ind_map_rec[x], rec_regions[x], x, 'R', res_num2name_map_rec[x], res_num2coord_map_rec[x])
                     for x in sorted(list(rec_regions.keys())) if x in res_ind_map_rec]
    res_inter_lig = [(res_ind_map_lig[x], lig_regions[x], x, 'L', res_num2name_map_lig[x], res_num2coord_map_lig[x])
                     for x in sorted(list(lig_regions.keys())) if x in res_ind_map_lig]

    reg_type = list(map(lambda x: x[1], res_inter_rec)) + list(map(lambda x: x[1], res_inter_lig))
    res_name = list(map(lambda x: [x[4]], res_inter_rec)) + list(map(lambda x: [x[4]], res_inter_lig))
    res_pos = list(map(lambda x: x[5], res_inter_rec)) + list(map(lambda x: x[5], res_inter_lig))

    # Merge these two files!
    with open('resinfo', 'w') as fh_res:
        for x in res_inter_rec:
            fh_res.write(str(x[2]) + ';' + x[3] + '\n')
        for x in res_inter_lig:
            fh_res.write(str(x[2]) + ';' + x[3] + '\n')

    with open('scrinfo', 'w') as fh_csr:
        for x in res_inter_rec:
            fh_csr.write(str(x[2]) + ';' + x[3] + ';' + x[1] + '\n')
        for x in res_inter_lig:
            fh_csr.write(str(x[2]) + ';' + x[3] + ';' + x[1] + '\n')

    if not res_inter_rec or not res_inter_lig:
        return [], [], []

    # tl.coarse_grain_pdb('train.pdb')
    mapcommand = [maps_gen_exe, "--mode", "map", "-i", name + '_complex.pdb', "--native", "-m", str(v_dim), "-t", "167",
                  "-v", "0.8", "-o", name + '_complex.bin']
    # call(mapcommand)
    output = subprocess.run(mapcommand, capture_output=True, check=True)
    dataset_train = load.read_data_set(name + '_complex.bin')

    print(dataset_train.maps.shape)

    # scaler = MinMaxScaler()
    # scaler.fit(dataset_train.maps)
    # data_norm = scaler.transform(dataset_train.maps)
    data_norm = dataset_train.maps

    X = np.reshape(data_norm, (-1, v_dim, v_dim, v_dim, 173))
    y = [0] * (len(res_inter_rec) + len(res_inter_lig))

    map_name = path.join( name)
    save_obj((X, y, reg_type, res_pos, res_name, res_inter_rec + res_inter_lig), name)

    check_call(
        [
            'lz4', '-f',
            map_name + '.pkl',
            map_name + '.lz4'
        ],
        stdout=sys.stdout)

    remove(name + '.pkl')
    remove(name + '_complex.bin')
    remove(name + '_r.pdb')
    remove(name + '_l.pdb')
    remove(name + '_complex.pdb')
    remove(name + '_rimcoresup.csv')

    # print(type(X))
    # print(X.shape)
    # return X, y, reg_type

def load_map(sample_path):
    check_call(
        [
            'lz4', '-d', '-f', '--rm',
            sample_path,
            sample_path + ".pkl"
        ],
        # stdout=sys.stdout)
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    X_train, y_train, reg_type, res_pos,_,info = load_obj(sample_path)
    # remove(sample_path.replace('.lz4',''))
    return X_train, y_train, reg_type, res_pos, info


def generate_cubes(decoys):
    os.makedirs(map_dir, exist_ok=True)
    do_processing(decoys, mapcomplex, use_multiprocessing)


def dla_rank(acceptable_score=0.06):
    print('Your tensorflow version: {}'.format(tf.__version__))
    print("GPU : "+tf.test.gpu_device_name())
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    samples_test = listdir(map_dir)
    model = load_model(path.join(dla_dir, 'Models', 'Dockground', '0_model'))

    # predictions_file = open('../Kir/predictions_kir.txt', 'w')
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    onehot = encoder.fit(np.asarray([['S'], ['C'], ['R']]))

    results = dict()

    for test_interface in samples_test:
        if not test_interface.endswith("lz4"):
            continue
        # print(test_interface)
        sample_path = map_dir + test_interface
        X_test, y_test, reg_type, res_pos, info = load_map(sample_path)
        remove(sample_path + ".pkl")
        X_aux = encoder.transform(list(map(lambda x: [x], reg_type)))

        if len(X_test) == 0 or len(X_aux) != len(X_test):
            continue

        all_scores = model.predict([X_test, X_aux], batch_size=X_test.shape[0])
        _ = gc.collect()

        test_preds = all_scores.mean()
        # print(y_test)
        print(f"{test_interface} \t{test_preds}")
        results[test_interface] = test_preds

    return results


def dla_ranker_filter(acceptable_score):
    # first, we need to get rid of dots in the file names because the fortran program naccess gets confused!
    nodot_decoys = [fname.replace("decoy.", "decoy_") for fname in confom_dict]
    for src, dst in zip(confom_dict, nodot_decoys):
        copyfile(decoy_dir + src, decoy_dir + dst)

    generate_cubes(nodot_decoys)
    results = dla_rank(acceptable_score)
    accepted = {k.replace(".lz4","").replace("decoy_", ""): v for k, v in results.items() if v >= acceptable_score}
    # clean up the no-dot files
    for file in nodot_decoys:
        remove(decoy_dir + file)

    # copy decoys with acceptavle score to a separate folder. If zero acceptable, we copy nothing
    rmtree(good_decoys, ignore_errors=True)
    os.makedirs(good_decoys)
    for dec_no in accepted:
        copy(f"{decoy_dir}decoy.{dec_no}.pdb", good_decoys)

    print(f"Good Decoys: {len(accepted)} out of {len(results)}")

if __name__ == '__main__':
    # decoy_dir = sys.argv[1]
    parser = argparse.ArgumentParser(description="DLA Ranker")
    parser.add_argument("threshold", type=float, help="")
    args = parser.parse_args()

    dla_ranker_filter(args.threshold)
