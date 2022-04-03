### Contents

- [Overview](#overview)
- [Requirements](#Requirements)
- [Quick Tutorial](#Tutorial)
- [Documentation](https://deeprank.readthedocs.io/)
- [License](./LICENSE)
- [Issues & Contributing](#Issues-and-Contributing)

## Overview
![](Images/method5.svg.png?raw=true "DLA-Ranker")

Deep Local Analysis (DLA)-Ranker is a deep learning framework applying 3D convolutions to a set of locally oriented cubes representing the protein interface. It explicitly considers the local geometry of
the interfacial residues along with their neighboring atoms and the regions of the interface with different solvent accessibility. DLA-Ranker identifies near-native conformations and discovers alternative interfaces from ensembles generated by molecular docking.

#### Features:

- Useful APIs for fast preprocessing of huge assembly of the complex conformations and classify them based on CAPRI criteria. 

- Representation of an interface as a set of locally oriented cubes.
   - *Atomic density map as a 3D gird.*
   - *Structure class based on solvant accessibility (Support, Core, Rim).*
   - *Information on Receptor and Ligand.*
   - *Information of interfacial residues.*

- Classification of docking conformations based on CAPRI criteria (Incorrect, Acceptable, Medium, High quality)

- Fast generation of cubes and and evaluation of interface.

- Training and testing 3D-CNN models.

- Support various per-score aggregation schemes.
   - *Considering only subset cubes for evaluation of interface.*
   - *Residues from Support or Core regions.*
   - *Residues from Core or Rim regions.*
   - *Selecting residues exclusively from the receptor or from the ligand.*

- Extraction of embeddings and the topology of the interface for graph representation learning.



## Requirements

#### Packages:

DLA-Ranker can be run on Linux, MacOS, and Windows. We recommend to use DLA-Ranker on the machines with GPU. It requires following packages:
- Python version 3.7 or 3.8.
- Tensorflow version 2.2 or 2.3.
- Cuda-Toolkit
- Scikit-Learn, numpy pandas matplotlib lz4 and tqdm (conda install -c pytorch -c pyg -c conda-forge python=3.9 numpy pandas matplotlib tqdm pytorch pyg scikit-learn cuda-toolkit lz4).

All-in-one: Run conda create --name dla-ranker --file dla-ranker.yml

- For requirements of InteractionGNN please visit its Readme.

## Tutorial

DLA-Ranker works in two steps:

-   Generating a set of locally orient cubes representing the interface.
-   Running the deep learning framework to:
   - *Train: creating a new model.*
   - *Test: Evaluating conformations using trained models.*
   - *Encode: Extracting embeddings and the topology of the interface.*

### Generating locally oriented cubes

#### Dataset of conformations:
Place the ensemble of conformations in a directory (*e.g. 'Examples/conformations_directory'*) like below: 

```
Example
|___conformations_list
|
|___conformations_directory
    |
    |___target complex 1
    |   |   Conformation 1
    |   |   Conformation 2
    |   |   ...
    |
    |___target complex 2
    |   |   Conformation 1
    |   |   Conformation 2
    |   |   ...
    |
    ..........
```

'conformations_list' is a csv file that contains five columns separated by ';': Name of target complex (`Comp`); receptor chain ID(s) (`ch1`), ligand chain ID(s) (`ch2`); Name of the conformation file (`Conf`); class of the conformation (`Class`, 0:incorrect, 1: near-native).


#### Processing the conformations
From directory 'Representation' run: ```python generate_cubes.py```

The output will be directory 'map_dir' with the following structure:

```
Example
|___map_dir
    |___target complex 1
    |   |___0
    |   |   |   conformation 1
    |   |   |   conformation 2
    |   |
    |   |___1
    |       |   conformation 3
    |       |   conformation 4
    |   
    |___target complex 2
    |   |___0
    |   |   |   conformation 1
    |   |   |   conformation 2
    |   |
    |   |___1
    |       |   conformation 3
    |       |   conformation 4
    ..........
```

Each output represents interface of a conformation and contains a set of local environments (*e.g. atomic density map, structure classes (S,C,R), topology of the interface, ...*)

### Deep learning framework

Following commands will use the trained models that can be found in the directory 'Models'. This directory includes 3 sets of models:

'BM5': 10 models generated following 10-fold cross validation procedure on the 142 dimers of the Docking Benchmakr version 5. The docking conformations had been generated by HADDOCK.
'Dockground': 4 models generated following 4-fold cross validation procedure on the 59 target complexes of the Dockground database. The docking conformations had been generated by GRAMM.
'CCD4PPI': 5 models generated following 5-fold cross validation procedure on the 400 target complexes.

For detailed information please read the article.

#### Evaluation of interfaces
From directory 'Test' run ```python test.py```
It processes all the target complexes and their conformations and produces csv file 'predictions_SCR'. Each row of the output file belongs to a conformation and it has 9 columns separated by 'tab':

Name of target complex and the conformation (`Conf`) 
Fold Id (`Fold`)
Score of each residue (`Scores`)
Region (SCR) of each residue (`Regions`)
Global averaged score of the interface (`Score`)
Processing time (`Time`)
Class of the conformation (`Class`, 0:incorrect, 1: near-native)
Partner (`RecLig`)
Residue number (`ResNumber`; according to PDB)

One can associate the Residues' numbers, regions, scores, and partner to evaluate the interface on a subset of interfacial residues.

#### Extraction of the embeddings
From directory 'Test' run ```python extract_embeddings.py```
It extracts embeddings and the topology for each given interface and write them in a an output file with the same name. Each row in a file belongs to a residue and includes the its coordinates, its region, and its embedding vector. These files can be used for aggregation of embeddings based on graph-learning.

