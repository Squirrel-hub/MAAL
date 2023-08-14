import os

os.environ['OMP_NUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import numpy as np
import pandas as pd
import math
import sys
import pickle
from pathlib import Path

sys.path.insert(0,'/Users/kaizer/Documents/Active Learning/Code/MAAL/Multi-Annotator-HIL') #INSERT YOUR PATH HERE TO IMPORT FROM SRC FOLDER


from functools import partial
from src.utils.statistic_functions import misclassification_costs

from sklearn.model_selection import train_test_split

import numpy as np
import os.path
import pandas as pd

from itertools import compress



def load_data(data_set_path):
    """
    Loads data set of given data set name.

    Parameters
    ----------
    data_set_name: str
        Name of the data set.

    Returns
    -------
    X: array-like, shape (n_samples, n_features)
        Samples as feature vectors.
    y_true: array-like, shape (n_samples)
        True class labels of samples.
    y: array-like, shape (n_samples, n_annotators_)
        Class label of each annotator (only available for grid data set).
    """
    
    data_set = pd.read_csv(data_set_path)
    columns = list(data_set.columns.values)
    features = list(compress(columns, [c.startswith('x_') for c in columns]))
    labels = list(compress(columns, [c.startswith('y_') for c in columns]))

    # Get features.
    X = data_set[features]

    # Getting assumed true labels.
    y_true = data_set['y']

    # Get labels of annotators.
    y = data_set[labels]

    return X, y_true, y



def get_data(x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,boot_size = 0.95,seed = 42):
    print('Train features shape, Train labels shape, Train Annotator Labels shape')
    print(x_train.shape, y_train.shape, y_annot_train.shape)
    print('Validation features Shape, Validation labels shape, Validation Annotators Label shape')
    print(x_val.shape,y_val.shape,y_annot_val.shape)
    print(x_train.shape,y_train.shape,y_annot_train.shape)
    x_boot, x_active, y_boot, y_active, y_annot_boot, y_annot_active = train_test_split(x_train, y_train, y_annot_train, test_size= 1-boot_size,stratify = y_train, random_state = seed)
    m = y_annot_boot.shape[1]

    print("boot up" , np.unique(y_boot,return_counts=True), len(x_boot))
    print("active up" , np.unique(y_active,return_counts=True))
    print("valid up" , np.unique(y_val,return_counts=True))

    print('Boot Data Features shape, Boot Data Labels shape, Boot Data Annotator Labels shape')
    print(x_boot.shape,y_boot.shape,y_annot_boot.shape)

    print('Active Data Features shape, Active Data Labels shape, Active Data Annotator Labels shape')
    print(x_active.shape,y_active.shape,y_annot_active.shape)

    TRAIN = [x_train, y_train, y_annot_train]
    VAL   = [x_val, y_val, y_annot_val]
    BOOT  = [x_boot, y_boot, y_annot_boot]
    ACTIVE = [x_active, y_active, y_annot_active]

    return TRAIN, VAL, BOOT, ACTIVE

def generate_MAPAL_data(data_dir,boot_size,seed):
    budget = 0.3
    test_ratio = 0.4
    seed = 1
    # /Users/kaizer/Documents/Active Learning/Code/MAAL/Multi-Annotator-HIL
    with open(data_dir+"/X_train", "rb") as fp:   # Unpickling
        x_train = pickle.load(fp)
    with open(data_dir+"/X_test", "rb") as fp:   # Unpickling
        x_val = pickle.load(fp)
    with open(data_dir+"/y_train", "rb") as fp:   # Unpickling
        y_annot_train = pickle.load(fp)
    with open(data_dir+"/y_test", "rb") as fp:   # Unpickling
        y_annot_val = pickle.load(fp)
    with open(data_dir+"/y_true_train", "rb") as fp:   # Unpickling
        y_train = pickle.load(fp)
    with open(data_dir+"/y_true_test", "rb") as fp:   # Unpickling
        y_val = pickle.load(fp)
    with open(data_dir+"/budget", "rb") as fp:   # Unpickling
        budget = pickle.load(fp)
    with open(data_dir+"/instance_annotator_pair", "rb") as fp:   # Unpickling
        instance_annotator_pair = pickle.load(fp)
    with open(data_dir+"/index_frame_train", "rb") as fp:   # Unpickling
        index_frame_train = pickle.load(fp)
    with open(data_dir+"/index_frame_test", "rb") as fp:   # Unpickling
        index_frame_test = pickle.load(fp)
    with open(data_dir+"/instances", "rb") as fp:   # Unpickling
        ordered_instances = pickle.load(fp)
    with open(data_dir+"/MAPAL_results_path", "rb") as fp:   # Unpickling
        MAPAL_results_path = pickle.load(fp)

    # x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,budget,m = run(Data_path,budget,test_ratio,seed)
    x_train = pd.DataFrame(x_train,index = index_frame_train)
    y_train = pd.Series(y_train,index = index_frame_train)
    y_annot_train = pd.DataFrame(y_annot_train,index = index_frame_train)

    x_val = pd.DataFrame(x_val,index = index_frame_test)
    y_val = pd.Series(y_val,index = index_frame_test)
    y_annot_val = pd.DataFrame(y_annot_val,index = index_frame_test)

    m = y_annot_train.shape[1]
    TRAIN, VAL, BOOT, ACTIVE = get_data(x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,boot_size,seed)
    print('MAPAL budget : ',budget)
    budget = budget - BOOT[0].shape[0]*m
    print("Our Budget : ",budget)

    new_x_train = x_train.loc[instance_annotator_pair.keys()]
    annot_index = []
    for x in list(instance_annotator_pair.values()):
        annot_index.append(x[-1])
    new_y_annot_train = y_annot_train.loc[instance_annotator_pair.keys()]
    new_y_train = []
    for i in range(len(annot_index)):
        new_y_train.append(new_y_annot_train.iloc[i][annot_index[i]])
    new_y_train = pd.Series(new_y_train,index = list(new_y_annot_train.index.values))

    Mapal_Data = [new_x_train,new_y_train,new_y_annot_train]
    return TRAIN, VAL, BOOT, ACTIVE, instance_annotator_pair, Mapal_Data, ordered_instances, budget, MAPAL_results_path

def generate_ActiveLab_MAPAL_data(data_dir,boot_size,seed):
    test_ratio = 0.4
    seed = 1
    # /Users/kaizer/Documents/Active Learning/Code/MAAL/Multi-Annotator-HIL
    with open(data_dir+"/X_train", "rb") as fp:   # Unpickling
        x_train = pickle.load(fp)
    with open(data_dir+"/X_test", "rb") as fp:   # Unpickling
        x_val = pickle.load(fp)
    with open(data_dir+"/y_train", "rb") as fp:   # Unpickling
        y_annot_train = pickle.load(fp)
    with open(data_dir+"/y_test", "rb") as fp:   # Unpickling
        y_annot_val = pickle.load(fp)
    with open(data_dir+"/y_true_train", "rb") as fp:   # Unpickling
        y_train = pickle.load(fp)
    with open(data_dir+"/y_true_test", "rb") as fp:   # Unpickling
        y_val = pickle.load(fp)
    with open(data_dir+"/budget", "rb") as fp:   # Unpickling
        budget = pickle.load(fp)
    with open(data_dir+"/instance_annotator_pair", "rb") as fp:   # Unpickling
        instance_annotator_pair = pickle.load(fp)
    with open(data_dir+"/index_frame_train", "rb") as fp:   # Unpickling
        index_frame_train = pickle.load(fp)
    with open(data_dir+"/index_frame_test", "rb") as fp:   # Unpickling
        index_frame_test = pickle.load(fp)
    with open(data_dir+"/instances", "rb") as fp:   # Unpickling
        ordered_instances = pickle.load(fp)
    with open(data_dir+"/MAPAL_results_path", "rb") as fp:   # Unpickling
        MAPAL_results_path = pickle.load(fp)

    # x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,budget,m = run(Data_path,budget,test_ratio,seed)
    x_train = pd.DataFrame(x_train,index = index_frame_train)
    y_train = pd.Series(y_train,index = index_frame_train)
    y_annot_train = pd.DataFrame(y_annot_train,index = index_frame_train)

    x_val = pd.DataFrame(x_val,index = index_frame_test)
    y_val = pd.Series(y_val,index = index_frame_test)
    y_annot_val = pd.DataFrame(y_annot_val,index = index_frame_test)

    m = y_annot_train.shape[1]
    TRAIN, VAL, BOOT, ACTIVE = get_data(x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,boot_size,seed)
    print('MAPAL budget : ',budget)
    budget = budget - BOOT[0].shape[0]
    print("Boot budget : ",BOOT[0].shape[0])
    print("Our active Budget : ",budget)

    new_x_train = x_train.loc[instance_annotator_pair.keys()]
    annot_index = []
    for x in list(instance_annotator_pair.values()):
        annot_index.append(x[-1])
    new_y_annot_train = y_annot_train.loc[instance_annotator_pair.keys()]
    new_y_train = []
    for i in range(len(annot_index)):
        new_y_train.append(new_y_annot_train.iloc[i][annot_index[i]])
    new_y_train = pd.Series(new_y_train,index = list(new_y_annot_train.index.values))

    Mapal_Data = [new_x_train,new_y_train,new_y_annot_train]
    return TRAIN, VAL, BOOT, ACTIVE, instance_annotator_pair, Mapal_Data, ordered_instances, budget, MAPAL_results_path

    
def generate_new_data (data_path,budget,test_ratio = 0.4, boot_size = 0.05,seed = 42):
    path = Path(data_path)
    df = pd.read_csv(path)

    cols = df.columns
    features = []
    label = []
    for c in cols:
        if 'x' in c:
            features.append(c)
        else:
            label.append(c)
    
    n_annotators = len(label) - 1

    X = df[features]
    y = df['y']
    y_annot = df[label]
    y_annot = y_annot.drop(['y'], axis=1)

    x_train, x_val, y_train, y_val, y_annot_train, y_annot_val = train_test_split(X, y, y_annot, test_size=0.4, random_state=seed)
    TRAIN, VAL, BOOT, ACTIVE = get_data(x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,boot_size,seed)
    m = y_annot_train.shape[1]

    if budget > len(X) * n_annotators * (1 - test_ratio):
        budget = int(math.floor(len(X) * n_annotators * (1 - test_ratio)))
    elif budget > 1:
        budget = int(budget)
    elif 0 < budget <= 1:
        budget = int(math.floor(len(X) * n_annotators * (1 - test_ratio) * budget))
    else:
        raise ValueError("'budget' must be a float in (0, 1] or an integer in [0, n_samples]")
    budget= np.min((budget, 1000))

    print('MAPAL budget : ',budget)
    
    budget = budget - m*BOOT[2].shape[0]

    print('Our Budget : ',budget)
    return TRAIN, VAL, BOOT, ACTIVE, budget

def generate_new_data_ActiveLab (data_path,budget,test_ratio = 0.4, boot_size = 0.05,seed = 42):
    path = Path(data_path)
    df = pd.read_csv(path)

    cols = df.columns
    features = []
    label = []
    for c in cols:
        if 'x' in c:
            features.append(c)
        else:
            label.append(c)
    
    n_annotators = len(label) - 1

    X = df[features]
    y = df['y']
    y_annot = df[label]
    y_annot = y_annot.drop(['y'], axis=1)

    x_train, x_val, y_train, y_val, y_annot_train, y_annot_val = train_test_split(X, y, y_annot, test_size=0.4, random_state=seed)
    TRAIN, VAL, BOOT, ACTIVE = get_data(x_train,x_val,y_train,y_val,y_annot_train,y_annot_val,boot_size,seed)
    m = y_annot_train.shape[1]

    if budget > len(X) * n_annotators * (1 - test_ratio):
        budget = int(math.floor(len(X) * n_annotators * (1 - test_ratio)))
    elif budget > 1:
        budget = int(budget)
    elif 0 < budget <= 1:
        budget = int(math.floor(len(X) * n_annotators * (1 - test_ratio) * budget))
    else:
        raise ValueError("'budget' must be a float in (0, 1] or an integer in [0, n_samples]")
    budget= np.min((budget, 1000))

    print('MAPAL budget : ',budget)
    
    budget = budget - BOOT[2].shape[0]
    print("Boot budget : ",BOOT[0].shape[0])
    print('Our Budget : ',budget)
    return TRAIN, VAL, BOOT, ACTIVE, budget

