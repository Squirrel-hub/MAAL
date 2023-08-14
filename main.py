
import argparse
import numpy as np
import pandas as pd
import os
import shutil
import json

from active_learning import MultiAnnotatorActiveLearner
import os


def prepare_results_directory(args):  #create results and data directory
    args.Data_path = args.data_folder_path + "/" + args.dataset + ".csv"
    parent_dir_1 = "results/Knowledge_Base/"
    dir_name_1 = args.dataset + "-"+str(args.budget_frac) + "-" + str(args.similarity_threshold) + "-" + str(args.weight_threshold)

    Path_results = os.path.join(parent_dir_1,dir_name_1)
    if not os.path.isdir(Path_results):
        os.mkdir(Path_results)
    print("Knowledge Base Results directory path : ",Path_results)
    args.Path_results = Path_results

    parent_dir_2 = "data_processing"
    dir_name_2 = args.dataset + "-" + str(args.budget_frac)
    Path_mapal_data = os.path.join(parent_dir_2,dir_name_2)
    if not os.path.isdir(Path_mapal_data):
        os.mkdir(Path_mapal_data)
    print("MAPAL executed data directory path : ",Path_mapal_data)
    args.Path_mapal_data = Path_mapal_data

    #specify path for export text file
    args.exp_txt_path = Path_results +"/comparison.txt"

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset",default='iris-simulated-x')
    parser.add_argument("--data_folder_path", default="datasets")
    parser.add_argument("--budget_frac", type=float, default=0.4)
    parser.add_argument("--boot_size", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.4)
    # Boot
    parser.add_argument("--boot_lr", type=float, default=0.01)
    parser.add_argument("--boot_n_epochs", type=int, default=3000)
    parser.add_argument("--boot_log_epochs", type=int, default=30000)
    parser.add_argument("--boot_batchsize", type=int, default=4)
    # Active Learning
    parser.add_argument("--active_lr", type=float, default=0.02)
    parser.add_argument("--active_n_epochs", type=int, default=2000)
    parser.add_argument("--active_log_epochs", type=int, default=20000)
    parser.add_argument("--active_batchsize", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    #parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--instance_strategy", default="random") 
    parser.add_argument("--method",default="KB")

    parser.add_argument("--LR_max_iter", type=int, default=10000)
    parser.add_argument("--classifier_name", default="logistic_regression") # logistic_regression, neural_net, xgboost
    parser.add_argument("--f1_average", default="macro")
    # Misc
    parser.add_argument("--log_dir", default="logs/iris-simulated-x/")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--labeling_type", default="max") # "max" to obtain label corresponding to most confident annotator / "weight" 
    parser.add_argument("--exp_name", default="Trial_1")
    #Knowledge Base
    parser.add_argument("--use_Knowledge_Base",default=True)
    parser.add_argument("--similarity_threshold",default=0.95)
    parser.add_argument("--weight_threshold",default=0.7)
    #Data Source
    parser.add_argument("--data_source",default="mapal") # "mapal" to use Baseline generated training data / "fresh" to generate new training data
    parser.add_argument("--w_opt",default = 1 ) #1 for labels chosen using annotator model
    parser.add_argument("--rl_flag",type=int,default=0) # 0 for generic explore exploit
    parser.add_argument("--ee_ratio", type = float, default=0.8)

    args = parser.parse_args()

    prepare_results_directory(args)
    
    with open(args.exp_txt_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    AL = MultiAnnotatorActiveLearner(args)
    
    AL.fully_supervised() # Fully Supervised Metrics as upper limit

    AL.boot_phase() # Boot/ Warm-Up Phase

    AL.active_learning_phase() # Active Phase

    AL.save_metrics()

if __name__ == "__main__":
    main()