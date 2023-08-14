from itertools import compress
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import os
import warnings
import random
import pandas as pd
import matplotlib.pyplot as plt
from classifier.warmup import warmedup_classifiers
from annotator.warmup import annotator_warmup
from annotator.evaluation import annot_eval_after_warmup,annotator_AL_loss,annot_eval_after_training
from AL_design.train_scheme import AL_train_cycle_KB,AL_train_cycle,AL_train_majority,AL_train_MAPAL_instances
from data_processing.read_data import generate_MAPAL_data,generate_new_data

# from annotator.annotator_selector import AnnotatorSelector
from instance_selector import InstanceSelector
from classifier.classifier import ClassifierModel
from annotator.annotator_selector import AnnotatorSelector
#from utils import get_weighted_labels, get_max_labels
from knowledge_base.utils import create_knowledge_base,count_instances,Knowledge_Base_Metrics
from classifier.evaluation import eval_model,classf_eval_after_warmup,classf_eval_after_training, compare_true_label,print_scores,classifier_Val_scores_during_AL

class MultiAnnotatorActiveLearner:
    def __init__(self, args):

        self.args = args
        self.create_data_splits(args)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.classifier_annotator_scores_log = self.create_classifier_annotator_metrics_dictionary()

        self.annotator_selector = AnnotatorSelector(n_features=self.n_features, n_annotators=self.n_annotators, hidden_dim = self.args.hidden_dim, seed = self.args.seed, device=self.device, log_dir=args.log_dir + args.exp_name, report_to_tensorboard=True)  
        # self.instance_selector = InstanceSelector(self.args.instance_strategy, self.args.seed)
        self.classifier_FS = ClassifierModel(self.args.classifier_name, self.n_features, self.n_classes, LR_max_iter = self.args.LR_max_iter, device=self.device)
        self.Classifiers = [ClassifierModel(self.args.classifier_name, self.n_features, self.n_classes, LR_max_iter = self.args.LR_max_iter, device=self.device) for i in range(4)]

        self.Knowledge_Base = dict()
    
    def set_seed(self,seed: int = 1) -> None:  
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set as {seed}")
    
    def create_classifier_annotator_metrics_dictionary(self):
        scores = dict()
        scores['data_c'] = [{},{},{},{}]
        scores['data_a'] = [{},{}]
        # scores['data_c_AM'] = [{},{},{},{}] ## Store Classifier Metrics corresponsind to classifier trained on Annotator Model Labels
        # scores['data_c_KB'] = [{},{},{},{}]  ## Store Clasifier Metrics corresponding to classifier traned with Knowledge Base
        # scores['data_c_WO'] = [{},{},{},{}] ## Store Classifier Metrics corresponsind to classifier trained on W Optimal Labels
        # scores['data_c_AM_EE'] = [{},{},{},{}] ## Store Classifier Metrics corresponsind to classifier trained on Annotator Model Labels with Exploration-Exploitation
        # scores['data_c_M'] = [{},{},{},{}]    ## Store Classifier Metrics corresponsind to classifier trained on Majority Labels
        # scores['data_c_MAPAL'] = [{},{},{},{}] ## Store Classifier Metrics corresponsind to classifier trained on MAPAL selected instances 
        # scores['data_a_AM'] = [{},{}] ## Store Annotator Metrics corresponsind to classifier trained on Annotator Model Labels
        # scores['data_a_KB'] = [{},{}] ## Store Annotator Metrics corresponding to classifier traned with Knowledge Base
        # scores['data_a_WO'] = [{},{}]  ## Store Annotator Metrics corresponsind to classifier trained on W Optimal Labels
        # scores['data_a_AM_EE'] = [{},{}]  ## Store Annotator Metrics corresponsind to classifier trained on Annotator Model Labels with Exploration-Exploitation
        # scores['data_a_M'] = [{},{}]     ## Store Annotator Metrics corresponsind to classifier trained on Majority Labels
        # scores['data_a_MAPAL'] = [{},{}]  ## Store Classifier Metrics corresponsind to classifier trained on MAPAL selected instances 
        return scores
    
    def create_data_splits(self, args):

        if args.data_source == "mapal":
            dir = os.listdir(args.Path_mapal_data)
  
            # Checking if the list is empty or not
            if len(dir) == 0:
                print("PLEASE EXECUTE MAPAL")
                exit(1)
            
            TRAIN, TEST, BOOT, ACTIVE, instance_annotator_pair, Mapal_Instances, ordered_instances, budget, MAPAL_results_path = generate_MAPAL_data(args.Path_mapal_data,args.boot_size,args.seed)
            self.train_x, self.train_y, self.train_annotator_labels = TRAIN
            self.test_x, self.test_y, self.test_annotator_labels = TEST
            self.boot_x, self.boot_y, self.boot_annotator_labels = BOOT
            self.active_x, self.active_y, self.active_annotator_labels = ACTIVE
            self.mapal_x,self.mapal_y,self.mapal_annotator_labels = Mapal_Instances
            self.instance_annotator_pair = instance_annotator_pair
            self.ordered_instances = ordered_instances
            self.budget = budget
            Mapal_Data_Frame = pd.read_csv(MAPAL_results_path)
            Mapal_Data_Frame_len = Mapal_Data_Frame.shape[0]
            self.Mapal_accuracy = 1 - Mapal_Data_Frame['test-micro-misclf-rate'].loc[Mapal_Data_Frame_len-1]
            with open(args.exp_txt_path, 'a') as f:
                f.write("\n\nMAPAL Accuracy : "+str(self.Mapal_accuracy)+"\n\n")

        else:

            TRAIN, TEST, BOOT, ACTIVE, budget = generate_new_data(args.Data_path,budget = args.budget_frac,test_ratio=args.test_size,boot_size = args.boot_size, seed = args.seed)
            self.train_x, self.train_y, self.train_annotator_labels = TRAIN
            self.test_x, self.test_y, self.test_annotator_labels = TEST
            self.boot_x, self.boot_y, self.boot_annotator_labels = BOOT
            self.active_x, self.active_y, self.active_annotator_labels = ACTIVE
            

        self.n_features = self.boot_x.shape[1]
        self.n_annotators = self.boot_annotator_labels.shape[1]
        self.n_classes = len(np.unique(self.train_y))
        with open(args.exp_txt_path, 'a') as f:
            f.write("\n\nNumber of classes : "+str(self.n_classes)+"\n\n")

        print(f"Boot Instances: {self.boot_x.shape[0]} \t Active Instances: {self.active_x.shape[0]} \t Test Instances: {self.test_x.shape[0]}")
        print(f"n_features: {self.boot_x.shape[1]}")
        print(f"n_annotators: {self.boot_annotator_labels.shape[1]}")

    def boot_phase(self):
        training_args = {}
        training_args["lr"] = self.args.boot_lr
        training_args["n_epochs"] = self.args.boot_n_epochs
        training_args["log_epochs"] = self.args.boot_log_epochs
        training_args["labeling_type"] = self.args.labeling_type
        training_args["batch_size"] = self.args.boot_batchsize
        training_args["f1_score"] = self.args.f1_average
        training_args["Path_results"] = self.args.Path_results
        # Train annotator model on boot data with all annotators
        boot_mask = np.ones_like(self.boot_annotator_labels)

        # self.set_seed(self.args.seed)

        self.boot_optimal_weights,loss_list = self.annotator_selector.warmup( self.boot_x, self.boot_annotator_labels,mask = boot_mask, batch_size=self.args.boot_batchsize,\
                                                              n_epochs= self.args.boot_n_epochs, learning_rate = self.args.boot_lr, device = self.device)
        BOOT = [self.boot_x, self.boot_y, self.boot_annotator_labels]
        TEST = [self.test_x, self.test_y, self.test_annotator_labels]

        # path = os.path.join(self.args.Path_results,"Annotator_Warmup.png")
        # plt.plot(loss_list)
        # plt.xlabel("warmup epochs")
        # plt.ylabel("loss")
        # plt.title("Annotator Warmup")
        # plt.savefig(path)
        
        self.Classifiers, self.Classifiers_y_boot = warmedup_classifiers(self.Classifiers, BOOT,self.annotator_selector.model,self.boot_optimal_weights,self.device)
        
        if self.args.use_Knowledge_Base: 
            annot_eval_after_warmup(self.annotator_selector.model,BOOT,self.classifier_annotator_scores_log["data_a"],self.args.f1_average)
            classf_eval_after_warmup(self.Classifiers,BOOT,TEST,self.classifier_annotator_scores_log["data_c"],average = self.args.f1_average)
        else:
            annot_eval_after_warmup(self.annotator_selector.model,BOOT,self.classifier_annotator_scores_log["data_a"],self.args.f1_average)
            classf_eval_after_warmup(self.Classifiers,BOOT,TEST,self.classifier_annotator_scores_log["data_c"],average = self.args.f1_average)
        
        create_knowledge_base(self.Knowledge_Base,self.annotator_selector.model,BOOT)
        Knowledge_Base_Metrics(self.Knowledge_Base,self.boot_y,self.args.Path_results,self.args.exp_txt_path,average=self.args.f1_average)
    
    def active_learning_phase(self):

        TRAIN  = [self.train_x, self.train_y, self.train_annotator_labels]
        TEST   = [self.test_x, self.test_y, self.test_annotator_labels]
        BOOT   = [self.boot_x, self.boot_y, self.boot_annotator_labels]
        ACTIVE = [self.active_x, self.active_y, self.active_annotator_labels]

        if self.args.method == "KB":
            self.Classifiers, self.annotator_selector.model,indexes,collected_active_data,similar_instances,loss, inst_annot,full, c_a, c_f = AL_train_cycle_KB(self.Classifiers,self.Classifiers_y_boot,self.annotator_selector,self.Knowledge_Base,TRAIN.copy(),BOOT.copy(),ACTIVE.copy(),TEST.copy(),
                self.boot_optimal_weights,self.budget, self.args, device = self.device) 
            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask = collected_active_data
            count_instances(self.args.exp_txt_path,self.Knowledge_Base,new_active_x) 
            Knowledge_Base_Metrics(self.Knowledge_Base,new_active_y,self.args.Path_results,self.args.exp_txt_path,average=self.args.f1_average)
        elif self.args.method == "AM":
            self.Classifiers, self.annotator_selector.model,indexes,collected_active_data,loss, inst_annot,full, c_a, c_f = AL_train_cycle(self.Classifiers,self.Classifiers_y_boot,self.annotator_selector,BOOT.copy(),ACTIVE.copy(),TEST.copy(),
                self.annotator_selector.boot_optimal_weights,self.budget, self.args, device = self.device) 
            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask = collected_active_data
        elif self.args.method == 'MAJ':
            self.Classifiers,indexes,collected_data,loss, inst_annot,full, c_a, c_f = AL_train_majority(self.Classifiers,self.Classifiers_y_boot,BOOT.copy(),ACTIVE.copy(),TEST.copy(),
                self.annotator_selector.boot_optimal_weights,self.budget, device = self.device)
            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask = collected_data
        elif self.args.method == 'MAPAL':
            Mapal_data = [self.mapal_x,self.mapal_y,self.mapal_annotator_labels]
            self.Classifiers, self.annotator_selector.model,collected_active_data, loss, inst_annot, full, c_a,c_f = AL_train_MAPAL_instances(self.Classifiers,self.Classifiers_y_boot,self.annotator_selector,self.ordered_instances,BOOT.copy(),TEST.copy(),
                Mapal_data,self.annotator_selector.boot_optimal_weights,self.args, device = self.device)
            new_active_x,new_active_y,new_active_y_opt,new_active_y_majority, new_active_y_true,new_active_y_annot, new_active_mask = collected_active_data
        
        # print('ON TRAINING DATA ')
        compare_true_label(new_active_y,new_active_y_opt,new_active_y_majority,new_active_y_true,self.args.exp_txt_path,average=self.args.f1_average)

        self.samples_composition(inst_annot,full,self.args.Path_results,self.args.exp_txt_path)

        annotator_AL_loss(loss,self.args.Path_results)

        classifier_Val_scores_during_AL(c_a,c_f,self.args.Path_results)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            annot_eval_after_training(self.annotator_selector.model,new_active_x,new_active_y_true,new_active_y_annot,new_active_mask,self.classifier_annotator_scores_log["data_a"],average=self.args.f1_average)
            classf_eval_after_training(self.Classifiers,new_active_x,new_active_y_opt,new_active_y_majority,new_active_y_true,TEST.copy(),self.classifier_annotator_scores_log["data_c"],average=self.args.f1_average)


    def fully_supervised(self):

        self.classifier_FS.train(self.train_x, self.train_y)
        accuracy, f1 = self.classifier_FS.eval(self.test_x, self.test_y,self.args.f1_average)

        with open(self.args.exp_txt_path, 'a') as f:
            f.write("\n\nFully Supervised\n")
            f.write("\nAccuracy : "+str(accuracy))
            f.write("\nF1 Score : "+str(f1))
    
    def save_metrics(self):
        path_classifier = os.path.join(self.args.Path_results,"Classifier_Metrics.csv")
        path_annotator = os.path.join(self.args.Path_results,"Annotator_Metrics.csv")

        dataframe_classifier_metrics = pd.DataFrame(self.classifier_annotator_scores_log["data_c"],index = ['Annotator Model', 'W Optimal', 'Majority','True Labels'])
        dataframe_annotator_metrics = pd.DataFrame(self.classifier_annotator_scores_log["data_a"],index=['Weighted Average','Maximum Index'])

        dataframe_classifier_metrics.to_csv(path_classifier)
        dataframe_annotator_metrics.to_csv(path_annotator)

        #export DataFrame to text file
        with open(self.args.exp_txt_path, 'a') as f:
            df_string_1 = dataframe_classifier_metrics.to_string(header=True, index=True)
            df_string_2 = dataframe_annotator_metrics.to_string(header=True, index=True)
            f.write("\n\n\nCLASSIFIER METRICS\n\n")
            f.write(df_string_1)
            f.write("\n\n\nANNOTATOR METRIC\n\n")
            f.write(df_string_2)
    
    def samples_composition(self,inst_annot,full,Path,exp_txt_path):
    
        unique, frequency = np.unique(inst_annot.sum(1),return_counts=True)
        unique = unique.tolist()
        frequency = frequency.tolist()
        num = [int(i) for i in unique]
        num.append(inst_annot.shape[1])
        frequency.append(full)

        with open(exp_txt_path, 'a') as f:
            f.write("\n\nCompostion of instance and number of queried annotators\n")
            f.write(str(num)+"\n")
            f.write(str(frequency))

        fig = plt.figure(figsize = (10, 5))
        
        # creating the bar plot
        plt.bar(num, frequency, color ='yellow',
                width = 0.4)
        
        plt.xlabel("No. of Annotators Queried")
        plt.ylabel("No. of Instances")
        plt.title("Diversity of samples")

        if not Path == None:
            path = os.path.join(Path,"Diverse_Samples.png")
            plt.savefig(path)
        plt.show()