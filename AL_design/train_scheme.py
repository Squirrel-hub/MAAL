import numpy as np
import torch
import torch.nn as nn
import tqdm
import pandas as pd
import random
import warnings
from classifier.utils import find_entropy,find_index,find_index_RL,Average
from classifier.evaluation import eval_model
from Lpp.utils import optimal_weights
from Loaders.dataset_loaders import Data,Data_AL
from torch.utils.data import Dataset, DataLoader
from annotator.Model import Annotator
import matplotlib.pyplot as plt
from knowledge_base.utils import Knowledge_Base_similarity
from annotator.trainer import annotator_training



def append_dataframe(x_new,y_true,y_annot,y_new,y_new_opt,y_new_majority,w_new,masks_new,x_boot,y_annot_boot,classifier_y_boot,W_optimal,m):
    
    X_new = list(x_new.values())
    Y_true = list(y_true.values())
    Y_annot = list(y_annot.values())
    Y_new = list(y_new.values())
    Y_new_opt = list(y_new_opt.values())
    Y_new_majority = list(y_new_majority.values())
    W_new = list(w_new.values())
    Masks_new = list(masks_new.values())
    
    # CREATING DATAFRAME OF ALL COLLECTED DATA
    df_X_new = pd.DataFrame(X_new, index = list(x_new.keys()))
    df_Y_true = pd.DataFrame(Y_true, index = list(y_true.keys()))
    df_Y_annot = pd.DataFrame(Y_annot,index = list(y_annot.keys()))
    df_Y_new = pd.DataFrame(Y_new, index = list(y_new.keys()))
    df_Y_new_opt = pd.DataFrame(Y_new_opt, index = list(y_new_opt.keys()))
    df_Y_new_majority = pd.DataFrame(Y_new_majority, index = list(y_new_majority.keys()))
    df_W_new = pd.DataFrame(W_new, index = list(w_new.keys()))
    df_M_new = pd.DataFrame(Masks_new, index = list(masks_new.keys()))

    # CONCATINATING DATAFRAME WITH THE BOOT DATA
    new_active_x = pd.concat([x_boot, df_X_new])
    new_active_y_true = pd.concat([classifier_y_boot[3], df_Y_true])
    new_active_y_annot = pd.concat([y_annot_boot,df_Y_annot])
    new_active_y = pd.concat([classifier_y_boot[0], df_Y_new])
    new_active_y_opt = pd.concat([classifier_y_boot[1],df_Y_new_opt])
    new_active_y_majority = pd.concat([classifier_y_boot[2],df_Y_new_majority])
    W_optimal = pd.DataFrame(W_optimal,index = list(x_boot.index.values))
    new_active_w = pd.concat([W_optimal,df_W_new])
    boot_masks = torch.ones((x_boot.shape[0],m))
    new_active_mask = pd.concat([pd.DataFrame(boot_masks,index=list(x_boot.index.values)),df_M_new])
    
    return new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask

def get_instance_annot_index(classifier_model,annotator_model,x_active,y_annot_active,inst_annot,device='cpu',choice=0,ee_ratio = 0,label_choice = 1):
    
    output_c = classifier_model.predict_proba(x_active) #Classifier Output on Active Data
    annotator_model = annotator_model.to(device)
    
    index_frame = list(x_active.index.values) # Inherent Index column of the Dataframe
    if choice == 0 :
        index = find_index(output_c,index_frame) # Index corresponding to inherent index of the Dataframe
    else :
        if random.uniform(0,1) >= ee_ratio: #Explore
            index = find_index_RL(output_c,index_frame,inst_annot,0) # EXPLORATION --- Index corresponding to inherent index of the Dataframe
        else :
            index = find_index_RL(output_c,index_frame,inst_annot,1) # EXPLOITATION --- Index corresponding to inherent index of the Dataframe
    row_index = index_frame.index(index) # Row Index
    
    output_a = annotator_model(torch.from_numpy(x_active.loc[index].to_numpy()).to(device).float()) # Annotator Model weights for a given instance

    output_a_1 = torch.mul(1-inst_annot[row_index].to(device),output_a) ## CHANGE VARIABLE NAMES
    annot_index_new = torch.argmax(output_a_1,dim=0) # Index of Annotator with max weight which is not queried yet for that particular instance

    inst_annot[row_index][annot_index_new] = 1

    output_a_2 = torch.mul(inst_annot[row_index].to(device),output_a) ## CHANGE VARIABLE NAMES

    if label_choice == 1:
        annot_index = torch.argmax(output_a_2,dim=0).item() # Index of Annotator with max weight amongst all queried annotators for that particular instance
    else:
        num_lab = np.unique(y_annot_active.loc[index].to_numpy())
        output_a_3 = np.delete(output_a_2.detach().numpy(), np.where(inst_annot[row_index] == 0))
        labels =  np.delete(y_annot_active.loc[index].to_numpy(), np.where(inst_annot[row_index] == 0))
        annotator_ids = [i for i in range(y_annot_active.shape[1])]
        annotator_ids =  np.delete(annotator_ids, np.where(inst_annot[row_index] == 0))
        max_weight = 0
        annot_index_list = list()
        max_weight_list = list()
        for lab in num_lab:
            weight = []
            a_i = []
            for w,l,id in zip(output_a_3,labels,annotator_ids):
                if l == lab:
                    weight.append(w)
                    a_i.append(id)
            if len(weight) == 0:
                average_weight = 0
            else:
                average_weight = sum(weight)/len(weight)
            if average_weight > max_weight:
                max_weight = average_weight
                annot_index_list = a_i
                max_weight_list = weight
        
        annot_index = annot_index_list[max_weight_list.index(max(max_weight_list))]

    return index_frame,index,row_index,annot_index,output_a


def get_labels(classifier_model,annotator_model,active_data,collected_data,inst_annot,device,choice = 0,ee_ratio = 0.8,label_choice = 1):

    x_active,y_active,y_annot_active = active_data
    x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new = collected_data

    index_frame,index,row_index,annot_index,_ = get_instance_annot_index(classifier_model,annotator_model,x_active,y_annot_active,inst_annot,device,choice,ee_ratio,label_choice)

    x_new[index] = x_active.loc[index] # Current Instance Feature
    y_true[index] = y_active.loc[index]
    y_annot[index] = y_annot_active.loc[index]
    y_new[index] = y_annot_active.loc[index][annot_index] # Label of Current Instance as per the annotator model
    if torch.sum(inst_annot[row_index]).item() == 1:
        w_new[index] = inst_annot[row_index].numpy()
    else:
        w_new[index] = optimal_weights(y_annot_active.loc[index].to_numpy(),inst_annot[row_index]) # Optimal Weights for this particular instance based on only queried annotators
    annot_index_opt = torch.argmax(torch.mul(torch.tensor(w_new[index]),inst_annot[row_index])) # Annotator Index of Current Instance as per the highest W_optimal obtained from only queried annotators
    y_new_opt[index] = y_annot_active.loc[index][annot_index_opt.item()] # Label of Current Instance as per the highest W_optimal obtained from only queried annotators
    y_maj = np.array(y_annot_active.loc[index]).copy()
    y_maj = np.delete(y_maj, np.where(inst_annot[row_index] == 0))
    unique, frequency = np.unique(y_maj,return_counts = True)
    unique = unique.tolist()
    frequency = frequency.tolist()
    majority_index = np.argmax(frequency) # Index of the majority Label. Note: All Annotators are considered for the instance
    y_new_majority[index] = unique[majority_index] # Majority Label of the given Instance
    masks_new[index] = inst_annot[row_index].numpy() # Mask representing all annotators which are queried for this Instance
    collected_data = [x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new]
    return index_frame,index,row_index,annot_index,collected_data

def update_data(index,Lab,collected_data):
    
    x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new = collected_data

    x_new_dummy = x_new[index]
    y_true_dummy = y_true[index]
    y_annot_dummy = y_annot[index]
    y_new_dummy = Lab
    w_new_dummy = w_new[index]
    y_new_opt_dummy = y_new_opt[index]
    y_new_majority_dummy = y_new_majority[index]
    masks_new_dummy = masks_new[index]


    x_new[index] = x_new_dummy 
    y_true[index] = y_true_dummy
    y_annot[index] = y_annot_dummy
    y_new[index] = y_new_dummy 
    w_new[index] = w_new_dummy
    y_new_opt[index] = y_new_opt_dummy
    y_new_majority[index] = y_new_majority_dummy
    masks_new[index] = masks_new_dummy

    collected_data = [x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new]
    return collected_data



def get_labels_KB(classifier_model,annotator_model,Knowledge_Base,similar_instances,TRAIN,ACTIVE,collected_data,inst_annot,device,RL_flag = 1,ee_ratio = 0.8,label_choice = 1,similarity_threshold = 0.9,weight_threshold = 0.7):

    x_active,y_active,y_annot_active = ACTIVE
    x_train, y_train, y_annot_train = TRAIN
    index_frame,index,row_index,annotator_AM,output_AM = get_instance_annot_index(classifier_model,annotator_model,x_active,y_annot_active,inst_annot,device,RL_flag,ee_ratio,label_choice)
    data = x_active.loc[index].to_numpy()
    data = torch.tensor(data,dtype=torch.float32)

    epoch_iter = 1
    # check number of extra instances we have obtained
    
    if torch.sum(inst_annot[row_index]).item() == 1: # EXPLORATION

        annotator_KB, similarity, similar_instance,lab = Knowledge_Base_similarity(data,Knowledge_Base,x_train)
        if similarity > similarity_threshold :
            if annotator_AM == annotator_KB:
                Lab = lab 
                if similar_instance in Knowledge_Base[annotator_KB]['similar_instances']:
                    Knowledge_Base[annotator_KB]['similar_instances'][similar_instance]['index'].append(index)
                    Knowledge_Base[annotator_KB]['similar_instances'][similar_instance]['label'] = Lab
                else:
                    Knowledge_Base[annotator_KB]['similar_instances'][similar_instance] = dict()
                    Knowledge_Base[annotator_KB]['similar_instances'][similar_instance]['index'] = [index]
                    Knowledge_Base[annotator_KB]['similar_instances'][similar_instance]['label'] = Lab
                epoch_iter = 0 
            else:
                Lab = y_annot_active.loc[index][annotator_AM] # TAKING LABEL FROM KNOWLEDGE BASE 
                if output_AM[annotator_AM] > weight_threshold:
                    if annotator_AM in Knowledge_Base:
                        Knowledge_Base[annotator_AM]['index'].append(index)
                        Knowledge_Base[annotator_AM]['label'].append(Lab)
                    else:
                        Knowledge_Base[annotator_AM] = dict()
                        Knowledge_Base[annotator_AM]['index'] = [index]
                        Knowledge_Base[annotator_AM]['label'] = [Lab]
                        Knowledge_Base[annotator_AM]['similar_instances'] = dict()
                
        else :
            Lab = y_annot_active.loc[index][annotator_AM] # TAKING LABEL FROM KNOWLEDGE BASE 
            if output_AM[annotator_AM] > weight_threshold:
                if annotator_AM in Knowledge_Base:
                    Knowledge_Base[annotator_AM]['index'].append(index)
                    Knowledge_Base[annotator_AM]['label'].append(Lab)
                else:
                    Knowledge_Base[annotator_AM] = dict()
                    Knowledge_Base[annotator_AM]['index'] = [index]
                    Knowledge_Base[annotator_AM]['label'] = [Lab]
                    Knowledge_Base[annotator_AM]['similar_instances'] = dict()
                    
    else: # EXPLOITATION
        Lab = y_annot_active.loc[index][annotator_AM] 
        for annotator in Knowledge_Base:
            if index in Knowledge_Base[annotator]['index']:
                idx = Knowledge_Base[annotator]['index'].index(index)
                del Knowledge_Base[annotator]['index'][idx]
                del Knowledge_Base[annotator]['label'][idx]
                break
        if output_AM[annotator_AM] > weight_threshold:
            if annotator_AM in Knowledge_Base:
                Knowledge_Base[annotator_AM]['index'].append(index)
                Knowledge_Base[annotator_AM]['label'].append(Lab)
            else:
                Knowledge_Base[annotator_AM] = dict()
                Knowledge_Base[annotator_AM]['index'] = [index]
                Knowledge_Base[annotator_AM]['label'] = [Lab]
                Knowledge_Base[annotator_AM]['similar_instances'] = dict()
        for key in Knowledge_Base:
            if index in Knowledge_Base[key]['similar_instances']:
                Knowledge_Base[key]['similar_instances'][index]['label'] = Lab
                for sim_inst in Knowledge_Base[key]['similar_instances'][index]['index']:
                    collected_data = update_data(sim_inst,Lab,collected_data) #index of all similar instances
                

    x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new = collected_data
    x_new[index] = x_active.loc[index] # Current Instance Feature
    y_true[index] = y_active.loc[index]
    y_annot[index] = y_annot_active.loc[index]
    y_new[index] = Lab

    if torch.sum(inst_annot[row_index]).item() == 1:
        w_new[index] = inst_annot[row_index].numpy()
    else:
        w_new[index] = optimal_weights(y_annot_active.loc[index].to_numpy(),inst_annot[row_index]) # Optimal Weights for this particular instance based on only queried annotators
    
    
    annot_index_opt = torch.argmax(torch.mul(torch.tensor(w_new[index]),inst_annot[row_index])) # Annotator Index of Current Instance as per the highest W_optimal obtained from only queried annotators
    y_new_opt[index] = y_annot_active.loc[index][annot_index_opt.item()] # Label of Current Instance as per the highest W_optimal obtained from only queried annotators
    y_maj = np.array(y_annot_active.loc[index]).copy()
    y_maj = np.delete(y_maj, np.where(inst_annot[row_index] == 0))
    unique, frequency = np.unique(y_maj,return_counts = True)
    unique = unique.tolist()
    frequency = frequency.tolist()
    majority_index = np.argmax(frequency) # Index of the majority Label. Note: All Annotators are considered for the instance
    y_new_majority[index] = unique[majority_index] # Majority Label of the given Instance
    masks_new[index] = inst_annot[row_index].numpy() # Mask representing all annotators which are queried for this Instance
    collected_data = [x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new]

    return index_frame,index,row_index,annotator_AM,collected_data,epoch_iter

def AL_train_cycle_KB(Classifiers,Classifier_y_boot,annotator_selector,Knowledge_Base,TRAIN,BOOT,ACTIVE,VAL,W_optimal,budget, args, device = 'cpu'):

    training_args = {}
    training_args["lr"] = args.active_lr
    training_args["n_epochs"] = args.active_n_epochs
    training_args["log_epochs"] = args.active_log_epochs
    training_args["labeling_type"] = args.labeling_type
    training_args["batch_size"] = args.active_batchsize
    training_args["f1_score"] = args.f1_average
    training_args["Path_results"] = args.Path_results

    scheme = args.w_opt
    RL_flag = args.rl_flag
    ee_ratio = args.ee_ratio
    if args.labeling_type == "max":
        label_choice = 1
    else:
        label_choice = 0
    similarity_threshold = args.similarity_threshold
    weight_threshold = args.weight_threshold
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL = Classifiers

        idx = dict()
        similar_instances = dict()
        x_train, y_train, y_annot_train = TRAIN
        x_boot, y_boot, y_annot_boot = BOOT
        x_active, y_active, y_annot_active = ACTIVE
        x_val, y_val, y_annot_val = VAL
        
        x_new = dict()
        y_new = dict()
        y_annot = dict()
        y_new_opt = dict()
        y_new_majority = dict()
        y_true = dict()
        w_new = dict()
        y_annot_new = dict()
        m = y_annot_active.shape[1]
        masks_new = dict()

        full = 0
        inst_annot = np.zeros_like(y_annot_active)
        inst_annot = torch.from_numpy(inst_annot)

        collected_active_data = []
        loss = []
        c_a = []
        c_f = []

        m = y_annot_active.shape[1]
        input_dim = x_boot.shape[1]
        H_dim = args.hidden_dim
        output_dim = y_annot_boot.shape[1]
        annotator_selector.model = annotator_selector.model.to(device)

        pbar = tqdm.tqdm(desc = 'Progress Bar ', total = budget)
        epoch = 1
        
        while epoch <= budget:

            ACTIVE = [x_active, y_active, y_annot_active]
            collected_data = [x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new]

            if scheme == 1:
                index_frame,index,row_index,annot_index,collected_data,epoch_iter = get_labels_KB(classifier_model_AM,annotator_selector.model,Knowledge_Base,similar_instances,TRAIN.copy(),ACTIVE.copy(),collected_data,inst_annot, device,RL_flag,ee_ratio,label_choice,similarity_threshold,weight_threshold)
            else :
                index_frame,index,row_index,annot_index,collected_data,epoch_iter = get_labels_KB(classifier_model_WO,annotator_selector.model,Knowledge_Base,similar_instances,TRAIN.copy(),ACTIVE.copy(),collected_data,inst_annot, device,RL_flag,ee_ratio,label_choice,similarity_threshold,weight_threshold)
            
            if index in idx :
                idx[index].append(annot_index)
            else :
                idx[index] = [annot_index]

            x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new = collected_data

            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask \
                = append_dataframe(x_new,y_true,y_annot,y_new,y_new_opt,y_new_majority,w_new,masks_new,x_boot,y_annot_boot,Classifier_y_boot,W_optimal,m)
            collected_active_data = [new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask]
            # TRAINING THE CLASSIFIER WITH AVAILABLE DATA FROM BOOT AND OBTAINED FROM ACTIVE LEARNING CYCLE

            if scheme == 1 :
                classifier_model_AM.train(new_active_x,new_active_y)
                y_pred_val = classifier_model_AM.model.predict(x_val)
            else :
                classifier_model_WO.train(new_active_x,new_active_y_opt)
                y_pred_val = classifier_model_WO.model.predict(x_val)

            a,f = eval_model(y_pred_val,y_val)
            c_a.append(a)
            c_f.append(f)

            if torch.sum(inst_annot[row_index]).item() == 1: # CHECK WHETHER MORE THAN ONE ANNOTATOR QUERIED FOR GIVEN INSTANCE BASED ON WHICH ANNOTATOR MODEL WILL BE CALLED
                continue

            # TRAINING THE ANNOTATOR MODEL
            annotator_selector.model = Annotator(input_dim,H_dim,output_dim)
            annotator_selector.model = annotator_selector.model.to(device)
            #loss_list, accuracy, f_1 = annotator_selector.train(training_args,new_active_x,new_active_y_annot,new_active_y_true , new_active_w,new_active_mask,train_phase="train")
            loss_list = annotator_selector.training( new_active_x, new_active_w, new_active_mask,training_args["batch_size"],training_args["n_epochs"], training_args["lr"], device = device)
            
            loss.append(loss_list[-1])
            
            # REMOVE INSTANCES FOR WHICH ALL ANNOTATORS ARE QUERIED
            if (inst_annot[index_frame.index(index)] == 1).all():
                full+=1
                x_active = x_active.drop(labels = index, axis = 0)
                y_active = y_active.drop(labels = index, axis = 0)
                y_annot_active = y_annot_active.drop(labels = index, axis = 0)
                inst_annot = np.delete(inst_annot,row_index,0)
            
            epoch = epoch + epoch_iter
            pbar.update(epoch_iter)
                
            
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
        classifier_model_AM.train(new_active_x,new_active_y)
        classifier_model_WO.train(new_active_x,new_active_y_opt)
        classifier_model_M.train(new_active_x,new_active_y_majority)
        classifier_model_TL.train(new_active_x,new_active_y_true)
        Classifiers = [classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL]  
        return Classifiers,annotator_selector.model,idx,collected_active_data,similar_instances,loss, inst_annot,full, c_a, c_f
    

def AL_train_cycle(Classifiers,Classifier_y_boot,annotator_selector,BOOT,ACTIVE,VAL,W_optimal,budget, args, device = 'cpu'):
    
    training_args = {}
    training_args["lr"] = args.active_lr
    training_args["n_epochs"] = args.active_n_epochs
    training_args["log_epochs"] = args.active_log_epochs
    training_args["labeling_type"] = args.labeling_type
    training_args["batch_size"] = args.active_batchsize
    training_args["f1_score"] = args.f1_average
    training_args["Path_results"] = args.Path_results

    scheme = args.w_opt
    RL_flag = args.rl_flag
    ee_ratio = args.ee_ratio
    if args.labeling_type == "max":
        label_choice = 1
    else:
        label_choice = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL = Classifiers

        idx = dict()
        x_boot, y_boot, y_annot_boot = BOOT
        x_active, y_active, y_annot_active = ACTIVE
        x_val, y_val, y_annot_val = VAL
        
        x_new = dict()
        y_new = dict()
        y_annot = dict()
        y_new_opt = dict()
        y_new_majority = dict()
        y_true = dict()
        w_new = dict()
        y_annot_new = dict()
        m = y_annot_active.shape[1]
        masks_new = dict()

        full = 0
        inst_annot = np.zeros_like(y_annot_active)
        inst_annot = torch.from_numpy(inst_annot)

        collected_active_data = []
        loss = []
        c_a = []
        c_f = []

        m = y_annot_active.shape[1]
        input_dim = x_boot.shape[1]
        H_dim = args.hidden_dim
        output_dim = y_annot_boot.shape[1]
        annotator_selector.model = annotator_selector.model.to(device)

        for epoch in tqdm.tqdm(range(budget)):

            active_data = [x_active,y_active,y_annot_active]
            collected_data = [x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new]

            if scheme == 1:
                index_frame,index,row_index,annot_index,collected_data = get_labels(classifier_model_AM,annotator_selector.model,active_data,collected_data,inst_annot,device,RL_flag,ee_ratio,label_choice)
            else :
                index_frame,index,row_index,annot_index,collected_data = get_labels(classifier_model_WO,annotator_selector.model,active_data,collected_data,inst_annot,device,RL_flag,ee_ratio,label_choice)
            
            if index in idx:
                idx[index].append(annot_index)
            else :
                idx[index] = [annot_index]

            x_new,y_true,y_annot,y_new,w_new,y_new_opt,y_new_majority,masks_new = collected_data

            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask \
                = append_dataframe(x_new,y_true,y_annot,y_new,y_new_opt,y_new_majority,w_new,masks_new,x_boot,y_annot_boot,Classifier_y_boot,W_optimal,m)
            collected_active_data = [new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask]
            # TRAINING THE CLASSIFIER WITH AVAILABLE DATA FROM BOOT AND OBTAINED FROM ACTIVE LEARNING CYCLE

            if scheme == 1 :
                classifier_model_AM.train(new_active_x,new_active_y)
                y_pred_val = classifier_model_AM.model.predict(x_val)
            else :
                classifier_model_WO.train(new_active_x,new_active_y_opt)
                y_pred_val = classifier_model_WO.model.predict(x_val)

            a,f = eval_model(y_pred_val,y_val)
            c_a.append(a)
            c_f.append(f)

            if torch.sum(inst_annot[row_index]).item() == 1: # CHECK WHETHER MORE THAN ONE ANNOTATOR QUERIED FOR GIVEN INSTANCE BASED ON WHICH ANNOTATOR MODEL WILL BE CALLED
                continue

            # TRAINING THE ANNOTATOR MODEL
            annotator_selector.model = Annotator(input_dim,H_dim,output_dim)
            annotator_selector.model = annotator_selector.model.to(device)
            # loss_list, accuracy, f_1 = annotator_selector.train(training_args,new_active_x,new_active_y_annot,new_active_y_true , new_active_w,new_active_mask,train_phase="train")
            loss_list = annotator_selector.training( new_active_x, new_active_w, \
                             new_active_mask,training_args["batch_size"],training_args["n_epochs"], training_args["lr"], device = device)
            
            loss.append(loss_list[-1])
            
            # REMOVE INSTANCES FOR WHICH ALL ANNOTATORS ARE QUERIED
            if (inst_annot[index_frame.index(index)] == 1).all():
                full+=1
                x_active = x_active.drop(labels = index, axis = 0)
                y_active = y_active.drop(labels = index, axis = 0)
                y_annot_active = y_annot_active.drop(labels = index, axis = 0)
                inst_annot = np.delete(inst_annot,row_index,0)
            
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
        classifier_model_AM.train(new_active_x,new_active_y)
        classifier_model_WO.train(new_active_x,new_active_y_opt)
        classifier_model_M.train(new_active_x,new_active_y_majority)
        classifier_model_TL.train(new_active_x,new_active_y_true)
        Classifiers = [classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL]  
        return Classifiers,annotator_model,idx,collected_active_data,loss, inst_annot,full, c_a, c_f
    
def AL_train_majority(Classifiers,Classifier_y_boot,BOOT,ACTIVE,VAL,W_optimal,budget, device = 'cpu'):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL = Classifiers

        idx = dict()
        x_boot, y_boot, y_annot_boot = BOOT
        x_active, y_active, y_annot_active = ACTIVE
        x_val, y_val, y_annot_val = VAL
        x_new = dict()
        y_true = dict()
        y_annot = dict()
        y_new = dict()
        y_new_opt = dict()
        y_new_majority = dict()
        w_new = dict()
        y_annot_new = dict()
        m = y_annot_active.shape[1]
        masks_new = dict()

        full = 0

        inst_annot = np.zeros_like(y_annot_active)
        inst_annot = torch.from_numpy(inst_annot)

        collected_active_data = []
        loss = []
        c_a = []
        c_f = []

        m = y_annot_active.shape[1]
        input_dim = x_boot.shape[1]
        output_dim = y_annot_boot.shape[1]

        for epoch in tqdm.tqdm(range(budget)):

            output_c = classifier_model_M.predict_proba(x_active) #Classifier Output on Active Data
            
            index_frame = list(x_active.index.values) # Inherent Index column of the Dataframe
        
            index = find_index(output_c,index_frame) # Index corresponding to inherent index of the Dataframe
            row_index = index_frame.index(index) # Row Index

            m = y_annot_active.shape[1]
            annot_index = torch.randint(0,m,(1,))
            while inst_annot[row_index][annot_index] == 1:
                annot_index = torch.randint(0,m,(1,))
            
            inst_annot[row_index][annot_index] = 1

            x_new[index] = x_active.loc[index] # Current Instance Feature
            y_true[index] = y_active.loc[index]
            y_annot[index] = y_annot_active.loc[index]
            y_new[index] = y_annot_active.loc[index][annot_index.item()] # Label of Current Instance as per the annotator model
            if torch.sum(inst_annot[row_index]).item() == 1:
                w_new[index] = inst_annot[row_index].numpy()
            else:
                w_new[index] = optimal_weights(y_annot_active.loc[index].to_numpy(),inst_annot[row_index]) # Optimal Weights for this particular instance based on only queried annotators
            annot_index_opt = torch.argmax(torch.mul(torch.tensor(w_new[index]),inst_annot[row_index])) # Annotator Index of Current Instance as per the highest W_optimal obtained from only queried annotators
            y_new_opt[index] = y_annot_active.loc[index][annot_index_opt.item()] # Label of Current Instance as per the highest W_optimal obtained from only queried annotators
            y_maj = np.array(y_annot_active.loc[index]).copy()
            y_maj = np.delete(y_maj, np.where(inst_annot[row_index] == 0))
            unique, frequency = np.unique(y_maj,return_counts = True)
            unique = unique.tolist()
            frequency = frequency.tolist()
            majority_index = np.argmax(frequency) # Index of the majority Label. Note: All Annotators are considered for the instance
            y_new_majority[index] = unique[majority_index] # Majority Label of the given Instance
            masks_new[index] = inst_annot[row_index].numpy() # Mask representing all annotators which are queried for this Instance

            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask \
                = append_dataframe(x_new,y_true,y_annot,y_new,y_new_opt,y_new_majority,w_new,masks_new,x_boot,y_annot_boot,Classifier_y_boot,W_optimal,m)
            collected_active_data = [new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask] 
            # TRAINING THE CLASSIFIER WITH AVAILABLE DATA FROM BOOT AND OBTAINED FROM ACTIVE LEARNING CYCLE

            classifier_model_M.train(new_active_x,new_active_y_majority)
            y_pred_val = classifier_model_M.model.predict(x_val)
            a,f = eval_model(y_pred_val,y_val)
            c_a.append(a)
            c_f.append(f)

            if torch.sum(inst_annot[row_index]).item() == 1: # CHECK WHETHER MORE THAN ONE ANNOTATOR QUERIED FOR GIVEN INSTANCE BASED ON WHICH ANNOTATOR MODEL WILL BE CALLED
                continue
            
            # REMOVE INSTANCES FOR WHICH ALL ANNOTATORS ARE QUERIED
            if (inst_annot[index_frame.index(index)] == 1).all():
                full+=1
                x_active = x_active.drop(labels = index, axis = 0)
                y_active = y_active.drop(labels = index, axis = 0)
                y_annot_active = y_annot_active.drop(labels = index, axis = 0)
                inst_annot = np.delete(inst_annot,row_index,0)

            #print('epoch {}, loss {}'.format(epoch, loss.item()))   
        classifier_model_AM.train(new_active_x,new_active_y)
        classifier_model_WO.train(new_active_x,new_active_y_opt)
        classifier_model_TL.train(new_active_x,new_active_y_true)
        Classifiers = [classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL] 
        return Classifiers,idx,collected_active_data,loss, inst_annot,full, c_a, c_f


def AL_train_MAPAL_instances(Classifiers,Classifier_y_boot,annotator_selector,instances,BOOT,VAL,Mapal_Data,W_optimal,args,  device = 'cpu'):
    
    training_args = {}
    training_args["lr"] = args.active_lr
    training_args["n_epochs"] = args.active_n_epochs
    training_args["log_epochs"] = args.active_log_epochs
    training_args["labeling_type"] = args.labeling_type
    training_args["batch_size"] = args.active_batchsize
    training_args["f1_score"] = args.f1_average
    training_args["Path_results"] = args.Path_results

    if args.labeling_type == "max":
        label_choice = 1
    else:
        label_choice = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        idx = []
        x_boot, y_boot, y_annot_boot = BOOT
        x_val, y_val, y_annot_val = VAL
        new_x_train,new_y_train,new_y_annot_train = Mapal_Data

        classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL = Classifiers
        
        x_new = dict()
        y_new = dict()
        y_annot = dict()
        y_new_opt = dict()
        y_new_majority = dict()
        y_true = dict()
        w_new = dict()
        y_annot_new = dict()
        m = new_y_annot_train.shape[1]
        masks_new = dict()

        full = 0
        inst_annot = np.zeros_like(new_y_annot_train)
        inst_annot = torch.from_numpy(inst_annot)

        collected_active_data = []
        loss = []
        c_a = []
        c_f = []

        input_dim = x_boot.shape[1]
        H_dim = args.hidden_dim
        output_dim = y_annot_boot.shape[1]
        annotator_model = annotator_selector.model.to(device)

        for epoch in tqdm.tqdm(range(len(instances))):
            annotator_model = annotator_selector.model.to(device)
            index = instances[epoch]
            index_frame = list(new_x_train.index.values) # Inherent Index column of the Dataframe
            row_index = index_frame.index(index) # Row Index
            
            output_a = annotator_model(torch.from_numpy(new_x_train.loc[index].to_numpy()).to(device).float()) # Annotator Model weights for a given instance
            output_a_1 = torch.mul(1-inst_annot[row_index],output_a)
            annot_index_new = torch.argmax(output_a_1,dim=0) # Index of Annotator with max weight which is not queried yet for that particular instance

            inst_annot[row_index][annot_index_new] = 1

            output_a_2 = torch.mul(inst_annot[row_index],output_a)

            if label_choice == 1:
                annot_index = torch.argmax(output_a_2,dim=0).item() # Index of Annotator with max weight amongst all queried annotators for that particular instance
            else:
                num_lab = np.unique(new_y_annot_train.loc[index].to_numpy())
                output_a_3 = np.delete(output_a_2.detach().numpy(), np.where(inst_annot[row_index] == 0))
                labels =  np.delete(new_y_annot_train.loc[index].to_numpy(), np.where(inst_annot[row_index] == 0))
                annotator_ids = [i for i in range(new_y_annot_train.shape[1])]
                annotator_ids =  np.delete(annotator_ids, np.where(inst_annot[row_index] == 0))
                max_weight = 0
                annot_index_list = list()
                max_weight_list = list()
                for lab in num_lab:
                    weight = []
                    a_i = []
                    for w,l,id in zip(output_a_3,labels,annotator_ids):
                        if l == lab:
                            weight.append(w)
                            a_i.append(id)
                    if len(weight) == 0:
                        average_weight = 0
                    else:
                        average_weight = sum(weight)/len(weight)
                    if average_weight > max_weight:
                        max_weight = average_weight
                        annot_index_list = a_i
                        max_weight_list = weight
                
                annot_index = annot_index_list[max_weight_list.index(max(max_weight_list))]
            
            x_new[index] = new_x_train.loc[index] # Current Instance Feature 
            y_true[index] = new_y_train.loc[index]
            y_annot[index] = new_y_annot_train.loc[index]
            y_new[index] = new_y_annot_train.loc[index][annot_index] # Label of Current Instance as per the annotator model

            if torch.sum(inst_annot[row_index]).item() == 1:
                w_new[index] = inst_annot[row_index].numpy()
            else:
                w_new[index] = optimal_weights(new_y_annot_train.loc[index].to_numpy(),inst_annot[row_index]) # Optimal Weights for this particular instance based on only queried annotators

            annot_index_opt = torch.argmax(torch.mul(torch.tensor(w_new[index]),inst_annot[row_index])) # Annotator Index of Current Instance as per the highest W_optimal obtained from only queried annotators
            y_new_opt[index] = new_y_annot_train.loc[index][annot_index_opt.item()] # Label of Current Instance as per the highest W_optimal obtained from only queried annotators
            
            y_maj = np.array(new_y_annot_train.loc[index]).copy()
            y_maj = np.delete(y_maj, np.where(inst_annot[row_index] == 0))
            unique, frequency = np.unique(y_maj,return_counts = True)
            unique = unique.tolist()
            frequency = frequency.tolist()
            majority_index = np.argmax(frequency) # Index of the majority Label. Note: All Annotators are considered for the instance
            y_new_majority[index] = unique[majority_index] # Majority Label of the given Instance
            masks_new[index] = inst_annot[row_index].numpy() # Mask representing all annotators which are queried for this Instance


            new_active_x,new_active_y_true,new_active_y_annot,new_active_y,new_active_y_opt,new_active_y_majority,new_active_w,new_active_mask \
                    = append_dataframe(x_new,y_true,y_annot,y_new,y_new_opt,y_new_majority,w_new,masks_new,x_boot,y_annot_boot,Classifier_y_boot,W_optimal,m)
            
            collected_active_data = [new_active_x,new_active_y,new_active_y_opt,new_active_y_majority, new_active_y_true,new_active_y_annot, new_active_mask]
            
            if torch.sum(inst_annot[row_index]).item() == 1: # CHECK WHETHER MORE THAN ONE ANNOTATOR QUERIED FOR GIVEN INSTANCE BASED ON WHICH ANNOTATOR MODEL WILL BE CALLED
                continue

            # TRAINING THE ANNOTATOR MODEL
            annotator_selector.model = Annotator(input_dim,H_dim,output_dim)
            annotator_selector.model = annotator_selector.model.to(device)
            #loss_list, accuracy, f_1 = annotator_selector.train(training_args,new_active_x,new_active_y_annot,new_active_y_true , new_active_w,new_active_mask,train_phase="train")
            loss_list = annotator_selector.training(new_active_x, new_active_w, new_active_mask,training_args["batch_size"],training_args["n_epochs"], training_args["lr"], device = device)
            loss.append(loss_list[-1])
            
            # REMOVE INSTANCES FOR WHICH ALL ANNOTATORS ARE QUERIED
            if (inst_annot[index_frame.index(index)] == 1).all():
                full+=1
                new_x_train = new_x_train.drop(labels = index, axis = 0)
                new_y_train = new_y_train.drop(labels = index, axis = 0)
                new_y_annot_train = new_y_annot_train.drop(labels = index, axis = 0)
                inst_annot = np.delete(inst_annot,row_index,0)

            #print('epoch {}, loss {}'.format(epoch, loss.item()))  
        classifier_model_AM.train(new_active_x,new_active_y)
        classifier_model_WO.train(new_active_x,new_active_y_opt)
        classifier_model_M.train(new_active_x,new_active_y_majority)
        classifier_model_TL.train(new_active_x,new_active_y_true)
        Classifiers = [classifier_model_AM,classifier_model_WO,classifier_model_M, classifier_model_TL]
        return Classifiers, annotator_selector.model,collected_active_data, loss, inst_annot, full, c_a, c_f

