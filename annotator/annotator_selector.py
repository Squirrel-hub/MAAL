import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from Lpp.utils import weights_optimal
from Loaders.dataset_loaders import Data, Data_AL
import matplotlib.pyplot as plt
import random

from annotator.Model import Annotator as AnnotatorModel
from annotator.utils import get_weighted_labels, get_majority_labels, get_max_labels

class AnnotatorSelector:
    def __init__(self, n_features, n_annotators,hidden_dim, device, log_dir="logs",seed = 1, report_to_tensorboard=False):
        self.n_features = n_features
        self.n_annotators = n_annotators
        self.hidden_dim = hidden_dim
        self.device = device
        self.seed = seed
        self.model = self.initialize_model()

        if report_to_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def initialize_model(self):
        self.set_seed(self.seed)
        return AnnotatorModel(n_features=self.n_features, hidden_dim=self.hidden_dim,n_annotators=self.n_annotators).to(self.device)
    

    def set_seed(self,seed: int = 1) -> None:  ## Set SEEDS FUNCTION
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set as {seed}")

    def write_to_tensorboard(self, epoch, data, type="train"):
        for key in data:
            self.writer.add_scalar(f"annotator/{key}/{type}", data[key], epoch)

    # Annotator weights where majority agreeing annotators have weight 1 and rest 0
    def get_annotator_majority_weights(self, annotator_labels, annotator_mask):
        annotator_weights = []
        majority_labels = get_majority_labels(annotator_labels, annotator_mask)
        for i in range(len(annotator_labels)):
            annotator_weights.append((annotator_labels[i] == majority_labels[i]).astype("int"))
            # print(annotator_labels[i])
            # print(annotator_mask[i])
            # print(annotator_weights[-1])
            # input()
        return np.array(annotator_weights)
    


    def coeff_annot(self,arr,idx,inst_annot):
        ones = 0
        m = len(inst_annot)
        
        coefficients = [1 for i in range(m)]

        for i in range(len(arr)):
            if inst_annot[idx] == 0 :
                # print('inst_annot[i] : ',inst_annot[i])
                coefficients[idx] = 0
                coefficients[i] = 0
            elif inst_annot[i] == 0:
                coefficients[i] = 0
            else :
                if i == idx:
                    coefficients[idx] = sum(inst_annot)-1
                    continue
                if arr[i] == arr[idx] :
                    coefficients[i] = -1
                else : 
                    coefficients[i] = 1
                    ones = ones + 1
        # print('coefficients : ',coefficients)
        # print('ones : ',ones)
        return coefficients,ones

    def weights_optimal(self,y_annot,inst_annot):
        W_optimal = []
        A_ub_list = []
        b_ub_list = []
        masks = []
        #print(type(y_annot_boot))

        m = inst_annot.shape[1]
        for i in range(m):
            arr1 = [0 for i in range(m)]
            arr1[i] = 1
            A_ub_list.append(arr1)
            arr2 = [0 for i in range(m)]
            arr2[i] = -1
            A_ub_list.append(arr2)
            b_ub_list.append(1)
            b_ub_list.append(0)


        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        c    = np.array([-1 for i in range(m)])


        for i,j in zip(y_annot,inst_annot):
            A_eq_list = []
            b_eq_list = []
            for idx in range(m):
                coefficients,ones = self.coeff_annot(i,idx,j)
                A_eq_list.append(coefficients)
                b_eq_list.append(ones)
            
            A_eq = np.array(A_eq_list)
            b_eq = np.array(b_eq_list)

            # Solve linear programming problem
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
            W_optimal.append(res.x)
            masks.append(j)
            
        return W_optimal
    
    def get_annotator_model_weights(self, instances):
        if not torch.is_tensor(instances):
            instances = torch.tensor(instances)

        if len(instances.shape) == 1: # If only a single instance is passed,  make it a 2d tensor
            instances = instances.unsqueeze(0)

        with torch.no_grad():
            annotator_weights = self.model(instances.float().to(self.device)).cpu().squeeze()

        return annotator_weights
    
    
    def warmup(self,x_boot, y_annot_boot,mask=None,batch_size = 4,n_epochs=1000,learning_rate=0.001, device='cpu'):
        self.model = self.model.to(device)
        if mask.any() == None:
            mask = np.ones_like(y_annot_boot)
        W_optimal = weights_optimal(y_annot_boot.to_numpy(),mask)
        W_optimal = pd.DataFrame(W_optimal,index = list(x_boot.index.values))
        mask = pd.DataFrame(mask,index = list(x_boot.index.values))
        loss_list = self.training(x_boot,W_optimal,mask,batch_size,n_epochs,learning_rate,device)
    
        plt.figure()
        plt.plot(loss_list)
        plt.title('Annotator Model Warmup Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        return W_optimal,loss_list

        
    def training(self, new_active_x, new_active_w, new_active_mask,batch_size = 4,n_epochs=1000, learning_rate = 0.001, device = 'cpu'):
        
        optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion_annotator = nn.MSELoss()
        data_set=Data_AL(new_active_x.to_numpy(),new_active_w.to_numpy(),new_active_mask.to_numpy())
        trainloader=DataLoader(dataset=data_set,batch_size=batch_size)
        loss_list = []
        for epoch in range(n_epochs):
            epoch_loss = 0
            for x, y, mask in trainloader:
                x = torch.tensor(x,dtype=torch.float32).to(device)
                #clear gradient 
                optimizer.zero_grad()
                #make a prediction 
                z = self.model(x)
                y = y.squeeze(1)
                y = y.type(torch.FloatTensor).to(device)
                mask = mask.to(device)
                x1 = torch.mul(mask,z)
                x2 = torch.mul(mask,y)
                loss = criterion_annotator(x1,x2)
                    
                # calculate gradients of parameters 
                loss.backward()
                epoch_loss += loss.to('cpu').data
                # update parameters 
                optimizer.step()
                
            loss_list.append(epoch_loss/batch_size)
            #print('epoch {}, loss {}'.format(epoch, loss_list[-1]))  
        
        # plt.figure()
        # plt.plot(loss_list)
        # plt.title('Annotator Model Training')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        
        return loss_list


    