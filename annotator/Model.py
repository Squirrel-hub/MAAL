import torch
import torch.nn as nn
    

# class Annotator(nn.Module):
#     def __init__(self,n_features,n_annotators, hidden_dim=32):
#         super(Annotator,self).__init__()
#         self.linear1 = nn.Linear(n_features,hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim,n_annotators)
#         self.linear3 = nn.Linear(hidden_dim,hidden_dim)
#         self.relu    = nn.ReLU()
        
#     def forward(self,x):
#         x = self.linear1(x)
#         x = self.relu(x)
#         # x = self.linear3(x)
#         # x = self.relu(x)
#         x = self.linear2(x)
#         x=torch.sigmoid(x)  
#         return x

class Annotator(nn.Module):
    def __init__(self,n_features,hidden_dim,n_annotators):
        super(Annotator,self).__init__()
        self.linear1 = nn.Linear(n_features,hidden_dim)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim,n_annotators)
        self.linear3 = nn.Linear(hidden_dim,hidden_dim)
        
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.linear3(x)
        # x = self.relu(x)
        x = self.linear2(x)
        x=torch.sigmoid(x)  
        return x