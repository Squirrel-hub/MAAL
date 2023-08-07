import numpy as np
import random
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,f1_score
from scipy.stats import entropy

# def find_entropy(output):
#   base = output.shape[1]
#   Entropy = list()
#   for i in output:
#     ent = 0
#     for j in i:
#       ent -= j*np.log(j)
#     # ent = entropy(i,base = base)
#     Entropy.append(ent)
#   return Entropy

def find_entropy(output):
  base = output.shape[1]
  Entropy = entropy(output,base = base, axis = 1)
  return Entropy

def find_index(output,index_frame):
  Entropy = find_entropy(output)
  max_index = np.argmax(Entropy,axis=0)
  return index_frame[max_index]

def find_entropy_RL(output,inst_annot,choice):
  base = output.shape[1]
  m = inst_annot.shape[1]
  Entropy = list()
  for iter,i in enumerate(output):
    if choice == 0 :
      if torch.sum(inst_annot[iter]).item() > 0.5: # Explore
        Entropy.append(0)
      else :
        # Entropy.append(-i[0]*np.log(i[0]) - i[1]*np.log(i[1]))
        Entropy.append(entropy(i,base=base))
    elif choice == 1 :
      if torch.sum(inst_annot[iter]).item() == 0:  # Exploit
        Entropy.append(0)
      else :
        # Entropy.append(-i[0]*np.log(i[0]) - i[1]*np.log(i[1]))
        Entropy.append(entropy(i,base=base))
        
  return Entropy


def find_index_RL(output,index_frame,inst_annot,choice):
  
  Entropy = find_entropy_RL(output,inst_annot,choice)
  max_index = np.argmax(Entropy,axis=0)
  return index_frame[max_index]


def Average(lst):
    return sum(lst) / len(lst)
