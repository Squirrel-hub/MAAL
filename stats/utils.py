import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import warnings
import os

def diverse_samples(inst_annot,full,Path):
    
    unique, frequency = np.unique(inst_annot.sum(1),return_counts=True)
    unique = unique.tolist()
    frequency = frequency.tolist()
    num = [int(i) for i in unique]
    num.append(inst_annot.shape[1])
    frequency.append(full)
    print(num,frequency)

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
