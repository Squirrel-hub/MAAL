import torch
import warnings
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

def create_knowledge_base(Knowledge_Base,annotator_model,BOOT):

    boot_x, boot_y, boot_annotator_labels = BOOT

    index_frame = boot_x.index.values.tolist()
    for index,data in zip(index_frame,boot_x.to_numpy()):
        data = torch.tensor(data,dtype=torch.float32)
        output = annotator_model(data)
        annot_index = torch.argmax(output,dim=0).item() ## CHECK WITH WEIGHT THRESHOLD ,Example 0.8 ETC
        if annot_index in Knowledge_Base:
            Knowledge_Base[annot_index]['index'].append(index)
            Knowledge_Base[annot_index]['label'].append(boot_annotator_labels.loc[index][annot_index])
        else:
            Knowledge_Base[annot_index] = dict()
            Knowledge_Base[annot_index]['index'] = [index]
            Knowledge_Base[annot_index]['label'] = [boot_annotator_labels.loc[index][annot_index]]
            Knowledge_Base[annot_index]['similar_instances'] = dict()

def Knowledge_Base_similarity(data,Knowledge_Base,x_train):

    annotator = 0
    max_similarity = 0
    instance = 0
    label = 0
    for key in Knowledge_Base:
        for idx,lab in zip(Knowledge_Base[key]['index'],Knowledge_Base[key]['label']) :
            x = x_train.loc[idx].to_numpy()
            x = torch.tensor(x,dtype=torch.float32)
            similarity = cosine_similarity(torch.unsqueeze(data,0),torch.unsqueeze(x,0))[0][0]
            if similarity > max_similarity:
                annotator =  key # Return idx to get label
                max_similarity = similarity
                label = lab
                instance = idx

    return annotator,max_similarity,instance, label
    

def count_instances(txt_path,Knowledge_Base,active_x):
    # Append-adds at last
    file = open(txt_path,"a")#append mode
    file.write('\n\n\nTotal number of Active Learning Instances : '+ str(active_x.shape[0])+"\n")
    file.close()



    count = 0
    for annot_base in Knowledge_Base.keys():
        for key in Knowledge_Base[annot_base]['similar_instances'].keys():
            count += len(Knowledge_Base[annot_base]['similar_instances'][key]['index'])

    # Append-adds at last
    file = open(txt_path,"a")#append mode
    file.write('\n\n\nExtra instances added due to knowledge base similarity : '+ str(count) + "\n")
    file.close()
    

def Knowledge_Base_Metrics(Knowledge_Base,new_active_y,Parent_dir,txt_path,average=None):

    data_KB = [{} for i in range(len(Knowledge_Base.keys()))]

    similar_instances_keys_list = list()

    for annotator in Knowledge_Base.keys():
        similar_instances_keys_list = similar_instances_keys_list + list(Knowledge_Base[annotator]['similar_instances'].keys())

    data_Sim = [{} for i in range(len(similar_instances_keys_list))]

    # df_c_KB = pd.DataFrame(data_c_KB,index = ['Annotator Model', 'W Optimal', 'Majority','True Labels'])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
     
        i = 0
        for key in Knowledge_Base:
            i = i + 1
            ground_truth = []
            for idx in Knowledge_Base[key]['index']:
                ground_truth.append(new_active_y.loc[idx])
            data_KB[i-1]['No. of Instances'] = len(ground_truth)
            data_KB[i-1]['Accuracy'] = accuracy_score(Knowledge_Base[key]['label'],ground_truth)
            data_KB[i-1]['f1-score'] = f1_score(Knowledge_Base[key]['label'],ground_truth,average=average)
        
        df_KB = pd.DataFrame(data_KB, index = Knowledge_Base.keys())
            

      
        i = 0
        for annotator in Knowledge_Base.keys():
            for inst in Knowledge_Base[annotator]['similar_instances']:
                i = i + 1
                ground_truth = []
                for idx in Knowledge_Base[annotator]['similar_instances'][inst]['index']:
                    ground_truth.append(new_active_y.loc[idx])
                labels = [Knowledge_Base[annotator]['similar_instances'][inst]['label'] for j in range(len(ground_truth))]
                
                data_Sim[i-1]['No. of similar Instances'] = len(ground_truth)
                data_Sim[i-1]['Similar Instances'] = Knowledge_Base[annotator]['similar_instances'][inst]['index']
                data_Sim[i-1]['Similar Instances Shared Label'] = Knowledge_Base[annotator]['similar_instances'][inst]['label']
                data_Sim[i-1]['Accuracy'] = accuracy_score(labels,ground_truth)
                data_Sim[i-1]['f1-score'] = f1_score(labels,ground_truth,average=average)
                data_Sim[i-1]['Annotator'] = annotator

        df_Sim = pd.DataFrame(data_Sim, index = similar_instances_keys_list)

        dir1 = "Knowledge_Base_Metrics.csv"
        dir2 = "Similar_Instances_Metrics.csv"

        path1 = os.path.join(Parent_dir,dir1)
        path2 = os.path.join(Parent_dir,dir2)

        df_KB.to_csv(path1)
        df_Sim.to_csv(path2)

        #export DataFrame to text file
        with open(txt_path, 'a') as f:
            df_string =df_KB.to_string(header=True, index=True)
            f.write("\n\n\nKNOWLEDGE BASE METRICS\n\n")
            f.write(df_string)

        with open(txt_path, 'a') as f:
            df_string =df_Sim.to_string(header=True, index=True)
            f.write("\n\n\nSIMILAR INSTANCES METRICS\n\n")
            f.write(df_string)