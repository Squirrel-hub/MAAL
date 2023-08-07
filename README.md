# Towards More Efficient Multi-Annotator Active Learning: Annotator Expertise Inference and Beyond

## Background
Active machine learning refers to a subfield of machine learning where an algorithm or model interacts with a human or an intelligent agent to actively select or acquire new data samples for training. Unlike traditional machine learning approaches that rely on pre-labelled datasets, active learning enables the model to query the most informative or uncertain instances from a pool of unlabelled data.

In active learning, the model initially learns from a small set of labelled examples. It then uses various query strategies to identify the most valuable samples for labeling. These query strategies often leverage uncertainty estimation or information gain measures to prioritize samples that are likely to improve the model’s performance the most. The selected samples are then presented to human annotators or other labeling mechanisms to obtain their ground truth labels.

After incorporating the newly labelled samples into the training set, the model iteratively repeats the process of selecting informative samples, obtaining their labels, and updating the model. This interactive feedback loop allows the model to progressively improve its accuracy and generalization performance with minimal human labeling effort.

## Problem Statement 
```Active Learning helps a machine learning model to preferentially obtain labels from available annotators and choose data samples which can be impactful to train classification models. In this work we question the assumption of considering a single omniscient and omnipresent annotator in traditional active learning and hence think about solving a more practical variant of the problem considering multiple annotators which could be individually error-prone or have varying expertise.```

## General Overview
We attempt to improve upon the existing operational approaches for the given problem with our novel scheme which we call as a Knowledge Base driven Multiple Annotator Active Learning.
Boot Phase comprises of randomly choosing small number of samples with all their annotations labeled. This is part of our budget consumption.
During Active Phase we use Entropy as a measure of our sample/Query utility measure along with a 2-layer neural network as our annotator performance model. We use Logistic Regression as our classifier model and provide various other classical alternatives.

Samples obtain their label from either the Knowledge Base (KB) of various annotators or indirectly from the annotator performance model. Labels are obtained from KB provided it exceeds a similarity threshold and the KB is enriched provided the confidence of an annotator for a sample is greater than a weight threshold.

Classification Model is trained on Labels obtained from either the Knowledge Base or that indicated by the Annotator Performance Model
To obtain labels from Knowledge Base we iterate over every instance of each Annotator Knowledge Base.

We calculate the cosine similarity metric between each knowledge base instance and the instance chosen as per the query utility measure.
Instances in Knowledge Base with highest similarity score share their label with the current chosen instance and hence do not consume any budget by asking any annotator for a fresh labelling request.
Annotator Performance Model is trained with optimal weights calculated using Linear Programming. Suppose for a sample we are given a set of weights corresponding those annotators which have been queried and are available as wi for ith annotator. Therefore, we obtain optimal weights to train our annotator performance model for each sample using the following two equations for Linear Programming. n stands for the number of queried annotators for a given sample

![Alt text](<Picture 3.png>)


### Installation
Install the required libraries
    
    pip install -r requirements.txt

## Our Implementation

## Project Structure
- datasets: contains .csv-files of data sets being not available at [OpenML](https://www.openml.org/home)
- data_preprocessing : consists of data in processed form for active learning
- plots: directory where the visualizations of MaPAL will be saved
- results/Knowledge_Base: path where all results will be stored such as classifier metrics, Annotator performance model metrics, Knowledge Base accuracy and various necessary plots for visualization
- AL-design: Consists of the implemented active learning training cycle
- classifier: Comprises of Classifier Models, Classifier Evaluation and other necessary utility functionalities.
- annotator : Comprises of Annotator Performance Models, Annotator Performance Evaluation and other necessary utility functionalities.
- knowledge_base : It holds the various utility functions with regard to estimating the diversity and correctness of our created knowledge base
- annotator : Comprises of utlity functions and various Model definitions of Annotator Performance Models
- stats : This contains various helper functions to visualize our data statistics
- results : It consists of the results obtained from our novel training scheme as well as the results generated from various baselines for all concerned datasets
- Lpp : Consists of Linear Programming helper functions
- src : Directory which comprises of the Baseline Codes
- utils : Comprises of Utility functions
- Loaders : Comprises of Custom DataLoaders

## How to execute an experiment?
Run only the main.py file with all the expected arguments to reproduce all results.

#### Active Learning

    python main.py [OPTIONS]

    Options:
        --dataset                                                            Name of Dataset to be used
        --data_folder_path                                                   Name of Directory or Folder where all datasets are stored
        --budget_frac                                                        Fraction of Trainind data to be consumed as budget
        --boot_size                                                          Fraction of Training Data to be used for Warm-Up Period
        --test_size                                                          Fraction of Total Data to be used for testing
        --boot_lr                                                            Learning rate of Annotator Model during Boot Phase
        --boot_n_epochs                                                      Number of training epochs of Annotator Model during Boot Phase
        --boot_log_epochs                                                    Interval of Epochs for saving running statistics during Boot Phase
        --boot_batchsize                                                     Batchsize of Annotator Model Training during Boot Phase
        --active_lr                                                          Learning rate of Annotator Model during Active Learning Phase
        --active_n_epochs                                                    Number of training epochs of Annotator Model during Active Learning Phase
        --active_log_epochs                                                  Interval of Epochs for saving running statistics during Active Learning Phase
        --active_batchsize                                                   Batchsize of Annotator Model Training during Active Learning Phase
        --hidden_dim                                                         Hidden Dimension of Annotator Model Architecture
        --instance_strategy                                                  Strategy of choosing an Instance
        --method                                                             To denote the source of labels to train classifier model
        --LR_max_iter                                                        Maximum iterations to train Logistic Regression Classifier
        --classifier_name                                                    To denote the type of classifier used
        --f1_average                                                         Average between ["micro","macro","weighted"]
        --log_dir                                                            Logging Directory
        --seed                                                               Execution seed
        --labeling_type                                                      To obtain maximum argument label or average weighted label of annotator model
        --exp_name                                                           Custom name to our executed experiments
        --use_Knowledge_Base                                                 Option to use Knowledge Base
        --similarity_threshold                                               Similarity threshold to decide whether labels are obtained from Knowledge Base
        --weight_threshold                                                   Threshold to identify recognoze whether an Annotator is confident       
        --data_source                                                        To decide whether to use same data as baseline
        --w_opt                                                              To decide whether to obtain label from calculated annotator weights
        --rl_flag                                                            To choose whether to explore and exploit as per a fixed ratio
        --ee_ratio                                                           To manually assign an exploration and exploitation ratio
        --use_Knowledge_Base                                                 Option to use Knowledge Base
        --similarity_threshold                                               Similarity threshold to decide whether labels are obtained from Knowledge Base
        --weight_threshold                                                   Threshold to identify recognoze whether an Annotator is confident  
        --Path_results                                                       Directory to store results
        --Path_mapal_data                                                    Directory to store MAPAL(Baseline) generated results
        --exp_txt_path                                                       Path to results text file 



Required Packages need to be installed from requirements.txt

## Results on our Novel Algorithm in comparison to Baseline (MAPAL)

![Alt text](<Picture 1.png>) 

The above bar plot displays the accuracies of individual classifier trained on different algorithms.
As we observe our algorithm based on Knowledge Base outperforms other related baselines.

![Alt text](<Picture 2.png>)

We can inspect the high accuracy of the labels present in the Knowledge Base for each Annotator throughout various datasets. This empirically validates the correctness of our approach to obtain labels from Knowledge Base

## BASELINE Implementation (MAPAL - ICPR 2021 )

## Project Structure
- data: contains .csv-files of data sets being not available at [OpenML](https://www.openml.org/home)
- plots: directory where the visualizations of MaPAL will be saved
- results: path where all results will be stored including csvs, learning curves, and ranking statistics
- src: Python package consisting of several sub-packages
    - base: implementation of DataSet and QueryStrategy class
    - classifier: implementation of Similarity based Classifier (SbC) being an advancement of the Parzen Window Classifier (PWC) 
    - evaluation_scripts: scripts for experimental setup
    - notebooks: jupyter notebooks for the investigation of MaPAL, simulation of annotators, and the illustration of results
    - query_strategies: implementation of all query/AL strategies
    - utils: helper functions

## How to execute an experiment?
Due to the large number of experiments, we executed the experiments on a computer cluster. This way, we were able to execute 100 experiments simultaneously. 

Without such a computer cluster, it will  probably take several days to reproduce all results of the article. Nevertheless, one can execute the 
experiments on a local personal computer by following the upcoming steps.

1. Setup Python environment:
```bash
projectpath$ sudo apt-get install python3-pip
projectpath$ pip3 install virtualenv
projectpath$ virtualenv mapal
projectpath$ source mapal/bin/activate
projectpath$ pip3 install -r requirements.txt
```
2. Simulate annotators: Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/notebooks/simulate_annotators.ipynb`. This must be the first step before executing any experiment.
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```
2. Get information about the available hyperparameters (argparse) for the experiments.
```bash
projectpath$ source mapal/bin/activate
projectpath$ export PYTHONPATH="${PYTHONPATH}":$(pwd)/src
projectpath$ python3 src/evaluation_scripts/experimental_setup.py -h
```
3. Example experiment: To test MaPAL with M_max=2 and beta_0=0.0001 on the dataset iris with annotators having instance-dependent performance values and with
    - a budget of 40% of all available annotations, 
    - a test ratio of 40%, 
    - and using the seed 1,
    
we have to execute the following commands:
```bash
projectpath$ source mapal/bin/activate
projectpath$ export PYTHONPATH="${PYTHONPATH}":$(pwd)/src
projectpath$ python3 src/evaluation_scripts/experimental_setup.py \
  --query_strategy mapal-1-0.0001-2-1-entropy \
  --data_set iris-simulated-x \
  --results_path results/simulated-x/csvs \
  --test_ratio 0.4 \
  --budget 0.4 \
  --seed 1
```
For this example, the results are saved in the directory `projectpath/results/simulated-x/csvs/` as a .csv-file.

The names of the possible data sets are given in the following files:
- `projectpath/data/data-set-names-real-world.csv`: contains the names of the data sets with real-world annotators (the data set grid is not available because it contains confidential data),
- `projectpath/data/data-set-names-simulated-o.csv`: contains the names of the data sets with simulated annotators having uniform performance values,
- `projectpath/data/data-set-names-simulated-y.csv`: contains the names of the data sets with simulated annotators having class-dependent performance values,
- `projectpath/data/data-set-names-simulated-x.csv`: contains the names of the data sets with simulated annotators having instance-dependent performance values.

To create the ranking statistics, there must be at least one run for each strategy on a data set.  The different AL strategies that can be used as `--query_strategy` argument are given in the following:
- MaPAL: `mapal-1-0.0001-2-1-entropy`,
- IEThresh: `ie-thresh`,
- IEAdjCost: `ie-adj-cost`,
- CEAL: `ceal`,
- ALIO: `alio`,
- Proactive: `proactive`,
- Random: `random`.

To conduct the experiments data sets with real-world annotators in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_real-world-local.sh 5
```
The argument `5` is an example and gives the maximum number of runs that can be executed in parallel. You can change this number.

To conduct the experiments data sets with simulated annotators having uniform performances values in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_simulated-o-local.sh 5
```

To conduct the experiments data sets with simulated annotators having class-dependent performances values in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_simulated-y-local.sh 5
```

To conduct the experiments data sets with simulated annotators having instance-dependent performances values in accordance to the article, execute the following command:
```bash
projectpath$ bash src/evaluation_scripts/evaluate_simulated-x-local.sh 5
```

## How to illustrate the experimental results?
Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/notebooks/experimental_results.ipynb`.
Remark: The ranking plots can only be created when we have for each dataset and each strategy the same number of 
executed experiments. 
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```

## How to reproduce the annotation performance and instance utility plots?
Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/notebooks/visualization.ipynb`.
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```

## How to reproduce study on hyperparameters?
Run experiments on toy data set by executing the following command.
```bash
projectpath$ bash src/evaluation_scripts/evaluate_toy-data.sh 5
```
The argument `5` is an example and gives the maximum number of runs that can be executed in parallel. You can change this number.

Start jupyter-notebook and run the jupyter-notebook file `projectpath/src/notebooks/hyperparameters.ipynb`.
```bash
projectpath$ source mapal/bin/activate
projectpath$ jupyter-notebook
```

## Referrences

1.M. Herde, D. Kottke, D. Huseljic and B. Sick, "Multi-Annotator Probabilistic Active Learning," 2020 25th International Conference on Pattern Recognition (ICPR), Milan, Italy, 2021, pp. 10281-10288, doi: 10.1109/ICPR48806.2021.9412298.
2.Goh, Hui Wen, and Jonas Mueller. "ActiveLab: Active Learning with Re-Labeling by Multiple Annotators." arXiv preprint arXiv:2301.11856 (2023).
