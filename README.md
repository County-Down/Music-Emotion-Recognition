# Music-Emotion-Recognition
This project is based on the research content of dynamic music emotion recognition.<br/>

## 1.Dataset source
PMEmo Dataset: http://next.zju.edu.cn/archive/pmemo/<br/>
DEAM Dataset: https://cvml.unige.ch/databases/DEAM/<br/>

## 2.Operating Environment
### 2.1 Data Preprocessing:
In this part, we adopt `Jupyter Notebook` for data preprocessing and feature dimensionality reduction.<br/>
Python version:
     ```
     Python 3.9.12
     ```<br/>
You can use `install -r requirements.txt` to configure the version of the module you need.
#### Requirements
```
pandas==1.4.2
numpy==1.21.5
matplotlib==3.5.1
scikit-learn==1.0.2
```
The code is stored in the `data_preprocessing` folder.

### 2.2 Neuro Networks:
In this part, we will use MATLAB to build the neural network for the data processed by data_preprocessing.<br/>
The parameters of our device are as follows:<br/>
operating system:`Ubuntu 16.04.7 LTS`<br/>
MATLAB:`2023a`<br/>
GPU:`NVIDIA GeForce RTX 2080 Ti`
The code is stored in the `Neuro_Networks` folder.<br/>

## 3.Usage
### 3.1 Data Preprocessing
You can change the main parameters in this section:<br/>
```
if __name__ == "__main__":
    # file path 
    path_file=Path.cwd()  
    # feature
    dynamic_features=pd.DataFrame(pd.read_csv(str(path_file)+"/Dataset/Pmemo/dynamic_features.csv")) # You can enter the path of the feature dataset.
    # target
    dynamic_annotations=pd.DataFrame(pd.read_csv(str(path_file)+"/Dataset/Pmemo/dynamic_annotations.csv")) # You can enter the path of the target dataset.
    score=0.0015 # Screening random forest importance is greater than or equal to scor.
    Data_Preprocessing(dynamic_features,dynamic_annotations,'Arousal(mean)',score,'Pmemo').train_model() # Start processing the data
    Data_Preprocessing(dynamic_features,dynamic_annotations,'Valence(mean)',score,'Pmemo').train_model() # Start processing the data
```
We will use random forest to evaluate the importance of features for dimensionality reduction:<br/>
![Pmemo-Arousal(mean)](https://github.com/user-attachments/assets/a92857d9-db31-4f03-86fb-4898e48e53a1)<br/>
After running the code, two folders should appear：<br/>
![{078E666A-DA7A-44F9-94AF-0596DC0ADF06}](https://github.com/user-attachments/assets/0345ef57-de7a-4991-bbf0-4fab5ffa85be)<br/>
frameTime_count:Store the length of time for each song in the dataset.<br/>
randomforest:Store reduced data sets.<br/>

### 3.2 Neuro Networks:
  `Main. m` is our main function that we run the code by running it.<br/>
The `target_data` refers to the `emotion values` of the dataset<br/>
```
target_data=table2array(readtable(['./',dataset_name,'/Dataset/',dataset_name,'/dynamic_annotations.csv']));
```
The `frameTime_count_all` is the time node data for each song of the dataset output from 3.1:
```
frameTime_count_all=xlsread(['./',dataset_name,'/frameTime_count/frameTime_count_all.xlsx']);
```
The `dynamic_features_analyze` is the preprocessed dataset output from 3.1：
```
dynamic_features_analyze=xlsread(['./',dataset_name,'/randomforest/Valence(mean)/dynamic_features_analyze.xlsx']);
```
  We invoke the time series modeling code with the following statement. In this step, We do time series modeling, data normalization, and dataset splitting, which is all encapsulated in `dataset_crossvalind.m`.<br/>
  ```
    [Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt]=dataset_crossvalind(save_folder,target,frameTime_count_all,dynamic_features_analyze,k,numFeatures);
  ```

 In Main.m, the `neural network` for this study is called as follows:<br/>
 ```
    DSelfAttention(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
      dataset_name,target_name,save_folder, ...
      numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
      validationFrequency,kernel_size1,kernel_size2,channel,k,block_num);
 ```

You can modify the neural network by clicking DSelfAttention.m.<br/>
![{96F63662-6FBE-42BF-8AA0-6B0B612D91D2}](https://github.com/user-attachments/assets/708cc39e-a8b0-485b-93ee-e065558b8f1e)

In this section, we conduct `ablation experiments`:
```
% % ablation study
% % cnn
CNN(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% bilstm
BiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm(No Residual block)
CNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm-Scale 1(No Residual block)
TCNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm-Scale 2(No Residual block)
SCNNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k);
% cnn-bilstm-Residual-block
DCNBiLSTM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,block_num);

```

  In this section, we test other types of `attention mechanisms`：
  ```
% Attention Mechanism
% SE+SPATIAL Attention Mechanism
DSESAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);
% CBAM Attention Mechanism
DCBAM(Train_xNorm,Test_xNorm,Validate_xNorm,Train_yNorm,Test_yNorm,Validate_yNorm,yopt, ...
    dataset_name,target_name,save_folder, ...
    numFeatures,numResponses,miniBatchSize,numHiddenUnits,maxEpochs,learning_rate, ...
    validationFrequency,kernel_size1,kernel_size2,channel,k,num_block);
```
