# Welcome to RiFNet

Imaging techniques are widely used for medical diagnostics. In some cases, a lack of medical practitioners who can manually analyze the images can lead to a bottleneck. Consequently, we develop a custom-made convolutional neural network (RiFNet = Rib Fracture Network) that can detect rib fractures in postmortem computed tomography. In a retrospective cohort study, we retrieved PMCT data from 195 postmortem cases with rib fractures from July 2017 up to April 2018 from our database. The computed tomography data were prepared using a plugin in the commercial imaging software Syngo.via whereby the rib cage was unfolded on a single-in-plane image reformation. Out of the 195 cases, a total of 585 images were extracted and divided into two groups labeled *with* and *without* fractures. These two groups were subsequently divided into training, validation, and test datasets to assess the performance of RiFNet. In addition, we explored the possibility of applying transfer learning techniques on our dataset by choosing two independent noncommercial off-the-shelf convolutional neural network architectures (ResNet50 V2 and Inception V3) and compared the performances of those two with RiFNet. When using pretrained convolutional neural networks, we achieved an F_1 score of 0.64 with Inception V3 and an F_1 score of 0.61 with ResNet50 V2. We obtained an average F_1 score of 0.91±0.04 with RiFNet. RiFNet is efficient in detecting rib fractures on postmortem computed tomography. Transfer learning techniques are not necessarily well adapted to make classifications in postmortem computed tomography.


## Installation

The implemented code has been tested on the following operating system:
- Ubuntu 18.04.3 / MacOS Catalina 10.15.1

Required packages:
- tensorflow 1.15.0
- keras 2.2.4
- scikit-learn 0.22.1
- matplotlib 3.1.1

**note:** *maybe additional missing packages are required!*

We highly recommend to install all the requirements in a new conda environment! If this is not familiar to you, refer to https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

## Preprocessing

**Input Files**

The rib fracture images need to be extracted from volumetric CT data with the Syngo.via rib unfolding tool CT Bone Reading (Siemens Healthineers GmbH, Erlangen, Germany) with bone window settings *center 450 HU* and *width 1500 HU* for the cross-sectional images. The window  settings for the unfolding tool have to be 1000 HU for the center and 2500 for the width. The resulting images are 2D slices (x,y) of the whole rib cage in RGB (see example image below). 

Example image:

<img src="https://github.com/victorjonathanibanez/test/blob/main/fracture.1.jpg" alt="alt text" width="700" height="500">

## Use pre-trained RiFNet 

If you wish to simply use RiFNet to predict on your PMCT images, we provide you with a pre-trained model *RiFNet.h5*. You only have to create a folder named *PMCT_images*, located in the same directory as the *prediction.py* code. The code will automatically rescale your 2D-images (**note:** your images have to be already pre-processed with the Syngo.via tool, refer to section **Preprocessing** - *Input Files*!) to the right size and subsequently create a folder *PMCT_predictions* to save images classified as *fractures* in one folder and images classified as *no fractures* in another folder. To start the prediction, change your directory to the location of the main folder in your terminal (command: 'cd *your main directory*') and subsequently run the command 'python3 prediction.py'.

## Train RiFNet on your own data

**Step 1 - Preprocessing**
After preprocessing steps (see section **Preprocessing**), the images need to be labeled by an expert into two classes: *rib fractures* and *no rib fractures*. The network reads images directly from a *raw_data* folder containing two folders with two class names (fracture / no_fracture), located in the main folder. The images in the two folders should be named and numbered accordingly (i.e. *fracture_1.jpg*, *no_fracture_1.jpg*).

**Step 2 - Training and Testing**

Training:
When your folder structure *raw_data* is ready with the appropriate images in the sub-folder, you can change the directory to the main folder in your terminal and then run the command 'python3 training.py'. The code will take your raw data and randomly splits it 5 times into training and testing data. In each iteration the training data will be split 5-fold into training and validation data to assess the performance (see figure below for an overview on the cross validation procedure) of the model and creates folders and files as follows: 


| Folders | File/s | Description |
| ------- | ------ | ------ |
| models_CV_m | RiFNet_CV_m_run_n.h5, accuracy_CV_m_n.png, loss_CV_m_n.png | Creates a folder for each training iteration with the saved models, accuracy and loss plots. | 
| test_CV_m/fracture, no_fracture | *fracture_1.jpg...*, *no_fracture_1.jpg...*,  | Creates a test data folder for each training iteration with two subfolders each. |

-> *m* denotes the number of training sessions and *n* the folds for each training iteration.


Testing:
When the training procedure is finished, run the command 'python3 testing.py' in your terminal. This will validate all the models on the test compartiments. The code stores a text file under the main folder with prediction accuracy values, precision values, recall values and F1-score values for each of the 25 predictions as well as mean values and standard deviations of the entire process.


| Folders | File/s | Description |
| ------- | ------ | ------ |
| models_CV_m | CV_stats | Training results of 5 models for each of the training iteration (accuracy, overall mean, overall std) |
| main folder | pred_stats | Prediction results all the 25 prediction iterations (accuracy, precision, recall, F1-score, overall mean, overall std) |

-> *m* denotes the number of training sessions


**Step 3 - Prediction**
-> Refer to section **Use pre-trained RiFNet**. Addtionally, select one of the 25 trained models *RiFNet_CV_m_run_n.h5*, rename the model to RiFNet.py and store it in the main folder.


Cross validation procedure:

<img src="https://github.com/victorjonathanibanez/test/blob/main/graph_CV.png" alt="alt text" width="800" height="500">

## Parameters
*default values:*

**Training Data**
- Splitting all data into test & training data: *default: 15\% / 85\%*
- Splitting training data into training & validation:  *default: 5 folds*

**Images**
- Image size: *according to the settings in section **Preprocessing**, code resizes to 500 x 1000*
- Image type: *JPG, RGB*

**Network Parameters**
- Training epochs: *30*
- Batch size: *15*
- Learning rate: *0.00015*
- Dropout rate: *0.5*
- Reduce learning rate on plateau: *monitor = 'acc', factor = 0.2, patience = 5, min_lr = 0.0005*
- Convolution kernel: *(3,3)*
- Max pooling kernel: *(2,2)*
- Filter size: *8, 16, 32, 64, 128*
- Dense layer size: *500*

## Problems
When encountering issues:
- Please refer to the publication -   DOI: [10.1007/s12024-021-00431-8](https://doi.org/10.1007/s12024-021-00431-8) and/or
- Contact akos.dobay@uzh.ch or victor.ibanez@uzh.ch

## Outlook

We are currently working on RiFNet2, which will be able to distinguish between different rib fracture types and locate them on PMCT images. Updates will follow.

