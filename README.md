# MMSIN: A Hybrid No-Reference Point Cloud Quality Assessment Framework

This is the GitHub repository that refers to the master's thesis [MMSIN: A Hybrid No-Reference Point Cloud Quality Assessment Framework](https://github.com/MichielCreemers/MMSIN/blob/main/thesis_mmsin_coosemans_jens_creemers_michiel.pdf) that has been submitted by Jens Coosemans & Michiel Creemers.

This github has a main branch, containing all the necessary files to train your own Point Cloud Quality Assessment (PCQA) model, assess the quality of a single point cloud, as well as the necessary files needed to run the PCQA-tool.

## Motivation
The motivation for this thesis started from the fact that as of today, no efficient No-Reference (NR) PCQA model exsits that is able to accurately predict the quality of a 3D-Point cloud. Because of this, we propose a hybrid multimodal model calles MMSIN that harnesses the benefits from both a Resnet based Deep learning network and a Statistical Machine Learning model.

## Framework
![Overview of the entire model](https://github.com/MichielCreemers/MMSIN/blob/main/imgs/complete_model.png)

Our approach leverages two feature modalities to predict quality metrics. The first being features from the 3D data itself, while the second modality contains features extracted by first projecting a point cloud to a 2D plane. Features are extracted from two modalities using both a statistical machine learning modal and a deep learning model. These features are then enhanced through mutual guidance using symmetric cross-modal attention, resulting in a final feature representation consisting both of the original and enhanced features. Ultimately, this feature representation is decoded into a single quality prediction throughthe quality regression model.

## Getting Started.
In order to train the model, several components and libraries need to be installed.

First, make sure that the correct Nvidia drivers are installed. We trained the model on linux 22.04 with Nvidia driver-545 with cuda 12.3. These can be downloaded from [here](https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network). To check wether the correct version is installed, you can run the following command. 
```bash
  nvidia-smi
```
To run the code itself, several packages need to be installed. An overview is given in [requirements.txt](https://github.com/MichielCreemers/MMSIN/blob/main/requirements.txt). Installing all libraries at once can be done by running the following command:

```bash
pip install -r requirements.txt
```
## Model Training
In order to train your own model, the projections of the point clouds from the different datasets need to be downloaded from [here](). The unzipped folder should be put in the folder SJTU/WPC/WPC2.0 depending on the downloaded projections. The following structure should be obtainend

```
SJTU-+-SJTU_MOS.csv
     |
     +-SJTU_NSS.csv
     |
     +-projections-+-hhi_0.ply-+-projection_0.png
                   |...        |...
                   |...        +-projection_5.png
                   |...
                   +-ULB_Unicorn_41.ply+-projection_0.png
                                       |...
                                       +-projection_5.png


```
The reference point clouds aren't provided but can be downloaded/access can be requested from their source.

The natural scene statistics, and their scalers are already be available. After this is added, the parameters of the model can be tweaked in the config.json. The standard configuration is the following where gpu 0 represents the dedicated graphics card on Linux

```json
{
    "gpu": 0,
    "num_epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.001,
    "decay_rate": 0.0001,
    "model": "nss1",
    "projections_dirs" : ["SJTU/projections"],
    "mos_data_paths": ["SJTU/SJTU_MOS.csv"],
    "nss_features_paths" : ["SJTU/SJTU_NSS.csv"],
    "number_of_projections": 6,
    "loss": "l2rank",
    "datasets": ["SJTU"],
    "k_fold_num": 5
}
```

After the wanted parameters are chosen, the model can be trained by running 

```bash
./train.sh
```
The outputs will be saved in a log file that is specified in train.sh, and the best model will be saved in the ckpts folder.






