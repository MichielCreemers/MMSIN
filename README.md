# MMSIN: A Hybrid No-Reference Point Cloud Quality Assessment Framework

This is the GitHub repository that refers to the master's thesis [MMSIN: A Hybrid No-Reference Point Cloud Quality Assessment Framework](https://github.com/MichielCreemers/MMSIN/blob/main/thesis_mmsin_coosemans_jens_creemers_michiel.pdf) that has been submitted by Jens Coosemans & Michiel Creemers.

This github has a main branch, containing all the necessary files to train your own Point Cloud Quality Assessment (PCQA) model, assess the quality of a single point cloud, as well as the necessary files needed to run the PCQA-tool.

## Motivation
The motivation for this thesis started from the fact that as of today, no efficient No-Reference (NR) PCQA model exsits that is able to accurately predict the quality of a 3D-Point cloud. Because of this, we propose a hybrid multimodal model calles MMSIN that harnesses the benefits from both a Resnet based Deep learning network and a Statistical Machine Learning model.

## Framework
![Overview of the entire model](https://github.com/MichielCreemers/MMSIN/blob/main/imgs/complete_model.png)

Our approach leverages two feature modalities to predict quality metrics. The first being features from the 3D data itself, while the second modality contains features extracted by first projecting a point cloud to a 2D plane. The network is shown in Figure 4.1. Features are extracted from two modalities using both a statistical machine learning modal and a deep learning model. These features are then enhanced through mutual guid-
ance using symmetric cross-modal attention, resulting in a final feature representation consisting both of the original and enhanced features. Ultimately, this feature representation is decoded into a single quality prediction throughthe quality regression model.

## Training your own model.

