# MMSIN: A Hybrid No-Reference Point Cloud Quality Assessment Framework

This is the GitHub repository that refers to the master's thesis [MMSIN: A Hybrid No-Reference Point Cloud Quality Assessment Framework](https://github.com/MichielCreemers/MMSIN/blob/main/thesis_mmsin_coosemans_jens_creemers_michiel.pdf) that has been submitted by Jens Coosemans & Michiel Creemers.

This github has a main branch, containing all the necessary files to train your own Point Cloud Quality Assessment (PCQA) model, assess the quality of a single point cloud, as well as the necessary files needed to run the PCQA-tool.

## Motivation
The motivation for this thesis started from the fact that as of today, no efficient No-Reference (NR) PCQA model exsits that is able to accurately predict the quality of a 3D-Point cloud. Because of this, we propose a hybrid multimodal model calles MMSIN that harnesses the benefits from both a Resnet based Deep learning network and a Statistical Machine Learning model.

## Framework
![Overview of the entire model](https://github.com/MichielCreemers/MMSIN/blob/main/imgs/complete_model.png)