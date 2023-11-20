# README

This repository contains code for our research project on motion synthesis, as described in our paper.

## Environment setup
To set up the environment, we provide an `env.yaml` file that you can use with Conda or Miniconda. Please follow the instructions below to create a new environment using the provided file:

```conda env create -f env.yaml\nconda activate <environment-name>```

## Data preparation
Due to licensing restrictions, we are unable to provide pre-processed data directly. However, you can refer to [TEACH](https://github.com/yangzhao1230/newPCMDM) for specific data processing methods.

## Pre-trained weights
We provide the pre-trained weights for the diffusion model and motion clip used in our paper, which can be found at the following link: https://2022cc1.blob.core.windows.net/dnamodel/clinvar_data/. 

## Running the code
To evaluate the performance of the pre-trained model, you can use the following command:

```python eval_humanml.py --model_path ./save/past_cond_hist_frame_5_mask_newdata/model000600000.pt --guidance_param 2 --composition True --inter_frames 2```

Please note that the `model_path` parameter should be set to the path of the pre-trained model you wish to evaluate. For more information on how to use our code, please refer to the documentation and comments in the source files.