# README

This repository contains official code for our research paper "Synthesizing Long-Term Human Motions with Diffusion Models via Coherent Sampling" (https://dl.acm.org/doi/10.1145/3581783.3611887).

## Environment setup
To set up the environment, we provide an `requirements.txt` file that you can use with pip:
```
pip install -r requirements.txt
```
## Data preparation
Due to licensing restrictions, we are unable to provide pre-processed data directly. However, you can refer to [TEACH](https://github.com/athn-nik/teach#data) for specific data processing methods.

Eventually, you should have a folder `data` with such a structure:
```
data
|-- babel
|   `-- babel_v2.1
|       `...
|   `-- babel-smplh-30fps-male 
|       `...
|
|-- smpl_models
|   `-- smplh
|       `--SMPLH_MALE.pkl
```

Besides, you should download the folder `deps` in [TEACH](https://github.com/athn-nik/teach/tree/main/deps) to this project.

## Pre-trained weights
We provide the pretrained models here: [pretrained models link](https://drive.google.com/drive/folders/1Lrj5FEt7bFFiv_VnfoDFoQgZzfF4X6RJ?usp=sharing). The 'pretrained.zip' file contains the pretrained model and training configurations used to report metrics in our paper, while 'MotionCLIP.zip' contains the model used for evaluation.

## Running the code
You can use the following three commands to obtain the results for the last three rows of the experimental results table in our paper:
```
python eval_humanml.py --model_path ./save/past_cond_hist_frame_5_mask_newdata/model000600000.pt --guidance_param 2 --inpainting_frames 0

python eval_humanml.py --model_path ./save/past_cond_hist_frame_5_mask_newdata/model000600000.pt --guidance_param 2 --inpainting_frames 2

python eval_humanml.py --model_path ./save/past_cond_hist_frame_5_mask_newdata/model000600000.pt --guidance_param 2 --composition True --inter_frames 2
```

Besides, if you want to train a model from scratchm, you can use this comman:
```
python train_mdm.py --save_dir ./save/pcmdm --dataset babel --hist_frames 5 
```

## Acknowledgments
Our code is based on [TEACH](https://github.com/athn-nik/teach) and [MDM](https://github.com/GuyTevet/motion-diffusion-model). Thanks for their greate work!
