# GenDP: 3D Semantic Fields for Category-Level Generalizable Diffusion Policy 

[Website](https://robopil.github.io/GenDP/) | [Paper](https://arxiv.org/abs/2410.17488) | [Colab](https://colab.research.google.com/drive/1Yk6uDg2So9A3yWALR7mF5dhl_KKR24eh?usp=sharing) | [Video](https://youtu.be/6jUGmUaAEOc)

<a target="_blank" href="https://wangyixuan12.github.io/">Yixuan Wang</a><sup>1</sup>,
<a target="_blank" href="https://robopil.github.io/GenDP/">Guang Yin</a><sup>2</sup>,
<a target="_blank" href="https://binghao-huang.github.io/">Binghao Huang</a><sup>1</sup>,
<a target="_blank" href="https://kelestemur.com/">Tarik Kelestemur</a><sup>3</sup>,
<a target="_blank" href="https://www.robo.guru/">Jiuguang Wang</a><sup>3</sup>,
<a target="_blank" href="https://yunzhuli.github.io/">Yunzhu Li</a><sup>1</sup>
            
<sup>1</sup>Columbia University,
<sup>2</sup>University of Illinois Urbana-Champaign,
<sup>3</sup>Boston Dynamics AI Institute<br>


## :bookmark_tabs: Table of Contents
- [Install](#hammer-install)
- [Generate Dataset](#floppy_disk-generate-dataset)
    - [Convert Whole Episodes](#convert-whole-episodes)
- [Train](#gear-train)
    - [Train in Real](#train-in-real)
    - [Config Explanation](#config-explanation)
- [Infer in Simulation](#video_game-infer-in-simulation)

## :hammer: Install
```console
mamba env create -f conda_environment.yaml
conda activate gendp
pip install -e gendp/
pip install -e sapien_env/
pip install -e robomimic/
pip install -e d3fields_dev/
```

## :floppy_disk: Generate Dataset
### Generate Camera Extrinsics
```console
python generate_extrinsics.py --calib_dir [PATH_TO_CALIBRATION]
```
`[PATH_TO_CALIBRATION]`: Path to calibration/ folder (containing rvecs.npy and tvecs.npy).

### Convert Single Episode Raw File
```console
cd [PATH_TO_REPO]/data
python convert_raw_to_hdf5.py --episode_dir cube_picking_processed/episode_0000 --output_path [OUTPUT_DIR]/episode_0.hdf5
```

### Convert Whole Episodes
Once collected Raw Data from Real World Robot, please follow below to convert your data for GenDP training
```console
cd [PATH_TO_REPO]/data
python batch_convert_raw_to_hdf5.py --input_root cube_picking_processed/ --output_root [OUTPUT_DIR]
```
### Code Explanation
`cube_picking_processed` is the raw dataset folder collected from real robot, which has `episode_0000` to `episode_N` subfolders, make sure each episode subfolder has:
- `calibration`: contains `base.pkl` to convert robot base to world frame; `intrinsics.npy`; `rvecs.npy` and `tvecs.npy` to generate `extrinsics.npy`.
- `camera_0` to `camera_N`, `0` to `6` cameras at most. Each camera folder contains `/depth` and `/rgb` folders for images.
- `robot`: `000000.txt` to `N_steps.txt` to record robot pose.
```console
1.978241334213923786e-01 1.320325309243938761e-01 -2.688415424187221014e-01
9.952046651029050617e-01 -1.143580994124750519e-02 9.714369155226811048e-02
1.322502194812277820e-02 9.997542005365072093e-01 -1.779430538349144331e-02
-9.691632139060073203e-02 1.899370318281683873e-02 9.951112831676248716e-01
8.010000000000000000e+02 0.000000000000000000e+00 0.000000000000000000e+00
```

## :gear: Train

### Train in Real
Run the following command for training:
```console
cd [PATH_TO_REPO]/gendp
python train.py --config-dir=config/[TASK_NAME] --config-name=distilled_dino_N_4000.yaml training.seed=42 training.device=cuda training.device_id=0 data_root=[PATH_TO_DATA]
```
For example, to train on `cube_real` in my local machine, I run the following command:
```console
python train.py --config-dir=config/cube_real --config-name=distilled_dino_N_1000.yaml training.seed=42 training.device=cuda training.device_id=6 data_root=/home/hz2999/gendp/
```
Please wait at least till 2 epoches to make sure that all pipelines are working properly. 

### Config Explanation
There are several critical entries within the config file. Here are some explanations:
```yaml
shape_meta: shape_meta contains the policy input and output information.
    action: output information
        shape: action dimension. In our work, it is 10 = (3 for translation, 6 for 6d rotation*, 1 for gripper)
        key: [optional] key for the action in the dataset. It could be 'eef_action' or 'joint_action'. Default is 'eef_action'.
    obs: input information
        ... # other input modalities if needed
        d3fields: 3D semantic fields
            shape: shape of the 3D semantic fields, i.e. (num_channel, num_points)
            type: type of inputs. It should be 'spatial' for point cloud inputs
            info: information of the 3D semantic fields.
                reference_frame: frame of input semantic fields. It should be 'world' or 'robot'
                distill_dino: whether to add semantic information to the point cloud
                distill_obj: the name for reference features, which are saved in `d3fields_dev/d3fields/sel_feats/[DISTILL_OBJ].npy`.
                view_keys: viewpoint keys for the semantic fields.
                N_gripper: number of points sampled from the gripper.
                boundaries: boundaries for the workspace.
                resize_ratio: our pipeline will resize images by this ratio to save time and memory.
task:
    env_runner: the configuration for the evaluation environment during the training
        max_steps: maximum steps for each episode, which should be adjusted according to the task
        n_test: number of testing environments
        n_test_vis: number of testing environments that will be visualized on wandb
        n_train: number of training environments
        n_train_vis: number of training environments that will be visualized on wandb
        train_obj_ls: list of objects that appear in the training environments
        test_obj_ls: list of objects that appear in the testing environments
training:
    checkpoint_every: the frequency of saving checkpoints
    rollout_every: the frequency of rolling out the policy in the env_runner
```
Also, the configuration might be repetitive in the config file. Please sync them manually.

## :video_game: Infer in Simulation
To run an existing policy in the simulator, use the following command:
```console
cd [PATH_TO_REPO]/gendp
python eval.py --checkpoint [PATH_TO_CHECKPOINT] --output_dir [OUTPUT_DIR] --n_test [NUM_TEST] --n_train [NUM_TRAIN] --n_test_vis [NUM_TEST_VIS] --n_train_vis [NUM_TRAIN_VIS] --test_obj_ls [OBJ_NAME_1] --test_obj_ls [OBJ_NAME_2] --data_root [PATH_TO_DATA]
```
For example, we can run
```console
python eval.py --checkpoint /home/hz2999/gendp/data/outputs/2025.03.30/23.25.56_train_diffusion_unet_hybrid_cube/checkpoints/latest.ckpt --output_dir /home/hz2999/gendp/eval_results/cube_real --n_test 10 --n_train 10 --n_test_vis 5 --n_train_vis 5 --test_obj_ls cube  --data_root /home/hz2999/gendp --dataset_dir /home/hz2999/gendp/
```

### Adapt to New Task
To adapt our framework to new tasks, you could follow the following steps:
1. You can select reference DINO features by running `python d3fields_dev/d3fields/scripts/sel_features.py`. This will provide an interactive interface to select the reference features given four arbitrary images. Click left mouse button to select the reference features and 'N' to next image. Click `Q` to quit and save the selected features.
2. For the new task, you may need to update several important configuration entries.
```console
shape_meta:
    action:
        shape: 10 if using single robot and 20 for bimanual manipulation
    obs:
        d3fields:
            shape: change the first number (number of channel). It is 3 if only using raw point cloud. It is 3 + number of reference features if using DINOv2 features.
            info:
                distill_dino: whether to add semantic information to the point cloud
                distill_obj: the name for reference features, which are saved in `d3fields_dev/d3fields/sel_feats/[DISTILL_OBJ].npy`.
                bounding_box: the bounding box for the workspace
task_name: name for tasks, which will be used in wandb and logging files
dataset_name: the name for the training dataset, which will be used to infer dataset_dir (e.g. ${data_root}/data/real_aloha_demo/${dataset_name} or  ${data_root}/data/sapien_demo/${dataset_name})
```

## :pray: Acknowledgement

This repository is built upon the following repositories. Thanks for their great work!
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [robomimic](https://github.com/ARISE-Initiative/robomimic)
- [D3Fields](https://github.com/WangYixuan12/d3fields)
