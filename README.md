# HDLayout

[[`Paper`](https://github.com/Hxiangdou/HDLayout/blob/main/src/assets/paper/AAAI25_CRC.out250103.pdf)] [[`Dataset`](https://drive.google.com/drive/folders/1kNas4WF7FscC43Lw-AV-BrhrKjZ3h0UK?usp=sharing)] [[`Project`](https://hxiangdou.github.io/HDLayout/)] [[`BibTeX`](#CitingHDLayout)]

![HDLayout design](assets/pipeline.jpg?raw=true)

The **HDLayout** outperforms several strong baselines in a variety of scenarios both qualitatively and quantitatively, yielding state-of-the-art performances on arbitrarily shaped visual text generation.

## Installation

The code requires `python>=3.9`, as well as `pytorch>=2.2.2` and `torchvision>=0.17.2`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

1. Install HDLayout:

```
pip install git+https://github.com/imu-group/hdlayout.git
```

or clone the repository locally and install with

```
git clone https://github.com/IMU-Group/HDLayout.git
```

2. Click the links below to download the checkpoint for the corresponding model type.
- **`default`: [HDLayout model.](https://drive.google.com/file/d/11SVTAViwBOIQaEs9zuIC0hLtHpQiDbT2/view?usp=sharing)**

## Requirements
1. Create environment
```
conda create -n hdlayout python=3.9
conda activate hdlayout
```
2. Install requirements

```
pip install -r requirements.txt
```

## Inference

Additionally, the HDLayout can be generated from the command line:

```
python inference.py --img_path ./test_data --output_dir ./outputs --resume /path/to/checkpoint.pth
```

## Train

Additionally, the HDLayout model can be trained from the command line:

```
python main.py --dataset_path ./path/to/dataset --output_dir ./outputs --device cuda
```

## Dataset

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1kNas4WF7FscC43Lw-AV-BrhrKjZ3h0UK?usp=sharing). By downloading the datasets you agree that you have read and accepted the terms of the HDLayout Research License. You need to download text-free background images from the [link](https://github.com/HCIILAB/SCUT-EnsText) and place them into the corresponding images folders under train/val directories to complete the dataset supplementation.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).



## Citing HDLayout
<a id="CitingHDLayout"></a>
If you use HDLayout in your research, please use the following BibTeX entry.

```
@inproceedings{feng2025hdlayout,
    title={HDLayout: Hierarchical and Directional Layout Planning for Arbitrary Shaped Visual Text Generation},
    author={Tonghui, Feng and Chunsheng, Yan and Qianru, Wang and Jiangtao, Cui and Xiaotian, Qiao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
}
```
