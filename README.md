# HDLayout

[[`Paper`](https://)] [[`Dataset`](https://)] [[`Blog`](https://hxiangdou.github.io/HDLayout/)] [[`BibTeX`](#CitingHDLayout)]

![HDLayout design](assets/model_diagram.png?raw=true)

The **HDLayout** outperforms several strong baselines in a variety of scenarios both qualitatively and quantitatively, yielding state-of-the-art performances on arbitrarily shaped visual text generation. It has been trained on a [dataset](https://) of 2,749 training data and 813 test data, and has strong zero-shot performance on a variety of segmentation tasks.

## Installation

The code requires `python>=3.9`, as well as `pytorch>=2.2.2` and `torchvision>=0.17.2`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Install HDLayout:

```
pip install git+https://github.com/imu-group/hdlayout.git
```

or clone the repository locally and install with

```
git clone git@github.com:imu-group/hdlayout.git
```

## <a name="GettingStarted"></a>Getting Started

First, download a [model checkpoint](#model-checkpoints)

Additionally, the layout can be generated from the command line:

```
python inference.py
```

## Checkpoint

Click the links below to download the checkpoint for the corresponding model type.

- **`default`: [HDLayout model.](https://)**

## Dataset

The dataset can be downloaded [here](https://). By downloading the datasets you agree that you have read and accepted the terms of the HDLayout Research License.

## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Citing HDLayout

If you use HDLayout in your research, please use the following BibTeX entry.

```
@inproceedings{feng2025hdlayout,
    title={HDLayout: Hierarchical and Directional Layout Planning for Arbitrary Shaped Visual Text Generation},
    author={Tonghui, Feng and Chunsheng, Yan and Qianru, Wang and Jiangtao, Cui and Xiaotian, Qiao},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2025}
}
```
