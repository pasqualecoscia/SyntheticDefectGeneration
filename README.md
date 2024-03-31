# Adversarial Defect Synthesis - PyTorch

### Overview
This repository contains the PyTorch implementation of [Adversarial Defect Synthesis for Industrial Products in Low Data Regime](https://ieeexplore.ieee.org/document/10222874).

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/pasqualecoscia/SyntheticDefectGeneration
$ cd SyntheticDefectGeneration/
$ pip3 install -r requirements.txt
```

#### Download dataset

Download the [MvTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract the data into the data/mvtec folder. Then, select one product and defect and run:

```bash
$ python3 data/create_mvtec_dataset.py --product product_name --defect defect_name
```

### Train

The following command can be used to train the model.

```bash
$ python3 train.py --cuda
```
See src/train_options.py and src/base_options.py for more details.

### Test

The following command can be used to test the model.

```bash
$ python3 test.py --cuda
```

#### Resume training

If you want to load pre-trained weights, run the following command.

```bash
# Select the epoch to load
$ python3 train.py --cuda\
    --netG_A2B weights/mvtec_dataset/netG_A2B_epoch_100.pth \
    --netG_B2A weights/mvtec_dataset/netG_B2A_epoch_100.pth \
    --netD_A weights/mvtec_dataset/netD_A_epoch_100.pth \
    --netD_B weights/mvtec_dataset/netD_B_epoch_100.pth \
    --netD_fit weights/mvtec_dataset/netD_fit_epoch_100.pth \
    --netD_mask weights/mvtec_dataset/netD_mask_epoch_100.pth
```
### Merics Evaluation

The following command can be used to evaluate the quality of the generated images.

```bash
# Select the epoch to evaluate
$ python3 evaluate.py --cuda --epoch 150
```

#### Classifier

To run the classification experiment, run the following command (different models are supported).

```bash
# Example: resnet18 model for 150 epochs
$ python3 classifier.py --cuda --model resnet18 --batch_size 50 --epochs 150
```

#### Adversarial Defect Synthesis for Industrial Products in Low Data Regime
_Pasquale Coscia, Angelo Genovese, Fabio Scotti, Vincenzo Piuri_ <br>

**Abstract** <br>
Synthetic defect generation is an important aid for advanced manufacturing and production processes. Industrial scenarios rely on automated image-based quality control methods to avoid time-consuming manual inspections and promptly identify products not complying with specific quality standards. However, these methods show poor performance in the case of ill-posed low-data training regimes, and the lack of defective samples, due to operational costs or privacy policies, strongly limits their large-scale applicability.To overcome these limitations, we propose an innovative architecture based on an unpaired image-to-image (I2I) translation model to guide a transformation from a defect-free to a defective domain for common industrial products and propose simultaneously localizing their synthesized defects through a segmentation mask. As a performance evaluation, we measure image similarity and variability using standard metrics employed for generative models. Finally, we demonstrate that inspection networks, trained on synthesized samples, improve their accuracy in spotting real defective products.

```
@INPROCEEDINGS{defsynthesis,
  author={Coscia, Pasquale and Genovese, Angelo and Scotti, Fabio and Piuri, Vincenzo},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)}, 
  title={Adversarial Defect Synthesis for Industrial Products in Low Data Regime}, 
  year={2023},
  pages={1360-1364},
  doi={10.1109/ICIP49359.2023.10222874}}
```
