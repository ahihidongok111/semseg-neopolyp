# Deep Learning Assignment: Colonoscopy Polyp Segmentation

## About
This is the repository for Semantic Segmentation assignment of IT3320E - Intro to Deep Learning, which contains the code for segmentation on a test image taken from [BKAI-IGH NeoPolyp competition on Kaggle](https://www.kaggle.com/competitions/bkai-igh-neopolyp/overview).

## Usage

To use the code, first, clone this repository:

```.bash
git clone https://github.com/ahihidongok111/semseg-neopolyp.git
```

Make sure to change your working directory to this repository.

```.bash
cd semseg-neopolyp
```

Install all required packages by running this following command. You may want to create an additional virtual environment to avoid dependency conflicts.

```.bash
pip install -r requirements.txt
```

Next, download the checkpoint from this [link](https://drive.google.com/file/d/1jahnvdyJMOl8jN294htqPsq25IrmFfeB/view?usp=sharing) and save in in `checkpoint` folder. Make sure the file name is `unet_sce_40_epochs.pth` before continuing. 

To segment a test image, run the following command:

```.bash
python3 infer.py --image_path image.jpeg
```

where `image.jpeg` is the path to your test image. The result is saved as `test_segmented.png` in the same working directory. It should consist of 3 colors: **red**, **green** and **black**, where the **red** region is the region of neoplastic polyps, the **green** region is the region of non-neoplastic polyps, and the **black** region is the background.


Input image             |  Segmented image
:-------------------------:|:-------------------------:
![test](https://github.com/user-attachments/assets/011a9754-dc5d-4d7a-ad43-2c774f359382) | ![test_segmented](https://github.com/user-attachments/assets/adc44d14-cd68-4016-a9ce-76dd30ee5be0)



