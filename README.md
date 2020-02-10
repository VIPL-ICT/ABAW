# Emotion Recognition for In-the-wild Videos
Code for the seven basic expression classification track of Affective Behavior Analysis in-the-wild (ABAW) Competition held in conjunction with the IEEE International Conference on Automatic Face and Gesture Recognition (FG) 2020.

### Requirements
- Python3
- PyTorch
- TorchVision
- NumPy
- Aff-Wild2 Dataset
- Pre-trained models (optional)

### Downloads:
Modal files: [link](https://hanyuliu-my.sharepoint.com/:f:/g/personal/i_hanyuliu_onmicrosoft_com/Es7u4IQatZZEo9ydiBpxFgYB0P-9ZUaombQNbcMHFXFnCQ?e=Ksn6ue)

Dataset: [link](https://mailsucaseducn-my.sharepoint.com/:f:/g/personal/zhangyuanhang15_mails_ucas_edu_cn/EvrhaNkFvfBGtjxASu7YIC0BgSSeP-tfLzz8vWwL1J7g_A?e=c1ylv9)


### Training
Put train.py, face_imgs (from downloads, or use official cropped_aligned), pre-trained modal (optional), EXPR_Set (official annotation) under the same path.
```
python3 train.py
```

### Testing
Put test.py, face_imgs (from downloads, or use official cropped_aligned), modal, expression_test_set.txt, frames.txt under the same path.
```
python3 test.py
```

