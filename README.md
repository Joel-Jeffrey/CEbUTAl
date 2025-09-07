# CEbUTAl
This is the official repository for CEbUTAl: A Confidence and Entropy-based Uncertainty Thresholding Algorithm.

## About CEbUTAl
Advances in AI and machine learning have transformed medical image analysis, improving diagnostic accuracy, image quality, and measurements. However, the "black-box" nature of deep learning models raises trust and ethical concerns. To address this, we introduce CEbUTAl (Confidence and Entropy-based Uncertainty Thresholding Algorithm), which enhances explainability without compromising performance by adjusting predictions based on uncertainty. CEbUTAl improves classification accuracy to 84% in intracranial hemorrhage detection, 94% in optical coherence tomography, 73% in breast cancer detection, and 63% in multi-class skin lesion classification. Its real-time uncertainty correction enhances model transparency, making it a valuable tool for trustworthy clinical decision-making.

## News
- 12<sup>th</sup>November, 2024 - Submitted to "Trustworthy Artificial Intelligence for Medical Imaging" (TA4MI), Computerized Medical Imaging and Graphics.
- 25<sup>th</sup> April, 2025 - Revised form submitted
- 5<sup>th</sup> August, 2025 - Accepted

## Usage

All the requirements have been put in the requirements.txt file

```pip install -r Code/General/requirements.txt```

## Project Summary
The following models were trained and evaluated with and without CEbUTAl
- SqueezeNet 1.0
- ResNet34
- DenseNet201
- MobileNetV2
- InceptionV3
- ConvNeXt-small

Loss functions used were
- Cross entropy loss

Datasets used
- [RSNA Intracranial Hemorrhage Detection 2019](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)
- [Retinal OCT Images (UCSD Dataset)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [RSNA Screening Mammography Breast Cancer Detection 2023](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview)
- A proprietary clinical dataset
- [MedMNIST2's DermaMNIST](https://zenodo.org/records/10519652)

Studies were done in comparison with popular data imbalance mitigation methods such as
- Data Augmentation
- Focal loss

Comparison with other uncertainty quantification methods, such as Monte Carlo Dropout (MCDO), Deep Ensembles, and Ensemble MCDO, was also done.

## Acknowledgements
+ This work was done at the Medical Imaging Group at the Department of Computational and Data Sciences, Indian Institute of Science.
+ We highly appreciate RSNA and the MedMNIST dataset owners for providing the public dataset to the community.
