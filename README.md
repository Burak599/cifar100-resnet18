# CIFAR-100 ResNet18 with CutMix

This project trains a ResNet18 on CIFAR-100 using PyTorch.  
It demonstrates a modern training pipeline with data augmentation, CutMix, label smoothing, and learning rate scheduling.

## Features
- Pretrained ResNet18 backbone
- Data augmentation (RandomCrop, RandomHorizontalFlip, Normalize)
- CutMix (via timm)
- Label smoothing
- AdamW optimizer with OneCycleLR scheduler
- Early stopping to prevent overfitting

## Usage
Clone the repository and install requirements:
```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt

## Results
- Training Accuracy: ~84%
- Validation Accuracy: 79–80%
- Model checkpoints saved as `best_model.pth` and `resnet18_cifar100_project.pth`
