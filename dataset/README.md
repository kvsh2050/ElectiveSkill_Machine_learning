# Wildlife Animals YOLO Dataset

## Dataset Overview

This dataset contains annotated wildlife animal images prepared for YOLO (You Only Look Once) model training. The dataset has been carefully collected and annotated to facilitate object detection tasks in wildlife monitoring and conservation applications.

## Dataset Structure

Here's the folder structure of the YOLO-formatted dataset:

```
wildlife_yolo_dataset/
│
├── images/
│   ├── train/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image_101.jpg
│   │   ├── image_102.jpg
│   │   └── ...
│   └── test/
│       ├── image_201.jpg
│       ├── image_202.jpg
│       └── ...
│
├── labels/
│   ├── train/
│   │   ├── image_001.txt
│   │   ├── image_002.txt
│   │   └── ...
│   ├── val/
│   │   ├── image_101.txt
│   │   ├── image_102.txt
│   │   └── ...
│   └── test/
│       ├── image_201.txt
│       ├── image_202.txt
│       └── ...
│
├── data.yaml
└── README.md
```

## Folder Contents Explanation

### 1. images/ directory
Contains all the wildlife animal images divided into three subsets:
- `train/`: Training images (typically 70-80% of total dataset)
- `val/`: Validation images (typically 10-15% of total dataset)
- `test/`: Test images (typically 10-15% of total dataset)

Image formats can be JPG, PNG, or other common formats supported by YOLO.

### 2. labels/ directory
Contains the annotation files corresponding to the images, with matching filenames but `.txt` extension. The directory structure mirrors the images/ directory.

Each `.txt` file contains annotations in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```
- Values are normalized (0-1) relative to image dimensions
- One object per line
- Example: `0 0.4453125 0.5324074 0.1822917 0.2592593`

### 3. classes.txt
A text file listing all wildlife animal classes in the dataset, one per line. Example:
```
elephant
lion
zebra
giraffe
rhinoceros
```

## YOLO Dataset Format Details

The dataset follows the standard YOLO format requirements:
1. Image and corresponding label files have the same name (different extensions)
2. Annotations are in normalized space (0-1)
3. Directory structure separates training, validation, and test sets
4. Class IDs correspond to line numbers in classes.txt (starting from 0)

## Dataset Statistics

- Total images: [X]
- Training images: [Y]
- Validation images: [Z]
- Test images: [W]
- Classes: [List of animal species included]
- Average annotations per image: [A]

## Usage

To use this dataset with YOLO (v5/v7/v8), you can reference it in your dataset YAML file:

```yaml
# dataset.yaml
path: /path/to/wildlife_yolo_dataset
train: images/train
val: images/val
test: images/test

nc: [number of classes]
names: [list of class names from classes.txt]
```

## Annotation Guidelines

The dataset was annotated following these principles:
1. Tight bounding boxes around visible animal bodies
2. Partial/occluded animals included when identifiable
3. Multiple animals in same image annotated separately
4. Challenging conditions (low light, motion blur) included for robustness

