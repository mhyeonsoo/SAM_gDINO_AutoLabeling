# SAM_gDINO_AutoLabeling
Auto Segmentation label generation with SAM (Segment Anything) + Grounding DINO

- This is a project based on the [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- Thanks to [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and [Yolov8-SAM](https://github.com/akashAD98/YOLOV8_SAM) to developing wonderful codes using SAM.


## How To Use
### 1. Requirements
- You should set the environment variable manually as follows if you want to build a local GPU environment for Grounded-SAM first.
```shell
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
```
- Install Segment Anything:
```shell
python -m pip install -e segment_anything
```
- Install Grounding DINO:
```shell
python -m pip install -e GroundingDINO
```
- Install diffusers:
```shell
pip install --upgrade diffusers[torch]
```
- These are necessary packages which may be required to run post-processes
```shell
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
```

### 2. How to Run the Label generation scripts.
#### 1. Download the pretrained weights
```shell
# download the pretrained groundingdino-swin-tiny model
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# move downloaded model to ./weights directory
```

#### 2. Run the jupyter notebook
[Auto-Label-Generation](https://github.com/mhyeonsoo/SAM_gDINO_AutoLabeling/blob/main/Auto_labeling.ipynb) is a notebook for generating labels

**- Format of generated annotation**
```
1. Yolov8
2. COCO
```

#### 3. Custom configs / arguments
To get the labels for your custom image inputs, we need to modify these in the [Auto-Label-Generation](https://github.com/mhyeonsoo/SAM_gDINO_AutoLabeling/blob/main/Auto_labeling.ipynb) notebook.

**1. text prompt input for grounding DINO**
```python
TEXT_PROMPT = "[classes of custom dataset]" 
# (for multi-class, you can add . between each class --> 'class1 . class2 . class3')
```
**2. category list for COCO annotation**
```python
CAT_ID = {'class1': 1, 'class2': 2, 'class3': 3, 'class4': 4, 'class5': 5, 'class6': 6}
# COCO category list is in form of a python dictionary
```

## Aknowledgements
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Segment-Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Yolov8-SAM](https://github.com/akashAD98/YOLOV8_SAM)
