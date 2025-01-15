# SynthPar2: Synthetic Faces with Demographic Parity

SynthPar2 aims to facilitate the development and evaluation of face recognition models, with the goal of closing the gap in performance across all demographic groups.

It provides 2 key resources:

- A conditional StyleGAN2 [generator](https://huggingface.co/pravsels/synthpar2) that allows users to create synthetic face images for specified attributes like skin color and sex.
    
- A [dataset](https://huggingface.co/datasets/pravsels/synthpar2) of ~8.9M synthetic face images with diverse skin colors and 2 sexes (Male, Female), built with [VGGFace2](https://github.com/ox-vgg/vgg_face2) dataset and labels.


## Loading the dataset

The dataset can be loaded from the HuggingFace repository:

```
from datasets import load_dataset

dataset = load_dataset("pravsels/synthpar2")
```

To load the images in a folder of your choosing and have control over the number of sharded zip files to download, use the `hf_dataset_dload.py` script. 


## Generating images

Setup anaconda environment:
```
./install_conda_env.sh
```

Activate the environment:
```
conda activate synthpar2
```

Run the generation script with the desired configuration:
```
python generation_script.py -c configs/ST2.yaml
```

Please find the configs for the other demographics in the `configs` folder. 


## Licence 

The code, [dataset](https://huggingface.co/datasets/pravsels/synthpar2) and [model](https://huggingface.co/pravsels/synthpar2) are released under the MIT license. 

