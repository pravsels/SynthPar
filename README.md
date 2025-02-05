# SynthPar: Synthetic Faces for Demographic Parity in Face Recognition Performance 

SynthPar aims to facilitate the development and evaluation of face recognition models, with the goal of closing the gap in performance across all demographic groups.

It provides 2 key resources:

- A conditional StyleGAN2 [generator](https://huggingface.co/pravsels/synthpar) that allows users to create synthetic face images for specified attributes like skin color and sex.
    
- A [dataset](https://huggingface.co/datasets/pravsels/synthpar) of ~8.9M synthetic face images with diverse skin colors and 2 sexes (Male, Female), built with [VGGFace2](https://github.com/ox-vgg/vgg_face2) dataset and labels.


## Loading the dataset

The dataset can be loaded from the HuggingFace repository:

```
from datasets import load_dataset

dataset = load_dataset("pravsels/synthpar")
```

To download a subset of the dataset, use the `hf_dataset_dload.py` script. But first, you need to setup a conda environment. 

Change permissions for `install_conda_env.sh` and execute it:
```
chmod u+x ./install_conda_env.sh

./install_conda_env.sh
```

Activate the environment:
```
conda activate synthpar
```

Run the download script and select from subsets ST1 through ST8, which are subsets with specific skin tone regions and sex. 
```
python hf_dataset_dload.py

Enter ST type (e.g. ST1, ST2 ... ST8):
```


## Generating images 

To generate images, you need to setup a docker container. 

To build and run the docker container, use `docker_build.sh` and `run_docker_container.sh` scripts respectively.

Once the container is running, the generation script can be run with the desired configuration:
```
python generation_script.py -c configs/ST2.yaml
```
This generates faces and several variations in pose and expression per face. 


To generate variations in lighting on top of these, please run: 
```
python network_demo_512.py -c configs/ST2.yaml
```
By default, it generates 7 lighting variations per image. 

## Licence 

The code, [dataset](https://huggingface.co/datasets/pravsels/synthpar) and [model](https://huggingface.co/pravsels/synthpar) are released under the MIT license. 

