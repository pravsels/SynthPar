
import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *

# other modules
import os
import numpy as np

from torch.autograd import Variable
from torchvision.utils import make_grid
import torch
import time
import cv2
import glob
from util import parse_arguments, get_config
from tqdm import tqdm

command_line_args = parse_arguments()
config = get_config(command_line_args.config)

race_folder = config.race_folder
batch_size = config.batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('device: ', device)

# ---------------- create normal for rendering half sphere ------
img_size = 256
x = np.linspace(-1, 1, img_size)
z = np.linspace(1, -1, img_size)
x, z = np.meshgrid(x, z)

mag = np.sqrt(x**2 + z**2)
valid = mag <=1
y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
x = x * valid
y = y * valid
z = z * valid
normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
normal = np.reshape(normal, (-1, 3))
#-----------------------------------------------------------------

modelFolder = 'trained_model/'

# load model
from defineHourglass_512_gray_skip import *
my_network = HourglassNet()
my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
my_network.to(device)
my_network.train(False)

lightFolder = 'data/example_light/'
generated_images_folder = config.dataset_folder

race_folder = os.path.join(generated_images_folder, race_folder)
print('race folder : ', race_folder)
image_folders = [os.path.join(race_folder, name) for name in os.listdir(race_folder) if os.path.isdir(f'{race_folder}/{name}')]
for image_folder in image_folders:
    image_files = [f for f in glob.glob(f'{image_folder}/*.png') if '_lighting_' not in f]
    for batch_start in range(0, len(image_files), batch_size):
        batch_files = image_files[batch_start:batch_start+batch_size]
        
        inputL_batch = []
        Lab_batch = []
        row_batch = []
        col_batch = []
        image_names = []
        
        for image_file in batch_files:
            image_name = image_file.split('/')[-1].split('.')[0]
            img = cv2.imread(image_file)
            if img is None:
                print(f"Warning: Unable to read image {image_file}. Skipping.")
                continue
            row, col, _ = img.shape
            img = cv2.resize(img, (512, 512))
            Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            inputL = Lab[:,:,0]
            inputL = inputL.astype(np.float32)/255.0
            inputL = inputL.transpose((0,1))
            inputL = inputL[None,None,...]
            
            inputL_batch.append(inputL)
            Lab_batch.append(Lab)
            row_batch.append(row)
            col_batch.append(col)
            image_names.append(image_name)
        
        if not inputL_batch:
            print(f"No valid images in this batch. Skipping.")
            continue

        inputL_batch = Variable(torch.from_numpy(np.concatenate(inputL_batch, axis=0)).to(device))
        
        for i in tqdm(range(7)):
            sh = np.loadtxt(os.path.join(lightFolder, 'rotate_light_{:02d}.txt'.format(i)))
            sh = sh[0:9]
            sh = sh * 0.7
            
            # --------------------------------------------------
            # rendering half-sphere
            sh = np.squeeze(sh)
            shading = get_shading(normal, sh)
            value = np.percentile(shading, 95)
            ind = shading > value
            shading[ind] = value
            shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
            shading = (shading *255.0).astype(np.uint8)
            shading = np.reshape(shading, (256, 256))
            shading = shading * valid
            # cv2.imwrite(os.path.join(save_folder, \
            #         'light_{:02d}.png'.format(i)), shading)
            # --------------------------------------------------
            
            # ----------------------------------------------
            # rendering images using the network
            sh = np.reshape(sh, (1,9,1,1)).astype(np.float32)
            sh = Variable(torch.from_numpy(sh).to(device))
            outputImgs, outputSH = my_network(inputL_batch, sh, 0)
            outputImgs = outputImgs.cpu().data.numpy()
            outputImgs = outputImgs.transpose((0,2,3,1))
            outputImgs = np.squeeze(outputImgs)
            outputImgs = (outputImgs*255.0).astype(np.uint8)
            
            for j, Lab in enumerate(Lab_batch):
                Lab[:,:,0] = outputImgs[j]
                resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)
                resultLab = cv2.resize(resultLab, (col_batch[j], row_batch[j]))

                output_filename = f'{image_names[j]}_lighting_{i:02d}.png'
                output_path = os.path.join(image_folder, output_filename)
                if not os.path.exists(output_path):
                    cv2.imwrite(output_path, resultLab)
                else:
                    print(f"Skipping {output_filename} as it already exists.")

            # ----------------------------------------------
