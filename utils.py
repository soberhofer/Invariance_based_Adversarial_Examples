import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.ndimage import center_of_mass
from pytorch_grad_cam import base_cam
from PIL import Image
import os, random
from tqdm import tqdm
from AdvExample import AdvExample

#Helper Function to plot
def imshow(inp, title=None, denorm=True):
  inp = inp.numpy().transpose((1, 2, 0))
  if denorm:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp=std*inp+mean
    inp = np.clip(inp, 0, 1)
  plt.imshow(inp)
  if title is not None:
    plt.title(title)


def imagenette_outputs(orig_outputs):
  imagenette_classes = [0,217,482,491,497,566,569,571,574,701]
  outputs = torch.index_select(orig_outputs,1,torch.tensor(imagenette_classes).to(orig_outputs.device))
  _, preds = torch.max(outputs, 1)
  return preds

def multiple_c_o_m(input):
  labels = np.ones_like(input).cumsum(0)
  c_o_m = center_of_mass(input,labels, index=np.arange(1,input.shape[0]+1))
  # We're only interested in the x and y coordinates
  c_o_m = np.array(c_o_m)[:,1:]
  #Swap x and y
  c_o_m = c_o_m[:,::-1]
  return c_o_m

def shift(image, offsets):
  dx, dy = offsets
  #Translation Matrix
  T = np.float32([[1, 0, dx], [0, 1, dy]]) 
  shifted = cv2.warpAffine(image.transpose(1,2,0), T, (image.shape[1], image.shape[2])) 
  return shifted.transpose(2,0,1)

def sort_pairs(data, labels, bs):
  correct = False
  tries = 0
  while (not correct and tries < bs**2):
      swapped = False
      for idx, img in enumerate(data):
        if idx >= len(labels)/2:
          break
        if labels[idx] == labels[idx+int((len(labels)/2))]:
          labels[idx], labels[idx+1] = labels[idx+1], labels[idx]
          data[idx], data[idx+1] = data[idx+1], data[idx]
          swapped = True
      if not swapped:
        correct = True
      tries += 1

def set_seeds(seed: int = 42):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  random.seed(seed)
