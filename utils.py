import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import center_of_mass

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
  c_o_m = np.array(c_o_m)[:,2:]
  #Swap x and y
  c_o_m = c_o_m[:,::-1]
  return c_o_m