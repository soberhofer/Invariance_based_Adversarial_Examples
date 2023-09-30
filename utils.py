import numpy as np
import matplotlib.pyplot as plt
import torch
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