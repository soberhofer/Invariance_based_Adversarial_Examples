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


def imagenette_outputs(orig_outputs):
  imagenette_classes = [0,217,482,491,497,566,569,571,574,701]
  outputs = torch.index_select(orig_outputs,1,torch.tensor(imagenette_classes).to(orig_outputs.device))
  _, preds = torch.max(outputs, 1)
  return preds