from pathlib import Path
import torchvision
from PIL import Image


class ImagenetteDataset(object):
    def __init__(self, patch_size, validation=False, should_normalize=True):
        self.folder = Path(f'imagenette2-{patch_size}/train') if not validation else Path(f'imagenette2-{patch_size}/val')
        self.classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']

        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)

        self.patch_size = patch_size
        self.validation = validation

        self.random_resize = torchvision.transforms.RandomResizedCrop(patch_size)
        self.center_resize = torchvision.transforms.CenterCrop(patch_size)
        self.should_normalize = should_normalize
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        image_fname = self.images[index]
        image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)

        if not self.validation: image = self.center_resize(image)
        else: image = self.center_resize(image)

        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, self.patch_size, self.patch_size)
        if self.should_normalize: image = self.normalize(image)
        return image, label

    def __len__(self):
        return len(self.images)