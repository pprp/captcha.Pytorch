import torch 
import numpy as np 
class Cutout(object): 
    """Randomly mask out one or more patches from an image.
    Args:n_holes (int): 
        Number of patches to cut out of each image. 
        length (int): The length (in pixels) of each square patch. 
    """ 
    def __init__(self, n_holes, length): 
        self.n_holes = n_holes self.length = length 
        
    def __call__(self, img): 
        """ Args:img (Tensor): 
                Tensor image of size (C, H, W). 
            Returns: 
                Tensor: Image with n_holes of dimension length x length cut out of it. 
        """ 
        h = img.size(1) w = img.size(2) 
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h) 
            x = np.random.randint(w) 
            y1 = np.clip(y - self.length // 2, 0, h) 
            y2 = np.clip(y + self.length // 2, 0, h) 
            x1 = np.clip(x - self.length // 2, 0, w) 
            x2 = np.clip(x + self.length // 2, 0, w) 
            mask[y1: y2, x1: x2] = 0. 

        mask = torch.from_numpy(mask) 
        mask = mask.expand_as(img) 
        img = img * mask 
        return img


## mixup
for (images, labels) in train_loader:
    l = np.random.beta(mixup_alpha, mixup_alpha) 
    index = torch.randperm(images.size(0)) 
    images_a, images_b = images, images[index] 
    labels_a, labels_b = labels, labels[index] 
    mixed_images = l * images_a + (1 - l) * images_b 
    outputs = model(mixed_images) 
    loss = l * criterion(outputs, labels_a) + (1 - l) * criterion(outputs, labels_b) 
    acc = l * accuracy(outputs, labels_a)[0] + (1 - l) * accuracy(outputs, labels_b)[0]
