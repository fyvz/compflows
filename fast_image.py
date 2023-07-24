from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FastImageFolder(Dataset):
    #This module preloads the entire dataset to ram.
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.internal_buffer = {}
        tensorize = transforms.ToTensor()
        for a in range(len(self.samples)):
            img = Image.open(self.samples[a]).convert("RGB")
            self.internal_buffer[a]=tensorize(img)
        
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        #img = Image.open(self.samples[index]).convert("RGB")
        img = self.internal_buffer[index]
        if self.transform:
            return self.transform(img)
        return img
    def __getdimension__(self,index):
        img = Image.open(self.samples[index]).convert("RGB")
        return img.size
    def __len__(self):
        return len(self.samples)
