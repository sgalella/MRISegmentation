import glob

import numpy as np
import torch
from skimage import io
from torchvision import transforms


class MRIDataset(torch.utils.data.Dataset):
    def __init__(self, folders, indices, train=True):
        """Initializes MRI dataset.

        Args:
            folders (np.array): Name of folders with images.
            indices (np.array): Indices to consider (train or validation).
            train (bool, optional): Dataset for train or validation (changes transforms). Defaults to True.
        """
        self.folders = folders
        self.indices = indices
        self.train = train

        # Find all the subdirectories of root. Split slices and masks.
        files = []
        for folder in self.folders:
            files.extend(glob.glob(folder + '*.tif'))
        slices = [name for name in files if 'mask' not in name]
        masks = [name for name in files if 'mask' in name]

        # Sort files numerically by patient index
        sorted_slices = sorted(slices, key=lambda x: (x.split('_')[2], int(x.split('_')[-1].split('.')[0])))
        sorted_masks = sorted(masks, key=lambda x: (x.split('_')[2], int(x.split('_')[-2])))
        self.datafiles = list(zip(sorted_slices, sorted_masks))

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        image_filename, mask_filename = self.datafiles[idx]
        image, mask = self._load_transform(image_filename, mask_filename)
        return image, mask

    def _load_transform(self, image_filename, mask_filename):
        """Loads and transforms a pair image-mask.

        Args:
            image_filename (str): Name of the image file.
            mask_filename (str): Name of the segmentation of the image file.

        Returns:
            tuple: Processed pair image-mask.
        """
        image = io.imread(image_filename)
        mask = io.imread(mask_filename)

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        if self.train:
            if np.random.random() > 0.5:
                seed = np.random.randint(123456789)
                rotation = transforms.RandomRotation(degrees=(-20, 20))
                torch.manual_seed(seed)
                image = rotation(image)
                torch.manual_seed(seed)
                mask = rotation(mask)

        mask = mask.long()

        return image, mask


def load_MRIDataset(root, batch_size, patients_valid):
    """Loads dataloader for the MRI dataset.

    Args:
        root (str): Path to the directory with the images.
        batch_size (int): Batch size for train and validation.
        patients_valid (int): Number of patients to keep for validation (out of 110).

    Returns:
        tuple: Dataloader and datasets.
    """
    # Split the folders into train and validation.
    folders = np.array(glob.glob(root + '*/'))
    num_folders = len(folders)
    valid_size = round(num_folders * patients_valid / num_folders)
    indices = np.array(range(num_folders))
    np.random.shuffle(indices)
    indices_train = indices[valid_size:]
    indices_valid = indices[:valid_size]
    train_folders, valid_folders = folders[indices_train], folders[indices_valid]

    # Generate dataloaders and datasets
    train_dataset = MRIDataset(train_folders, indices_train, train=True)
    valid_dataset = MRIDataset(valid_folders, indices_valid, train=False)
    trainloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    validloader = torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=2)

    return (trainloader, validloader), (train_dataset, valid_dataset)
