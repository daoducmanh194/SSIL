import numpy as np
import torch


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def filter_images(dataset, labels, labels_old=None, overlap=True):
    # Filter images without any label in LABELS (using labels not reordered)
    idxs = []

    if 0 in labels:
        labels.remove(0)

    print(f"Filtering images...")
    if labels_old is None:
        labels_old = []
    labels_cum = labels + labels_old + [0, 255]

    if overlap:
        fil = lambda c: any(x in labels for x in cls)
    else:
        fil = lambda c: any(x in labels for x in cls) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
        if i % 1000 == 0:
            print(f"\t{i}/{len(dataset)} ...")
    return idxs 


class Subset(torch.utils.data.Dataset):
    """ A class supports the loader of each dataset. This would return a subset
    of dataset given specified indices.

    The class tries to check whether exists a subset of dataset at a indices or
    not. If it exists, return the sample with its corresponding label, others
    return alerts. Moreover, the class check the given augmentation for subset
    data is used or not.

    Arguments:
        dataset (Dataset): The whole Dataset
        indicies (sequence): Indicies in the whole set selected for subset
        transform (callable): Way to transform the images and the targets
        target_tranform (callable): Way to transform the target labels
    """
    def __init__(self, dataset, indicies, transform=None, target_transform=None):
        # super().__init__()
        self.dataset = dataset
        self.indicies = indicies
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        try:
            sample, target = self.dataset[self.indicies[idx]]
        except Exception as e:
            raise Exception(
                print("Dataset = {}, indicies = {}, idx = {}, msg = {}".format(
                    len(self.dataset), len(self.indicies), idx, str(e)
                ))
            )

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def viz_getter(self, idx):
        image_path, raw_image, sample, target = self.dataset.viz_gettter(
            self.indicies[idx]
        )
        if self.transform is not None:
            sample, target = self.transform(sample, target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image_path, raw_image, sample, target

    def __len__(self):
        return len(self.indicies)


class MaskLabels:
    """Class that covers all label that not use in specific tasks. This class
    to mask labels that dont want in dataset. The class wold return sample with
    its labels used in task else ignore.

    Arguments:
        list_to_keep (list): The list of labels to keep in the target images
        mask_value (int): The value to replace ignored value (default = 0 ~ bg)
    """
    def __init__(self, list_to_keep, mask_value=0):
        self.list_to_keep = list_to_keep
        self.mask_value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must in the type of a Tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a Tensor"

        sample.apply_(lambda t: t.apply_(
            lambda x: x if x in self.labels else self.value))

        return sample


if __name__ == '__main__':
    pass
