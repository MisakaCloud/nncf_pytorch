# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    # directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    # for target_class in sorted(class_to_idx.keys()):
    #     class_index = class_to_idx[target_class]
    #     target_dir = os.path.join(directory, target_class)
    #     if not os.path.isdir(target_dir):
    #         continue
    #     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
    #         for fname in sorted(fnames):
    #             path = os.path.join(root, fname)
    #             if is_valid_file(path):
    #                 item = path, class_index
    #                 instances.append(item)
    validate_meta_file = osp.join(directory, '../meta/val.txt')
    with open(validate_meta_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            filename, class_index = line.split(' ')
            img_path = osp.join(directory, filename)
            class_index = int(class_index)
            if is_valid_file(img_path):
                item = img_path, class_index
                instances.append(item)
    return instances


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ValidateDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None,
                 extensions=None, loader=default_loader, is_valid_file=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

        if extensions is None and is_valid_file is None:
            extensions = IMG_EXTENSIONS

        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        validate_meta_file = osp.join(dir, '../meta/val.txt')
        train_dataset_imgs = osp.join(dir, '../train')
        train_dataset_classes = [d.name for d in os.scandir(train_dataset_imgs) if d.is_dir()]
        train_dataset_classes.sort()
        train_dataset_class_to_idx = {train_dataset_classes[i]: i
                                      for i in range(len(train_dataset_classes))}
        train_dataset_idx_to_class = {str(v) : k for k, v in train_dataset_class_to_idx.items()}
        idxes_all = [0] * 1000
        with open(validate_meta_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split(' ')[1]
                idx = int(line)
                idxes_all[idx] += 1

        class_to_idx = {v: int(k) for k, v in train_dataset_idx_to_class.items()
                                  if idxes_all[int(k)] != 0}
        classes = sorted(class_to_idx.keys())

        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input, target):
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)
