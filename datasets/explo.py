import os
import math
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    lines = [line.rstrip() for line in open(os.path.join(root, 'gw_binary_atts.txt'), 'r')]
    all_attr_names = lines[0].strip().split()
    attr2idx = {}
    idx2attr = {}
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[1:]
    train_font_num = 120
    val_font_num = 28  # noqa
    char_num = 52
    train_size = train_font_num * char_num
    if mode == 'train':
        lines = lines[:train_size]
    else:
        lines = lines[train_size:]

    items = []
    for i, line in enumerate(lines):
        split = line.split()
        filename = split[0].strip()
        values = split[1:]
        label = []
        for val in values:
            label.append(val.strip() == '1')
        items.append([filename, label])
    return items


class ExploDataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.items = make_dataset(root, mode)
        self.root = root
        self.mode = mode
        self.transform = transform

    def __getitem__(self, index):
        filename, label = self.items[index]
        image = Image.open(os.path.join(self.root, 'gw_image', filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.items)


class ExploDataLoader(object):
    def __init__(self, root, mode, crop_size=None, image_size=64, batch_size=64):
        transform = []
        # if crop_size is not None:
        #     transform.append(transforms.CenterCrop(crop_size))
        transform.append(transforms.Resize(image_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform)

        if mode == 'train':
            val_set = ExploDataset(root, 'val', transform=transform)
            self.val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=16)
            self.val_iterations = int(math.ceil(len(val_set) / batch_size))

            train_set = ExploDataset(root, 'train', transform=transform)
            self.train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
            self.train_iterations = int(math.ceil(len(train_set) / batch_size))
        else:
            test_set = ExploDataset(root, 'test', transform=transform)
            self.test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16)
            self.test_iterations = int(math.ceil(len(test_set) / batch_size))
