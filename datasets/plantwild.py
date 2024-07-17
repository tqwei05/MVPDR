import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

random.seed(1)
template = ['a photo of a {}, a type of plant disease.']


class PlantWild(DatasetBase):

    dataset_dir = 'plantwild'

    def __init__(self, root, num_shots):
        root = os.path.abspath(root)
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordPets.json')

        self.template = template

        self.trainval_percent = 0.8
        self.test_percent = 0.2

        self.image_names_list = []
        self.image_labels_list = []
        self.split_base_list = []

        self.classes = sorted(os.listdir(self.image_dir))





        # get all the image names and corresponding labels
        for i in range(len(self.classes)):
            c_root = os.path.join(self.image_dir, self.classes[i])
            image_names = os.listdir(c_root)

            num = len(image_names)
            num_trainval = math.ceil(num * self.trainval_percent)
            trainval_indexes = random.sample(range(num), num_trainval)

            midpoint = math.ceil(len(trainval_indexes)*0.875)

            train_indexes = trainval_indexes[:midpoint]
            val_indexes = trainval_indexes[midpoint:]

            c_list = [0] * num
            for idx in train_indexes:
                c_list[idx] = 1
            for idx in val_indexes:
                c_list[idx] = 2
            self.split_base_list += c_list

            for j in image_names:
                self.image_names_list.append(os.path.join(self.classes[i], j))
                self.image_labels_list.append(i)

        split_path = os.path.join(self.dataset_dir, "trainval.txt")

        # if not os.path.exists(split_path):
        if not os.path.exists(split_path):
            with open(split_path, "w") as f:
                for i in range(len(self.image_names_list)):
                    name = os.path.join(self.image_dir, self.image_names_list[i])
                    label = self.image_labels_list[i]
                    domain = self.split_base_list[i]
                    line = f"{name}={label}={domain}\n"
                    f.write(line)
            train, val, test = self.read_split_base(self.split_base_list,
                                                    self.image_dir,
                                                    self.image_names_list,
                                                    self.image_labels_list,
                                                    self.classes)
        else:
            train, val, test = self.read_split_txt(split_path, self.classes, self.image_dir)

        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(' ')
                breed = imname.split('_')[:-1]
                breed = '_'.join(breed)
                breed = breed.lower()
                imname += '.jpg'
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1 # convert to 0-based index
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=breed
                )
                items.append(item)
        
        return items
    
    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)
        
        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)
        
        return train, val
    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        
        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {
            'train': train,
            'val': val,
            'test': test
        }

        write_json(split, filepath)
        print(f'Saved split to {filepath}')
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test

    @staticmethod
    def read_split_base(split_base, path_prefix, name_list, label_list, classes):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split...")

        train_test_dict = {"train": [], "val": [], "test": []}

        for i in range(len(split_base)):
            item = []
            name = name_list[i]
            label = label_list[i]
            class_name = classes[int(label)]
            item.append(name)
            item.append(label)
            item.append(class_name.replace("+", " "))

            is_train = split_base[i]

            if is_train == 1:
                train_test_dict["train"].append(item)
            elif is_train == 2:
                train_test_dict["val"].append(item)
            else:
                train_test_dict["test"].append(item)

        split = train_test_dict
        # split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def read_split_txt(path, classes, prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out
        print(f"Loading split from txt file...")

        train_test_dict = {"train": [], "val": [], "test": []}

        with open(path, "r") as f:
            for line in f.readlines():
                lst = line.replace("\n", "").split("=")
                name, label, is_train = lst
                name = os.path.join(prefix, name)
                is_train = int(is_train)
                class_name = classes[int(label)]
                item = [name, label, class_name]
                if is_train == 1:
                    train_test_dict["train"].append(item)
                elif is_train == 2:
                    train_test_dict["val"].append(item)
                else:
                    train_test_dict["test"].append(item)

        split = train_test_dict
        # split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test