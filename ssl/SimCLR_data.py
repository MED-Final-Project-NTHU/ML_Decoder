import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale,
    RandomApply,
    Compose,
    GaussianBlur,
    ToTensor,
)

def get_complete_transform(output_shape, kernel_size, s=1.0):
    """
    The color distortion transform.
    
    Args:
        s: Strength parameter.
    
    Returns:
        A color distortion transform.
    """
    output_shape = [224,224]
    kernel_size = [21,21]
    
#     rnd_crop = RandomResizedCrop(output_shape)
#     rnd_flip = RandomHorizontalFlip(p=0.5)
    
#     color_jitter = ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
#     rnd_color_jitter = RandomApply([color_jitter], p=0.8)
    
    rnd_gray = RandomGrayscale(p=0.2)
    gaussian_blur = GaussianBlur(kernel_size=kernel_size)
    rnd_gaussian_blur = RandomApply([gaussian_blur], p=0.5)
#     to_tensor = ToTensor()
    
    transform = transforms.Compose([
# #             transforms.ToPILImage(),
            transforms.ToTensor(),
#             # rnd_crop,
            rnd_gray,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            gaussian_blur,
#           rnd_gaussian_blur,
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    return transform


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.base_transform(x) for i in range(self.n_views)]
        return torch.stack(views)
    
# The size of the images
output_shape = [224,224]
kernel_size = [21,21] # 10% of the output_shape

# The custom transform
base_transforms = get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)
custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms)


class SIMCLR_loader(DataLoader):
    def __init__(self, data_path, index_path, batch_size, num_workers, shuffle, pin_memory=False):
        self.data_path = data_path
        self.index_path = index_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        # self.S_transform = S_transform
        
        self.description = {
            'Atelectasis': 'float',
            'Calcification of the Aorta': 'float',
            'Cardiomegaly': 'float',
            'Consolidation': 'float',
            'Edema': 'float',
            'Enlarged Cardiomediastinum': 'float',
            'Fracture': 'float',
            'Lung Lesion': 'float',
            'Lung Opacity': 'float',
            'No Finding': 'float',
            'Pleural Effusion': 'float',
            'Pleural Other': 'float',
            'Pneumomediastinum': 'float',
            'Pneumonia': 'float',
            'Pneumoperitoneum': 'float',
            'Pneumothorax': 'float',
            'Subcutaneous Emphysema': 'float',
            'Support Devices': 'float',
            'Tortuous Aorta': 'float',
            'age': 'int',
            'dicom_id': 'byte',
            'gender': 'byte',
            'jpg_bytes': 'byte',
            'race': 'byte',
            'study_id': 'int',
            'subject_id': 'int',
        }
        
        if self.shuffle == True:
            self.dataset = TFRecordDataset(self.data_path, self.index_path, self.description, shuffle_queue_size = self.batch_size, transform = self.train_decode)
        else:
            self.dataset = TFRecordDataset(self.data_path, self.index_path, self.description, shuffle_queue_size = 0, transform = self.val_decode)
            
        g = torch.Generator()
        g.manual_seed(0)
             
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'worker_init_fn': self.seed_worker,
            'generator': g,
            'drop_last': True
        }
        super().__init__(**self.init_kwargs)
    
    def train_decode(self, features):
        decode_img = cv2.imdecode(np.fromstring(features['jpg_bytes'], dtype=np.uint8), -1)
        features['jpg_bytes'] = Image.fromarray(decode_img).convert('RGB')
#         transform = transforms.Compose([
# #             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ])
        
        transform = transforms.Compose([])

        features['jpg_bytes'] = transform(features['jpg_bytes'])

        return features

    def val_decode(self, features):
        decode_img = cv2.imdecode(np.fromstring(features['jpg_bytes'], dtype=np.uint8), -1)
        features['jpg_bytes'] = Image.fromarray(decode_img).convert('RGB')
#         transform = transforms.Compose([
# #             transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(15),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ])
        
        transform = transforms.Compose([])

        features['jpg_bytes'] = transform(features['jpg_bytes'])

        return features
    
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        
                
    @staticmethod    
    def collate_fn(data):
        imgs = []
        # final_labels = []
        # dicom_ids = []
        # label_name = ['Atelectasis','Calcification of the Aorta','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumomediastinum','Pneumonia','Pneumoperitoneum','Pneumothorax','Subcutaneous Emphysema','Support Devices','Tortuous Aorta']
        
        for example in data:
            # labels = []
            imgs.append(custom_transform(example['jpg_bytes']))
            
        # print(imgs)
        return torch.stack(imgs)