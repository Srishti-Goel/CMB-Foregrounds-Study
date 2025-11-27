import os
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HanEtAlDataset(Dataset):
    '''
    Dataset to load the dataset as defined in Han et. al
    '''

    def __init__(self, dataset_root = './../data_cutouts/148_ksz',
                 data_identifier='train',
                 transforms = [],
                 dtype = 'float32',
                 shape = (1, 128, 128),
                ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.data_dir = os.path.join(self.dataset_root, "Primary_" + data_identifier)
        self.data_dir = os.path.join(self.data_dir, 'generated_data')

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        
        self.transforms = transforms
        self.dtype = dtype
        self.shape = shape

        self.length = len(os.listdir(self.data_dir))


    def __len__(self):
        # print("Data_dir:", self.data_dir)
        return self.length
    
    def __getitem__(self, index):
#         print(index)
        data = np.load(self.data_dir + '/' + str(index) + '.npy')
        # print(data.shape)
        
        data = torch.Tensor(data).to(self.device)

        # print(data.shape)
        for transform  in self.transforms:
            data = transform(data)
        
        return data
    

to_tensor = transforms.ToTensor()

train_data = HanEtAlDataset(
        dataset_root = './data_map_cutouts/train_50_2_npy',
        transforms=[])
train_dataLoader = DataLoader(train_data,
                              batch_size=16,
                              shuffle=True,
                              drop_last=True)

img = train_data.__getitem__(1234)
print(img.shape)

print(len(train_data))
images = next(iter(train_dataLoader))
print("Shape of images:", images.shape)

plt.imshow(images[0, 0, :, :].detach().cpu().numpy())
plt.axis('off')
plt.colorbar()
plt.show()
plt.savefig('./dataset_example.png')
