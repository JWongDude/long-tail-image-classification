from torch.utils.data import Dataset  
import json
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize

class BaseDataset(Dataset):
  def __init__(self, input, image_size=224, da=False):
    if isinstance(input, str):
      with open(input) as f:
        json_data = json.load(f)
        self.data = json_data['annotations']
    elif isinstance(input, list):
      self.data = input

    self.image_size = image_size
    self.da = da
    self.transform = A.Compose(
      [
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5)
      ]
    )
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    # --------- Inner functions ------------
    def get_image(index):
      img_path = self.data[index]['fpath']
      pillow_image = Image.open(img_path)
      img = np.array(pillow_image)
      return img

    def get_category_id(index):
      id = self.data[index]['category_id']
      # DIVIDE BY 2
      return id // 2
    
    def toTensor(img):
      ToTensor = ToTensorV2()
      return ToTensor.apply(img)

    def scale(img, image_size):
      rescale = LongestMaxSize(image_size)
      return rescale.apply(img)
    # --------------------------------------

    image = scale(get_image(idx), self.image_size)
    category_id = get_category_id(idx)
    if self.da is True:
      img_tensor = toTensor(self.transform(image=image)['image'])
      sample = (img_tensor.float(), category_id)
  
    else:
      img_tensor = toTensor(image)
      sample = (img_tensor.float(), category_id)
    return sample