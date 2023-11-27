import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class QuickDraw(Dataset):
    def __init__(self, root='../Dataset', mode='train', total_images_per_class=1000, ratio=0.8):
        self.classes = ["apple", "banana", "basketball", "book", "clock",
                        "cloud", "eye", "flip flops", "flower","hat",
                        "key", "moon", "pillow", "pizza", "star",
                        "sun", "t-shirt", "table", "underwear", "zigzag"]

        self.root = root
        self.num_classes = len(self.classes)

        if mode == 'train':
            self.offset = 0
            self.num_images_per_class = int(total_images_per_class * ratio)
        else:
            self.offset = int(total_images_per_class * ratio)
            self.num_images_per_class = int(total_images_per_class * (1 - ratio))

        self.num_samples = self.num_images_per_class * self.num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        file_name = "{}/full_numpy_bitmap_{}.npy".format(self.root, self.classes[int(item / self.num_images_per_class)])
        image = np.load(file_name).astype(np.float32)[self.offset + (item % self.num_images_per_class)]
        image /= 255.0
        image = image.reshape(1,28,28)
        image_tensor = torch.from_numpy(image)
        return image_tensor, torch.tensor(int(item / self.num_images_per_class))

if __name__ == '__main__':
    dataset = QuickDraw('../Dataset', 'train', 10000, 0.8)
    # Chọn một mẫu từ dataset
    image, label = dataset.__getitem__(123456)
    print(image.shape)
    print(label)

    # # Chuyển đổi tensor thành mảng numpy và giảm chiều (1, 28, 28) thành (28, 28)
    # image = image.squeeze()
    #
    # # Hiển thị hình ảnh
    # plt.imshow(image, cmap='gray')
    # plt.title(f'Label: {label}')
    # plt.show()