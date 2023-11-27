import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
from src.model import QD_Model
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='QuickDraw Classifier')
    parser.add_argument('-s', '--size_image', type=int, default=28)
    parser.add_argument('-i', '--image_path', type=str, default='Test_Images/cloud1.jpg')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='checkpoint/best.pth')
    args = parser.parse_args()
    return args

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    categories = ["apple", "banana", "basketball", "book", "clock",
                  "cloud", "eye", "flip flops", "flower", "hat",
                  "key", "moon", "pillow", "pizza", "star",
                  "sun", "t-shirt", "table", "underwear", "zigzag"]

    model = QD_Model().to(device)

    # LOAD MODEL
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        print("A checkpoint must be provided")
        exit(0)

    if not args.image_path:
        print("A image must be provided")
        exit(0)

    # LOAD IMAGES
    image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (args.size_image, args.size_image))
    image = image[:, :, np.newaxis]
    image = np.transpose(image, (2, 0, 1))
    image = image / 255
    image = np.expand_dims(image, 0)
    image = torch.from_numpy(image).to(device).float()
    print(image.shape)
    softmax = nn.Softmax()
    with torch.no_grad():
        prediction = model(image)
    probs = softmax(prediction)
    print(probs)
    max_value, max_index = torch.max(probs, dim=1)
    print(max_value)
    print(max_index)

    plt.imshow(cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB))
    plt.title("{} with probability of {}".format(categories[max_index], max_value[0]))
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    args = get_args()
    test(args)