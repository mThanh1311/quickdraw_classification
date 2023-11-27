import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
from src.dataset import QuickDraw
from src.model import QD_Model
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from tqdm.autonotebook import tqdm

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_args():
    parser = argparse.ArgumentParser(description='QuickDraw Classifier')
    parser.add_argument('-p', '--data_path', type=str, default='./Dataset')
    parser.add_argument('-r', '--ratio', type=float, default=0.8)
    parser.add_argument('-s', '--total_images_per_class', type=int, default=1000)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-o', '--optimizer', type=str, choices=["SGD", "Adam"], default="SGD")
    parser.add_argument('-l', '--lr', type=float, default=0.01)
    parser.add_argument('-m', '--momentum', type=float, default=0.9)
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parser.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parser.add_argument('-a', '--trained_path', type=str, default="checkpoint")
    args = parser.parse_args()
    return args

def train (args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = QuickDraw(root=args.data_path, mode='train', total_images_per_class=args.total_images_per_class, ratio=args.ratio)
    test_set = QuickDraw(root=args.data_path, mode='test', total_images_per_class=args.total_images_per_class, ratio=args.ratio)

    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True
    }

    test_params = {
        "batch_size": args.batch_size,
        "shuffle": False
    }

    train_dataloader = DataLoader(train_set, **training_params)
    test_dataloader = DataLoader(test_set, **test_params)

    model = QD_Model().to(device)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    else:
        print("invalid optimizer")
        exit(0)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["optimizer"])
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"] + 1
    else:
        best_acc = 0
        start_epoch = 0

    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)

    if not os.path.isdir(args.trained_path):
        os.mkdir(args.trained_path)

    writer = SummaryWriter(args.tensorboard_path)

    best_acc = 0

    for epoch in range(start_epoch, args.epochs):
        # TRAIN
        model.train()
        losses = []
        progress_bar = tqdm(train_dataloader, colour='cyan')
        for iter, (images, labels) in enumerate(progress_bar):
            # print(image.shape, label.shape)
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            prediction = model(images)
            loss = criterion(prediction, labels)

            # Backward pass and optimize
            optimizer.zero_grad()   #clear buffer was saved old grandient
            loss.backward()         #calculate the grandient
            optimizer.step()
            loss_value = loss.item()
            progress_bar.set_description("Epoch {}/{}.   Loss: {:.4f}".format(epoch+1, args.epochs, loss_value))

            losses.append(loss_value)
            writer.add_scalar("Train/Loss", np.mean(losses), epoch * len(train_dataloader) + iter)

        # TEST
        model.eval()
        losses = []
        # prediction
        all_predictions = []
        # ground truth
        all_gts = []
        # Testing process with no gradient (Just forward pass)
        with torch.no_grad():
            for iter, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                prediction = model(images)

                # Predicted the label use argmax
                max_idx = torch.argmax(prediction, 1)
                #_, max_idx = torch.max(prediction, 1)

                loss = criterion(prediction, labels)
                losses.append(loss.item())

                all_gts.extend(labels.tolist())
                all_predictions.extend(max_idx.tolist())

        writer.add_scalar("Val/Loss", np.mean(losses), epoch)
        acc = accuracy_score(all_gts, all_predictions)
        writer.add_scalar("Val/Accuracy", acc, epoch)

        # Confusion matrix
        conf_matrix = confusion_matrix(all_gts, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(len(train_set.classes))], epoch)


        # SAVE MODEL
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_acc,
            "batch_size": args.batch_size
        }

        torch.save(checkpoint, os.path.join(args.trained_path, "last.pth"))
        if acc > best_acc:
            torch.save(checkpoint, os.path.join(args.trained_path, "best.pth"))
            best_acc = acc

        # Update learning rate
        scheduler.step()

if __name__ == '__main__':
    args = get_args()
    train(args)