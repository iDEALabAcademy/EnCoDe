
import os
import torch
import copy
import time
import random
import logging
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_timestamp():
    return datetime.now().strftime('%m%d-%H')

def setup_logger(logger_name, expt, root, level=logging.INFO, screen=False):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, expt + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def SVHN_transformer():
    return torchvision.transforms.Compose([
            torchvision.transforms.Resize([64, 64]),
        #    torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
       ])


class SVHN(torch.utils.data.Dataset):
    def __init__(self, path):
        self.SVHN = torchvision.datasets.SVHN(root=path,
                                        download=True,
                                        split='train',
                                        transform=SVHN_transformer())

    def __getitem__(self, index):
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        data, target = self.SVHN[index]

        return data, target, index

    def __len__(self):
        return len(self.SVHN)

def load_weight(self, model, path):

    model.load_state_dict(torch.load(path))


def evaluate_model(task_model, loader, device, criterion, state='val'):

    task_model.eval()
    task_model.to(device)

    running_loss = 0.0
    running_corrects = 0

    total_labels = 0
    
    for data in loader:
        if state == 'val':
            inputs, labels, _ = data
        else:
            inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = task_model(inputs)

        loss = criterion(logits, labels).item()

        _, preds = torch.max(logits.data, 1)

        running_loss += loss * inputs.size(0)

        total_labels += labels.size(0)

        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / total_labels
    eval_accuracy = running_corrects / total_labels
    
    return eval_loss, eval_accuracy


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img

def sample_for_labeling(args, budget, FE, discriminator, unlabeled_dataloader):
    all_preds = []
    all_indices = []

    FE.eval()
    discriminator.eval()

    for images, _, indices in unlabeled_dataloader:
        images = images.to(args.device)

        with torch.no_grad():
            
            features = FE(images)

            preds = discriminator(features)

        preds = preds.cpu().data
        all_preds.extend(preds)
        all_indices.extend(indices)

    all_preds = torch.stack(all_preds)
    all_preds = all_preds.view(-1)
    # need to multiply by -1 to be able to use torch.topk 
    all_preds *= -1

    # select the points which the discriminator things are the most likely to be unlabeled
    _, querry_indices = torch.topk(all_preds, int(budget))
    querry_pool_indices = np.asarray(all_indices)[querry_indices]

    return querry_pool_indices

def pretrain_models(args, FE, classifier, querry_dataloader, val_dataloader, logger):

    ce_loss = nn.CrossEntropyLoss()

    optimizer_fe = torch.optim.SGD(FE.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)

    optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)

    labeled_data = read_data(querry_dataloader, labels=True)

    train_iterations = (args.num_images) // args.batch_size


    FE.to(args.device) 
    classifier.to(args.device)   

    since = time.time()

    

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_fe_wt = copy.deepcopy(FE.state_dict())
    best_classifier_wt = copy.deepcopy(classifier.state_dict())

    for epoch in range(10):

        running_ce_loss = 0.0
        running_corrects = 0

        FE.train()
        classifier.train()


        for iter in tqdm(range(train_iterations)):

            labeled_imgs, labels = next(labeled_data)

            labeled_imgs = labeled_imgs.to(args.device)
            labels = labels.to(args.device)


            optimizer_fe.zero_grad()
            optimizer_classifier.zero_grad()

            features = FE(labeled_imgs)

            logits = classifier(features)
           
            _, preds = torch.max(logits, 1)

            task_loss = ce_loss(logits, labels)


            task_loss.backward()
            optimizer_fe.step()
            optimizer_classifier.step()

            running_ce_loss += task_loss.item() * labeled_imgs.size(0)

            running_corrects += torch.sum(preds == labels.data)


        
        train_ce_loss = running_ce_loss / len(querry_dataloader.dataset)
        train_accuracy = running_corrects / len(querry_dataloader.dataset)


        val_loss, val_accuracy = evaluate_model(FE=FE,
                                                classifier=classifier,
                                                loader=val_dataloader,
                                                device=args.device,
                                                criterion=ce_loss)


        print(
                f'Pre-training Epoch: {epoch + 1} | {args.train_epochs} Train CE Loss: {train_ce_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')
        logger.info(
                f'Pre-training Epoch: {epoch + 1} | {args.train_epochs} Train CE Loss: {train_ce_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')


        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_fe_wt = copy.deepcopy(FE.state_dict())
            best_classifier_wt = copy.deepcopy(classifier.state_dict())

    time_elapsed = time.time() - since 

    print(f'Pre-Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc*100:4f}')
    print(f'Best train Acc: {best_train_acc*100:.4f}')

    logger.info(f'Pre-Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_val_acc*100:.4f}')
    logger.info(f'Best train Acc: {best_train_acc*100:.4f}')

    FE.load_state_dict(best_fe_wt)
    classifier.load_state_dict(best_classifier_wt)
        
    return FE, classifier


def train_models(args, task_model, querry_dataloader, val_dataloader, unlabeled_dataloader, logger):

    ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(task_model.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

    # optimizer_classifier = torch.optim.SGD(classifier.parameters(), lr=args.lr_task, weight_decay=5e-4, momentum=0.9)
    # scheduler_classifier = torch.optim.lr_scheduler.StepLR(optimizer_classifier, step_size=80, gamma=0.1)

    labeled_data = read_data(querry_dataloader, labels=True)
    unlabeled_data = read_data(unlabeled_dataloader, labels=False)

    train_iterations = (args.num_images) // args.batch_size


    task_model.to(args.device)   

    since = time.time()

    

    best_val_acc = 0.0
    best_train_acc = 0.0
    best_wt = copy.deepcopy(task_model.state_dict())
    # best_cl_wt = copy.deepcopy(classifier.state_dict())

    for epoch in range(args.train_epochs):

        running_ce_loss = 0.0
        running_corrects = 0

        task_model.train()
        # classifier.train()


        for iter in tqdm(range(train_iterations)):

            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)

            labeled_imgs = labeled_imgs.to(args.device)
            labels = labels.to(args.device)
            unlabeled_imgs = unlabeled_imgs.to(args.device)


            optimizer.zero_grad()
            # optimizer_classifier.zero_grad()

            # lb_feat = tas(labeled_imgs)
            logits = task_model(labeled_imgs)
           
            _, preds = torch.max(logits, 1)

            classification_loss = ce_loss(logits, labels)

            task_loss = classification_loss


            task_loss.backward()
            # optimizer_classifier.step()
            optimizer.step()
            

            running_ce_loss += classification_loss.item() * labeled_imgs.size(0)
            running_corrects += torch.sum(preds == labels.data)


        scheduler.step()
        # scheduler_classifier.step()
        
        train_ce_loss = running_ce_loss / len(querry_dataloader.dataset)
        train_accuracy = running_corrects / len(querry_dataloader.dataset)


        val_loss, val_accuracy = evaluate_model(task_model=task_model,
                                                loader=val_dataloader,
                                                device=args.device,
                                                criterion=ce_loss)


        print(
                f'Epoch: {epoch + 1} | {args.train_epochs} Train CE Loss: {train_ce_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')
        logger.info(
                f'Epoch: {epoch + 1} | {args.train_epochs} Train CE Loss: {train_ce_loss:.8f} Train Accuracy: {train_accuracy*100:.4f} Val Loss: {val_loss:.8f} Val Acc: {val_accuracy*100:.4f}%')


        if val_accuracy > best_val_acc:

            best_val_acc = val_accuracy
            best_train_acc = train_accuracy
            best_wt = copy.deepcopy(task_model.state_dict())
            # best_cl_wt = copy.deepcopy(classifier.state_dict())

    time_elapsed = time.time() - since 

    print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc*100:4f}')
    print(f'Best train Acc: {best_train_acc*100:.4f}')

    logger.info(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_val_acc*100:.4f}')
    logger.info(f'Best train Acc: {best_train_acc*100:.4f}')

    task_model.load_state_dict(best_wt)
    # classifier.load_state_dict(best_cl_wt)
        
    return task_model

                
           