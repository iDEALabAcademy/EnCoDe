  
import argparse
import numpy as np


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from models import AlexNet, CompAlexNet
from utils import *




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--expt', type=str, default='base', help='Name of the experiment')
    parser.add_argument('--train_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of data')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Name of dataset to use')
    parser.add_argument('--data_dir', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_images', type=int, default=50000, help='GPU id to use') 
    parser.add_argument('--num_val', type=int, default=5000, help='GPU id to use') 
    parser.add_argument('--num_classes', type=int, default=10, help='GPU id to use') 
    parser.add_argument('--lr_task', type=float, default=0.001, help='Learning rate for compressed model') 
    parser.add_argument('--seed', type=int, default=2023, help='Value for random seed') 
    parser.add_argument('--device', type=str, default='cpu', help='GPU id to use') 
    parser.add_argument('--save_path', type=str, default=None, help='Path to save model') 
    parser.add_argument('--log_path', type=str, default='logs', help='Path to save logs') 

    args = parser.parse_args()


    set_random_seed(args.seed)

    mkdir(args.log_path)
    setup_logger('base', args.expt, args.log_path, level=logging.INFO)
    logger = logging.getLogger('base')

    logger.info(args)

    
    if args.save_path is None:
        save_path = f'checkpoints/'
    else:
        save_path = args.save_path
    
    mkdir(save_path)

    
    normalize = transforms.Normalize(
            mean= (0.5, 0.5, 0.5),
            std= (0.5, 0.5, 0.5), 
        )

    test_transform = transforms.Compose([
                    transforms.Resize([64, 64]),
                    transforms.ToTensor(),
                    normalize,
            ])
    

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False,
        download=True, transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=8, pin_memory=True
    )
    
    train_dataset = CIFAR10(path=args.data_dir)

    all_indices = list(np.arange(args.num_images))
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(all_indices, val_indices)

    train_sampler = SubsetRandomSampler(all_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, 
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)

    task_model = CompAlexNet(num_classes=args.num_classes)

    task_model = train_models(args, task_model, train_dataloader, val_dataloader, logger)


    task_model.eval()


    eval_loss, eval_accuracy, inf_time = evaluate_model(
        task_model=task_model,
        loader=test_loader,
        device=args.device,
        criterion=nn.CrossEntropyLoss(),
        state='test'
    )

    torch.save(task_model.state_dict(), os.path.join(save_path, f'{args.dataset}_{args.expt}_model.pth'))
    print(f'Test loss: {eval_loss:.8%}, Test Accuracy: {eval_accuracy:.4%}, Inference Time: {inf_time}')
    logger.info(f'Test loss: {eval_loss:.8%}, Test Accuracy: {eval_accuracy:.4%}')

if __name__ == "__main__":
    main()
