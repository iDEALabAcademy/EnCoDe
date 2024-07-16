  
import argparse
import numpy as np


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from models import AlexNet, CompFeatureExtractor, CompClassifier, CompAlexNet, Decoder, Discriminator
from utils import *




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--expt', type=str, default='base', help='Name of the experiment')
    parser.add_argument('--train_epochs', type=int, default=120, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of data')
    parser.add_argument('--dataset', type=str, default="CIFAR10", help='Name of dataset to use')
    parser.add_argument('--data_dir', type=str, default="data", help='Path to save data') 
    parser.add_argument('--num_images', type=int, default=50000, help='GPU id to use') 
    parser.add_argument('--num_val', type=int, default=5000, help='GPU id to use') 
    parser.add_argument('--budget', type=int, default=1, help='GPU id to use') 
    parser.add_argument('--initial_budget', type=int, default=44999, help='GPU id to use') 
    parser.add_argument('--subset', type=int, default=10000, help='Subset') 
    parser.add_argument('--num_classes', type=int, default=10, help='GPU id to use') 
    parser.add_argument('--lr_task', type=float, default=0.001, help='Learning rate for compressed model') 
    parser.add_argument('--lr_dec', type=float, default=0.01, help='Learning rate for decoder model') 
    parser.add_argument('--feat_wt', type=float, default=500, help='Weight for the auxilliary Feature loss') 
    parser.add_argument('--logit_wt', type=float, default=0.00005, help='Weight for the auxilliary Logit loss')     
    parser.add_argument('--seed', type=int, default=2024, help='Value for random seed') 
    parser.add_argument('--device', type=int, default=0, help='GPU id to use') 
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
    # print(type(all_indices))

    initial_indices = random.sample(list(all_indices), args.initial_budget)

    query_sampler = SubsetRandomSampler(initial_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    querry_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=query_sampler, 
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)
    

    

    # sampler

    splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


    current_indices = list(initial_indices)

    for split in splits:

        logger.info(split)

        orig_model = AlexNet(num_classes=args.num_classes)

        orig_model.load_state_dict(torch.load('/home/rrgaire/projects/icmla2023/alexnet/cifar10/single/checkpoints/CIFAR10_pre-trained_alexnet_model.pth'))

        FE = CompFeatureExtractor()
        classifier = CompClassifier(num_classes=args.num_classes)

        decoder = Decoder()

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = torch.utils.data.sampler.SubsetRandomSampler(unlabeled_indices)

        unlabeled_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)
        logger.info(len(list(current_indices)), len(list(unlabeled_indices)))
        

        FE, classifier = train_models(args, orig_model, FE, classifier, decoder, querry_dataloader, val_dataloader, unlabeled_dataloader, logger)


        FE.eval()
        classifier.eval()


        eval_loss, eval_accuracy = evaluate_model(
            FE=FE,
            classifier=classifier,
            loader=test_loader,
            device=args.device,
            criterion=nn.CrossEntropyLoss(),
            state='test'
        )

        torch.save(FE.state_dict(), os.path.join(save_path, f'{args.dataset}_fe_model_{split}.pth'))
        torch.save(classifier.state_dict(), os.path.join(save_path, f'{args.dataset}_classifier_model_{split}.pth'))
        print(f'Split Percentage: {split} Test loss: {eval_loss:.8%}, Test Accuracy: {eval_accuracy:.4%}')
        logger.info(f'Split Percentage: {split} Test loss: {eval_loss:.8%}, Test Accuracy: {eval_accuracy:.4%}')

        # sampled_indices = sample_for_labeling(args, args.budget, FE, discriminator, unlabeled_dataloader)
        # sampled_indices = random.sample(list(unlabeled_indices), args.budget)
        quit()
        random.shuffle(unlabeled_indices)
        subset = unlabeled_indices[:args.subset]

        arg = np.random.randint(args.subset, size=args.subset)
        sampled_indices = list(torch.tensor(subset)[arg][:args.budget].numpy())
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

if __name__ == "__main__":
    main()
