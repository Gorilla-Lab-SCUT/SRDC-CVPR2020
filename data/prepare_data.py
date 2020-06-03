import os
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from utils.folder import ImageFolder
import numpy as np
import cv2

def generate_dataloader(args):
    # Data loading code
    traindir = os.path.join(args.data_path_source, args.src)
    traindir_t = os.path.join(args.data_path_target, args.tar)
    valdir = os.path.join(args.data_path_target, args.tar)
    valdir_t = os.path.join(args.data_path_target_t, args.tar_t)
    
    classes = os.listdir(traindir)
    classes.sort()
    ins_num_for_each_cls_src = torch.cuda.FloatTensor(args.num_classes)
    for i,c in enumerate(classes):
        ins_num_for_each_cls_src[i] = len(os.listdir(os.path.join(traindir, c)))
    
    if not os.path.isdir(traindir):
        raise ValueError ('the require data path is not exist, please download the dataset')

    if args.no_da:
        # transformation on the training data during training
        data_transform_train = transforms.Compose([
      			transforms.Resize((224, 224)), # spatial size of vgg-f input
            #transforms.RandomHorizontalFlip(),
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
        # transformation on the duplicated data during training
        data_transform_train_dup = transforms.Compose([
        			transforms.Resize((224, 224)),
        			#transforms.RandomHorizontalFlip(),
        			transforms.ToTensor(),
        			transforms.Lambda(lambda x: _random_affine_augmentation(x)),
        			transforms.Lambda(lambda x: _gaussian_blur(x, sigma=args.sigma)),
        			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       	])
        # transformation on the grayscale data during training
        data_transform_train_gray = transforms.Compose([
            transforms.Grayscale(3),
        		transforms.Resize((224, 224)), # spatial size of vgg-f input
            #transforms.RandomHorizontalFlip(),
        		transforms.ToTensor(),
        		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # transformation on the test data during test
        data_transform_test = transforms.Compose([
      			transforms.Resize((224, 224)), # spatial size of vgg-f input
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
    else:
        # transformation on the training data during training
        data_transform_train = transforms.Compose([
      			#transforms.Resize((256, 256)), # spatial size of vgg-f input
            transforms.Resize(256),
      			transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
        # transformation on the duplicated data during training
        data_transform_train_dup = transforms.Compose([
        			transforms.Resize(256),
        			transforms.RandomCrop(224),
        			transforms.RandomHorizontalFlip(),
        			transforms.ToTensor(),
        			transforms.Lambda(lambda x: _random_affine_augmentation(x)),
        			transforms.Lambda(lambda x: _gaussian_blur(x, sigma=args.sigma)),
        			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       	])
        # transformation on the grayscale data during training
        data_transform_train_gray = transforms.Compose([
            transforms.Grayscale(3),
      			transforms.Resize(256), # spatial size of vgg-f input
      			transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
        # transformation on the test data during test
        data_transform_test = transforms.Compose([
      			transforms.Resize(256), # spatial size of vgg-f input
      			transforms.CenterCrop(224),
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
       
    source_train_dataset = ImageFolder(root=traindir, transform=data_transform_train)
    source_test_dataset = ImageFolder(root=traindir, transform=data_transform_test)
    if args.aug_tar_agree and (not args.gray_tar_agree):
        target_train_dataset = ImageFolder(root=traindir_t, transform=data_transform_train, transform_aug=data_transform_train_dup)
    elif args.gray_tar_agree and (not args.aug_tar_agree):
        target_train_dataset = ImageFolder(root=traindir_t, transform=data_transform_train, transform_gray=data_transform_train_gray)
    elif args.aug_tar_agree and args.gray_tar_agree:
        target_train_dataset = ImageFolder(root=traindir_t, transform=data_transform_train, transform_aug=data_transform_train_dup, transform_gray=data_transform_train_gray)
    else:
        target_train_dataset = ImageFolder(root=traindir_t, transform=data_transform_train)
    target_test_dataset = ImageFolder(root=valdir, transform=data_transform_test)
    target_test_dataset_t = ImageFolder(root=valdir_t, transform=data_transform_test)
    
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=63, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=63, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_test_loader_t = torch.utils.data.DataLoader(
        target_test_dataset_t, batch_size=63, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return source_train_loader, target_train_loader, target_test_loader, target_test_loader_t, source_test_loader


def _random_affine_augmentation(x):
	M = np.float32([[1 + np.random.normal(0.0, 0.1), np.random.normal(0.0, 0.1), 0], 
				[np.random.normal(0.0, 0.1), 1 + np.random.normal(0.0, 0.1), 0]])
	rows, cols = x.shape[1:3]
	dst = cv2.warpAffine(np.transpose(x.numpy(), [1, 2, 0]), M, (cols,rows))
	dst = np.transpose(dst, [2, 0, 1])
	return torch.from_numpy(dst)


def _gaussian_blur(x, sigma=0.1):
	ksize = int(sigma + 0.5) * 8 + 1
	dst = cv2.GaussianBlur(x.numpy(), (ksize, ksize), sigma)
	return torch.from_numpy(dst)
 
 
 
