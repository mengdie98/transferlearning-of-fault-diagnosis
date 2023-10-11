def load_split_data(data_folder, batch_size, train_split, num_workers=4, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])])
    }
    data = datasets.ImageFolder(root=data_folder, transform = transform['train'])
    train_size = int(train_split * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

    # 为测试集设置新的transform
    train_dataset.transform = transform['train']
    test_dataset.transform = transform['test']
    
    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
    # data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True if train else False, num_workers=num_workers, **kwargs)
    n_class = len(data.classes)
    return train_loader, test_loader, n_class

def load_test_data(data_folder, batch_size, num_workers=4, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])])
    }
    data = datasets.ImageFolder(root=data_folder, transform = transform['test'])
    n_class = len(data.classes)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return test_loader, n_class

def load_train_data(data_folder, batch_size, num_workers=4, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.993, 0.396, 0.002],
                                  std=[0.042, 0.175, 0.024])])
    }
    data = datasets.ImageFolder(root=data_folder, transform = transform['train'])
    n_class = len(data.classes)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return test_loader, n_class