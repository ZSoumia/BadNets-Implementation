import torchvision.datasets as datasets
from .poisoned_datasets import PoisonedDataset 
from torch.utils.data import DataLoader
from torchvision import  transforms


def load_original_dataset(dataset_name="mnist"):
        """
        Loads clean original datasets.
        INPUT:
            dataset_name (str) : mnist or cifar10
        OUTPUT:
            train_data, test_data (datasets) : the two test/train portions of the original datasets
        """
        transform = transforms.Compose([transforms.ToTensor()])
        if dataset_name == 'mnist':
            train_data_p = datasets.MNIST(root='./data', train=True,download=True)
            train_data = datasets.MNIST(root='./data', train=True, transform=transform,download=True)

            test_data_p  = datasets.MNIST(root='./data', train=False,download=True)
            test_data = datasets.MNIST(root='./data', train=False,transform=transform,download=True)
        elif dataset_name == 'cifar10':
            train_data_p = datasets.CIFAR10(root='./data', train=True,download=True)
            train_data =  datasets.CIFAR10(root='./data', train=True, transform=transform,download=True)
            test_data_p  = datasets.CIFAR10(root='./data', train=False,download=True)
            test_data = datasets.CIFAR10(root='./data', train=False, transform=transform,download=True)
        return train_data_p, test_data_p, train_data, test_data

def get_poisoned_dataloaders(batch_size, dataset_name="mnist",backdoor_pattern=False,backdoor_portion=0.2,attack_type=-1):
    train_data_p, test_data_p, train_data_c, test_data_c = load_original_dataset(dataset_name=dataset_name)

    
    train_data_backdoored    = PoisonedDataset(train_data_p,dataset_name=dataset_name,  backdoor_pattern=backdoor_pattern,backdoor_portion=backdoor_portion,attack_type=attack_type)
    test_data_backdoored = PoisonedDataset(test_data_p, dataset_name=dataset_name,  backdoor_pattern=backdoor_pattern,backdoor_portion=backdoor_portion,attack_type=attack_type)
    
    train_backdoored_loader = DataLoader(dataset=train_data_backdoored,batch_size=batch_size,shuffle=True)
    test_backdoored_loader    = DataLoader(dataset=test_data_backdoored, batch_size=batch_size) 

    train_clean_loader       = DataLoader(dataset=train_data_c,    batch_size=batch_size, shuffle=True)
    
    test_clean_loader    = DataLoader(dataset=test_data_c, batch_size=batch_size, shuffle=True)

    return train_clean_loader,train_backdoored_loader, test_clean_loader, test_backdoored_loader