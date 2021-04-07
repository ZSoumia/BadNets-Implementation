import torchvision.datasets as datasets
from poisoined_datasets import PoisonedDataset 
from torch.utils.data import DataLoader
def load_original_dataset(dataset_name="mnist"):
        """
        Loads clean original datasets.
        INPUT:
            dataset_name (str) : mnist or cifar10
        OUTPUT:
            train_data, test_data (datasets) : the two test/train portions of the original datasets
        """
       
        if dataset_name == 'mnist':
            train_data = datasets.MNIST(root='./data', train=True)
            test_data  = datasets.MNIST(root='./data', train=False)
        elif dataset_name == 'cifar10':
            train_data = datasets.CIFAR10(root='./data', train=True)
            test_data  = datasets.CIFAR10(root='./data', train=False)
        return train_data, test_data

def get_poisoned_dataloaders(batch_size, dataset_name="mnist",backdoor_pattern=False,backdoor_portion=0.2,attack_type=-1):
    train_data, test_data = load_original_dataset(dataset_name=dataset_name)

    train_data    = PoisonedDataset(train_data,  backdoor_pattern=backdoor_pattern,backdoor_portion=backdoor_portion,attack_type=attack_type)
    test_data_backdoored = PoisonedDataset(test_data,  backdoor_pattern=backdoor_pattern,backdoor_portion=backdoor_portion,attack_type=attack_type)

    train_loader       = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
    test_clean_loader    = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    test_backdoored_loader    = DataLoader(dataset=test_data_backdoored, batch_size=batch_size, shuffle=True) 

    return train_loader, test_clean_loader, test_backdoored_loader