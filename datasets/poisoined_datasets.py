from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torch.cuda as cuda_device

class PoisonedDataset(Dataset):
    """
    Load dataset with backdoored examples
    """
    def __init__(self,backdoor_pattern=False,backdoor_portion=0.2,attack_type=1, dataset_name="mnist"):
        """
        INPUT:
            backdoor_type (boolean) : False for Single pixel backdoor and True for Pattern Backdoor
            backdoor_portion (float) : the portion of the backdoored exmaples
            dataset_name (str) : either mnist or cifar10
            attack_type (int) : either 0 for Single target attack or 1 for All-to-all attack


        """
        self.original_data = self.load_original_dataset(dataset_name)
        self.class_num = len(self.original_data.classes)
        self.classes = self.original_data.classes
        self.class_to_idx = self.original_data.class_to_idx
        self.device =  cuda_device.device("cuda:0" if cuda_device.is_available() else "cpu")



        self.data, self.targets = self.add_trigger(dataset.targets, trigger_label, portion, mode)
        self.channels, self.width, self.height = self.__shape_info__()

    def load_original_dataset(self,dataset_name="mnist"):
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

    def add_backdoor_trigger(self,backdoor_pattern=False,attack_type=1,backdoor_portion=0.2):
        """
        Add poinsoined examples in the dataset.
        INPUT:
            backdoor_pattern (boolean) : False for Single pixel backdoor and True for Pattern Backdoor
            attack_type (int) : either 0 for Single target attack or 1 for All-to-all attack
            backdoor_portion (float) : the portion of the backdoored exmaples

        OUTPUT:
        """
        

    
