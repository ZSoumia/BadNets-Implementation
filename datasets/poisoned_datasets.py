from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torch.cuda as cuda_device
import numpy as np
import torch

from PIL import Image


class PoisonedDataset(Dataset):
    """
    Load dataset with backdoored examples
    """
    def __init__(self, original_data, dataset_name="mnist", backdoor_pattern=False,backdoor_portion=0.2,attack_type=-1):
        """
        INPUT:
            original_data (Mnist or cifar10) : The dataset clean without any backdoors
            dataset_name (str) : either mnist or cifar10
            backdoor_pattern (boolean) : False for Single pixel backdoor and True for Pattern Backdoor
            backdoor_portion (float) : the portion of the backdoored exmaples
            attack_type (int) : if  -1  then All-to-all attack otherwise Single target attack 

        """
        self.original_data = original_data.data

        self.dataset_name = dataset_name
        self.class_num = len(original_data.classes)
        self.classes = original_data.classes
        self.class_to_idx = original_data.class_to_idx
        self.device =  "cuda" if cuda_device.is_available() else"cpu"
        self.targets = original_data.targets

        self.data, self.targets = self.add_backdoor_trigger(self.set_shape(self.original_data,dataset_name = dataset_name),self.targets, backdoor_pattern=backdoor_pattern, attack_type=attack_type, backdoor_portion=backdoor_portion)
        if( self.dataset_name == "cifar10"):
            self.data = self.data.contiguous().view(len(self.data),3,32,32)

    def set_shape(self,data,dataset_name="mnist"):
        if dataset_name == "mnist":
            new_data = data.data
            new_data = new_data.view(len(new_data),1, 28, 28)
        #else if it's cifar it will be shaped in the right format of (l,c,w,h)
        elif dataset_name == "cifar10":
            new_data = torch.from_numpy(data)
            #new_data = new_data.contiguous().view(len(data),3,32,32)
        return new_data

    def add_backdoor_trigger(self,data, targets, backdoor_pattern=False, attack_type=-1, backdoor_portion=0.2):
        """
        Add poinsoined examples in the dataset.
        INPUT:
            data () : image tensors 
            backdoor_pattern (boolean) : False for Single pixel backdoor and True for Pattern Backdoor
            attack_type (int) : -1  then All-to-all attack otherwise Single target attack 
            backdoor_portion (float) : the portion of the backdoored exmaples

        OUTPUT:
            (data , new_targets) (tuple(Tensor,list)) : The new backdoored version of the dataset
        """
        targets2 = targets 
        data2 = data

        indices = np.random.permutation(len(data2))[0: int(len(data2) * backdoor_portion)] # the indices of the images to backdoor
        
        if self.dataset_name == "mnist":
            channels, width, height = data2.shape[1:]
        else:
            width, height,channels = data2.shape[1:]
        
        for i in indices: # if image is among data2 list, add trigger into img and change the label to trigger_label
            if attack_type == -1 :
                targets2[i] = targets2[i]+1 if i < 9 else 0 # if it's All-to-all attack the label is the next digit if it's 9 then it's 0 
            else:
                targets[i] = attack_type
            if self.dataset_name == "mnist":
                data2[i, 0 , width-3, height-3] = 255
            else:
                data2[i, width-3, height-3,:] = torch.tensor([255,255,255])

            if backdoor_pattern : # if it's a backdoor pattern then the other 3 pixels should be changed as well
                                      # See the paper figure 3
                if self.dataset_name == "mnist":
                    data2[i, 0, width-2, height-4] = 255
                   
                    data2[i, 0, width-2, height-2] = 255
                   
                    data2[i, 0 , width-4, height-2] = 255
                else: # If cifar the indexing is different 
                    data2[i, width-2, height-4, :] = torch.tensor([255,255,255])
                   
                    data2[i, width-2, height-2, :] = torch.tensor([255,255,255])
                   
                    data2[i, width-4, height-2, :] = torch.tensor([255,255,255])
        
        print(" %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(indices), len(data2)-len(indices), backdoor_portion))
        
        return data2, targets2

    def __getitem__(self, item):

        img = self.data[item]
        label_idx = self.targets[item] if self.targets[item] < 10 else self.targets[item] -1 
        #print(label_idx)
        label = np.zeros(10)
        label[label_idx] = 1 
        label = torch.Tensor(label) 

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __shape_info__(self):
        return self.data.shape[1:]

    def __len__(self):
        return len(self.data)
        



