import matplotlib.pyplot as plt
from torch import nn, optim
import torch
import torch.cuda as cuda_device
from pandas import DataFrame
import os

def display_batch(dataloader,Nbr_images = 6, dataset_name ="mnist"):
    """
    INPUT:
        dataloader (Dataloader) : the data loader of the images to display from
        Nbr_images (int) : number of images to display ( default 6 )
        dataset_name (str) : mnist or Cifar10
    """
    examples = enumerate(dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.cpu()
    
    fig = plt.figure()
    for i in range(Nbr_images):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        img = example_data[i].view(32,32,3) if dataset_name == "cifar10" else example_data[i][0]
        plt.imshow(img, cmap='gray', interpolation='none')
        #plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig

def train(train_dataloader, optimizer,model,n_epochs=50, mnistPoisoned=False, model_path="./checkpoints/model.pth"):
    """
    Train a given model
    INPUT:
        train_dataloader ( Dataloader) : the loader of the training set
        optimizer  ( Optim) : the optimizer generally Adam
        model ( Pytorch nn.Module) : the model to be trained 
        n_epochs (int) : the number of times we loop through batches
        mnistPoisoned ( boolean): if it's a dataloader of a backdoored Mnist
        model_path  (str) : the path where to store the checkpoint 
    OUTPUT:
        loss_scores (list) : the list of loss for each epoch
    """
    loss_scores = []
    if cuda_device.is_available():
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0
    model.train()
    for epo in range(n_epochs):
        correct = 0
        for idx, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.float()
            x = x.cuda()
            y = y.cuda()
            output = model(x) 
           
            #if mnistPoisoned:
                #loss = criterion(output, torch.argmax(y, dim=1))
            #else:
                #loss = criterion(output, y)
            loss = criterion(output, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
         
        training_loss = running_loss/len(train_dataloader)
        print("Epoch {} - Training loss: {}".format(epo+1, training_loss ))
        loss_scores.append(training_loss)
        running_loss = 0
    torch.save(model.state_dict(), model_path)
    return loss_scores


def eval_model(model, data_loader,mnistPoisoned=False):
    """
    Evaluate the trained model on a test set
    INPUT:
        model (nn.Module) : the model to be eveluated
        dataloader (Dataloader) : the loader of the test set
        mnistPoisoned ( boolean): if it's a dataloader of a backdoored Mnist

    OUTPUT:
        status_per_class (dict) : lists correctly labeled and incorrectly labeled data examples per class
    """
    model.eval()
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    model = model.cpu()
    status_per_class = { # class : [correctly_predicted , wrongly_predicted]
        '0': [0,0],
        '1': [0,0],
        '2': [0,0],
        '3': [0,0],
        '4': [0,0],
        '5': [0,0],
        '6': [0,0],
        '7': [0,0],
        '8': [0,0],
        '9': [0,0],
    }
    for i , (data, target) in enumerate(data_loader):
        data = data.float()
        #if torch.cuda.is_available():
            #data = data.cuda()
            #target = target.cuda()
        data = data.cpu()
        output = model(data)
        target = target.cpu()
        if mnistPoisoned:
            loss += criterion(output, torch.argmax(target, dim=1))
            result = torch.all(torch.eq(output, target),  dim=1)
            correct += result.sum().item()
            t = torch.argmax(target, dim=1)
        else:
            loss += criterion(output, target)
            result = torch.eq(torch.argmax(output, dim=1), target)
            correct += result.sum().item()
            t = target
            
        for i in range(len(result)):
            if result[i]:
                status_per_class[str(t[i].item())][0] += 1
            else:
                status_per_class[str(t[i].item())][1] += 1  

    loss /= len(data_loader.dataset)
        
    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return status_per_class

def save_training_loss(loss,file_name):
    """
    Saves statistics about the training loss as an csv file
    INPUT:
        loss (list) : contains loss per epoch
        filename (str) : where to store the statistics
    """
    epochs = [x+1 for x in range(len(loss))] 
    liste = list(zip(epochs,loss))
    df = DataFrame(liste,columns=['epoch','training_loss'])
    outdir = "./results/"+ os.path.dirname(file_name) + "/" 
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, os.path.basename(file_name)) 
    df.to_csv(fullname, index=False)

def save_eval(eval_dict,file_name):
    """
    Saves evaluation statistics as a csv file 
    INPUT:
        eval_dict (dict) : dict about evaluation stats per class 
        file_name (str) : where to store the stats.
    """
    df = DataFrame(eval_dict)
    df.insert(loc=0, column='labels', value=["correctly labeled","wrongly labeled"])
    print(df)
    outdir = "./results/" +  os.path.dirname(file_name) + "/"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, os.path.basename(file_name))  
    df.to_csv(fullname, index=False)