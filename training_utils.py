import matplotlib.pyplot as plt
from torch import nn, optim
import torch
import torch.cuda as cuda_device


def display_batch(dataloader,Nbr_images = 6):
    examples = enumerate(dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data = example_data.cpu()

    fig = plt.figure()
    for i in range(Nbr_images):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        print(example_data[i][0].shape)
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        #plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig

def train(train_dataloader, optimizer,model,n_epochs=50, mnistPoisoned=False, model_path="./checkpoints/model.pth"):
    """
    Train a given model
    INPUT:
    OUTPUT:
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
           
            if mnistPoisoned:
                loss = criterion(output, torch.argmax(y, dim=1))
            else:
                loss = criterion(output, y)
        
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
    model.eval()
    loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    model = model.cpu()
    for i , (data, target) in enumerate(data_loader):
        data = data.float()
        #if torch.cuda.is_available():
            #data = data.cuda()
            #target = target.cuda()
        
        output = model(data)
      
        if mnistPoisoned:
            loss += criterion(output, torch.argmax(target, dim=1))
        else:
            loss += criterion(output, target)

        correct += ( torch.all(torch.eq(output, target),  dim=1)).sum()
        
        

    loss /= len(data_loader.dataset)
        
    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))