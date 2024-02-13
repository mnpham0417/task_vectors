import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
from task_vectors import TaskVector
import argparse

class SimpleMLP(nn.Module):
    def __init__(self, num_out=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_out)

    def forward(self, x):
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_out=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_out)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def test(model, test_loader):
    # Step 5: Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct, total

# Function to filter only '0' class
def filter_digit(indices, dataset, label):
    return [i for i in indices if dataset.targets[i] == label]

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--finetuned_checkpoint', type=str, required=True)
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    
    if(args.dataset == 'mnist'):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_out = 10
    elif(args.dataset == 'cifar10'):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize((28, 28))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_out = 10
    elif(args.dataset == 'cifar100'):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize((28, 28))])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_out = 100
        
    task_vector = TaskVector(pretrained_checkpoint=args.pretrained_checkpoint, 
                                finetuned_checkpoint=args.finetuned_checkpoint)
    neg_task_vector = -task_vector
    model = neg_task_vector.apply_to(args.pretrained_checkpoint, scaling_coef=args.alpha)
    
    print(args.pretrained_checkpoint)
    print(args.finetuned_checkpoint)

    acc_list = []
    acc_list_control = []
    
    for i in range(num_out):
        # sub_train_indices = filter_digit(range(len(train_dataset)), train_dataset, i)
        sub_test_indices = filter_digit(range(len(test_dataset)), test_dataset, i)
        
        # sub_train_dataset = Subset(train_dataset, sub_train_indices)
        sub_test_dataset = Subset(test_dataset, sub_test_indices)
        
        # sub_train_loader = DataLoader(dataset=sub_train_dataset, batch_size=64, shuffle=True)
        sub_test_loader = DataLoader(dataset=sub_test_dataset, batch_size=1000, shuffle=False)
        
        correct, total = test(model, sub_test_loader)
                
        print(f"Accuracy of {i} class: {correct/total}")

        acc_list.append(correct/total)
        if(i == args.target_class):
            continue
        acc_list_control.append(correct/total)
        
    print("Average accuracy: ", sum(acc_list)/len(acc_list))
    print("Average accuracy control: ", sum(acc_list_control)/len(acc_list_control))