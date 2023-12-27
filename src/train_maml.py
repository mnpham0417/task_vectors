import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse

#import cycle
from itertools import cycle

#fix random seed
torch.manual_seed(0)

def manual_forward_mlp(model_0, model_1, x, alpha=0.2):
    # Ensure input is properly shaped
    x = x.view(-1, x.size(1)*x.size(2)*x.size(3))  # Flatten the image

    # Manually perform operations of each layer with parameter differences
    # Layer 1
    weight_diff1 = model_0.fc1.weight - alpha*(model_1.fc1.weight - model_0.fc1.weight)
    bias_diff1 = model_0.fc1.bias - alpha*(model_1.fc1.bias - model_0.fc1.bias)
    x = torch.relu(torch.matmul(x, weight_diff1.t()) + bias_diff1)

    # Layer 2
    weight_diff2 = model_0.fc2.weight - alpha*(model_1.fc2.weight - model_0.fc2.weight)
    bias_diff2 = model_0.fc2.bias - alpha*(model_1.fc2.bias - model_0.fc2.bias)
    x = torch.relu(torch.matmul(x, weight_diff2.t()) + bias_diff2)

    # Layer 3
    weight_diff3 = model_0.fc3.weight - alpha*(model_1.fc3.weight - model_0.fc3.weight)
    bias_diff3 = model_0.fc3.bias - alpha*(model_1.fc3.bias - model_0.fc3.bias)
    x = torch.matmul(x, weight_diff3.t()) + bias_diff3

    return x

def manual_forward_single_model_mlp(model_0, x):
    
    # Ensure input is properly shaped
    x = x.view(-1, x.size(1)*x.size(2)*x.size(3))  # Flatten the image
    
    # Manually perform operations of each layer with parameter 
    weight1 = model_0.fc1.weight
    bias1 = model_0.fc1.bias
    x = torch.relu(torch.matmul(x, weight1.t()) + bias1)
    
    weight2 = model_0.fc2.weight
    bias2 = model_0.fc2.bias
    x = torch.relu(torch.matmul(x, weight2.t()) + bias2)
    
    weight3 = model_0.fc3.weight
    bias3 = model_0.fc3.bias
    
    x = torch.matmul(x, weight3.t()) + bias3
    
    return x

def manual_forward_cnn(model_0, model_1, x, alpha=0.2):
    # Convolutional Layer 1
    weight_diff1 = model_0.conv1.weight - alpha*(model_1.conv1.weight - model_0.conv1.weight)
    bias_diff1 = model_0.conv1.bias - alpha*(model_1.conv1.bias - model_0.conv1.bias)
    x = F.conv2d(x, weight_diff1, bias_diff1, stride=1, padding=1)
    x = F.relu(x)

    # Max Pooling Layer 1
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Convolutional Layer 2
    weight_diff2 = model_0.conv2.weight - alpha*(model_1.conv2.weight - model_0.conv2.weight)
    bias_diff2 = model_0.conv2.bias - alpha*(model_1.conv2.bias - model_0.conv2.bias)
    x = F.conv2d(x, weight_diff2, bias_diff2, stride=1, padding=1)
    x = F.relu(x)

    # Max Pooling Layer 2
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Flatten the output for the fully connected layer
    x = x.view(-1, 64 * 7 * 7)

    # Fully Connected Layer 1
    weight_diff3 = model_0.fc1.weight - alpha*(model_1.fc1.weight - model_0.fc1.weight)
    bias_diff3 = model_0.fc1.bias - alpha*(model_1.fc1.bias - model_0.fc1.bias)
    x = F.linear(x, weight_diff3, bias_diff3)
    x = F.relu(x)

    # Fully Connected Layer 2
    weight_diff4 = model_0.fc2.weight - alpha*(model_1.fc2.weight - model_0.fc2.weight)
    bias_diff4 = model_0.fc2.bias - alpha*(model_1.fc2.bias - model_0.fc2.bias)
    x = F.linear(x, weight_diff4, bias_diff4)

    return x

def manual_forward_single_model_cnn(model_0, x):
    # Convolutional Layer 1
    weight1 = model_0.conv1.weight
    bias1 = model_0.conv1.bias - model_0.conv1.bias
    x = F.conv2d(x, weight1, bias1, stride=1, padding=1)
    x = F.relu(x)

    # Max Pooling Layer 1
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Convolutional Layer 2
    weight2 = model_0.conv2.weight
    bias2 = model_0.conv2.bias
    x = F.conv2d(x, weight2, bias2, stride=1, padding=1)
    x = F.relu(x)

    # Max Pooling Layer 2
    x = F.max_pool2d(x, kernel_size=2, stride=2)

    # Flatten the output for the fully connected layer
    x = x.view(-1, 64 * 7 * 7)

    # Fully Connected Layer 1
    weight3 = model_0.fc1.weight
    bias3 = model_0.fc1.bias
    x = F.linear(x, weight3, bias3)
    x = F.relu(x)

    # Fully Connected Layer 2
    weight_diff4 = model_0.fc2.weight
    bias_diff4 = model_0.fc2.bias
    x = F.linear(x, weight_diff4, bias_diff4)

    return x

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--arch', type=str, required=True, choices=['mlp', 'cnn'])
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--target_class', type=int, required=True)
    parser.add_argument('--num_epoch', type=int, required=True, default=60)
    parser.add_argument('--pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=1.1)
    parser.add_argument('--control_weight', type=float, default=0.3)

    args = parser.parse_args()
    return args

# Function to filter only '0' class
def filter_digit(indices, dataset, label_list):
    return [i for i in indices if dataset.targets[i] in label_list]

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

    if(args.arch == 'mlp'):
        model_theta_0 = SimpleMLP(num_out)
        model_theta_1 = SimpleMLP(num_out)
        if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
            model_theta_0.fc1 = nn.Linear(28*28*3, 512).to(device)
            model_theta_1.fc1 = nn.Linear(28*28*3, 512).to(device)
        model_theta_0.to(device)
        model_theta_1.to(device)
        
    elif(args.arch == 'cnn'):
        model_theta_0 = SimpleCNN(num_out)
        model_theta_1 = SimpleCNN(num_out)

        if(args.dataset == 'cifar10' or args.dataset == 'cifar100'):
            model_theta_0.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            model_theta_1.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        model_theta_0.to(device)
        model_theta_1.to(device)
        
    model_theta_0.load_state_dict(torch.load(args.pretrained_checkpoint).state_dict())
    model_theta_1.load_state_dict(torch.load(args.pretrained_checkpoint).state_dict())

    #freeze parameters of model_theta_0
    for param in model_theta_0.parameters():
        param.requires_grad = True
        
    # Ensuring gradients are tracked for theta_0
    for param in model_theta_1.parameters():
        param.requires_grad = False
        
    optimizer = optim.SGD(model_theta_0.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    target_digit = [args.target_class]

    sub_train_target_indices = filter_digit(range(len(train_dataset)), train_dataset, target_digit)
    sub_train_dataset_target_indices = Subset(train_dataset, sub_train_target_indices)
    sub_train_loader_target_indices = DataLoader(dataset=sub_train_dataset_target_indices, batch_size=64, shuffle=True)

    control_digit = []
    for i in range(num_out):
        if i == args.target_class:
            continue
        control_digit.append(i)
    
    #make sure there are num_out - 1 control digits
    assert len(control_digit) == num_out - 1
        
    sub_train_control_indices = filter_digit(range(len(train_dataset)), train_dataset, control_digit)
    sub_train_dataset_control_indices = Subset(train_dataset, sub_train_control_indices)
    sub_train_loader_control_indices = DataLoader(dataset=sub_train_dataset_control_indices, batch_size=64, shuffle=True)

    for epoch in range(args.num_epoch):
        total_target_loss = 0
        total_control_loss = 0
        total_loss = 0
        #zip both loaders together, so that we can iterate through them simultaneously, use cycle 
        #to repeat the shorter loader
        for batch_idx, ((data_target, target_target), (data_control, target_control)) in enumerate(zip(cycle(sub_train_loader_target_indices), sub_train_loader_control_indices)):
            model_theta_0.train()
            model_theta_1.eval()
            optimizer.zero_grad()
            data_target, target_target = data_target.to(device), target_target.to(device)
            data_control, target_control = data_control.to(device), target_control.to(device)
            
            if(args.arch == 'mlp'):
                target_output = manual_forward_single_model_mlp(model_theta_0, data_target)
                control_output = manual_forward_mlp(model_theta_1, model_theta_0, data_control, args.alpha)
            elif(args.arch == 'cnn'):
                target_output = manual_forward_single_model_cnn(model_theta_0, data_target)
                control_output = manual_forward_cnn(model_theta_1, model_theta_0, data_control, args.alpha)
            
            loss_target = criterion(target_output, target_target)
            loss_control = criterion(control_output, target_control)
            
            loss = loss_target + args.control_weight*loss_control
            loss.backward()
            optimizer.step()
            
            total_target_loss += loss_target.item()
            total_control_loss += loss_control.item()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Target Loss: {total_target_loss/(batch_idx+1)}, Control Loss: {total_control_loss/(batch_idx+1)}, Total Loss: {total_loss/(batch_idx+1)}")

    #save model
    torch.save(model_theta_0, args.model_name)