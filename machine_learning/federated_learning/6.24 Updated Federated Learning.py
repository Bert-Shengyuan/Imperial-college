import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import random
import math
import torch
from torchvision import datasets, transforms
import gzip
from pathlib import Path

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training data
trainset = datasets.MNIST('mnist_data', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('mnist_data', download=True, train=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 设置数据路径
DATA_PATH = Path("mnist_data")
PATH = DATA_PATH / "MNIST/raw"

# 文件名
train_images_file = PATH / "train-images-idx3-ubyte.gz"
train_labels_file = PATH / "train-labels-idx1-ubyte.gz"
test_images_file = PATH / "t10k-images-idx3-ubyte.gz"
test_labels_file = PATH / "t10k-labels-idx1-ubyte.gz"

x_train = load_mnist_images(train_images_file)
y_train = load_mnist_labels(train_labels_file)
x_test = load_mnist_images(test_images_file)
y_test = load_mnist_labels(test_labels_file)

# 正规化数据
x_train = x_train / 255.0
x_test = x_test / 255.0

# 将训练数据分割为训练集和验证集
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 打印数据集的形状以验证
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

def split_and_shuffle_labels(y_data, seed, amount):
    y_data=pd.DataFrame(y_data,columns=["labels"])
    y_data["i"]=np.arange(len(y_data))
    label_dict = dict()
    for i in range(10):
        var_name="label" + str(i)
        label_info=y_data[y_data["labels"]==i]
        np.random.seed(seed)
        label_info=np.random.permutation(label_info)
        label_info=label_info[0:amount]
        label_info=pd.DataFrame(label_info, columns=["labels","i"])
        label_dict.update({var_name: label_info })
    return label_dict
def get_iid_subsamples_indices(label_dict, number_of_samples, amount):
    sample_dict= dict()
    batch_size=int(math.floor(amount/number_of_samples))
    for i in range(number_of_samples):
        sample_name="sample"+str(i)
        dumb=pd.DataFrame()
        for j in range(10):
            label_name=str("label")+str(j)
            a=label_dict[label_name][i*batch_size:(i+1)*batch_size]
            dumb=pd.concat([dumb,a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict

def create_iid_subsamples(sample_dict, x_data, y_data, x_name, y_name):
    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples
        xname = x_name + str(i)
        yname = y_name + str(i)
        sample_name = "sample" + str(i)

        indices = np.sort(np.array(sample_dict[sample_name]["i"]))

        x_info = x_data[indices, :]
        x_data_dict.update({xname: x_info})

        y_info = y_data[indices]
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def add_noise_to_weights(weights, noise_scale):
    """
    Add Gaussian noise to the weights.

    :param weights: A tuple of tensors (weights and biases)
    :param noise_scale: The scale of the Gaussian noise to add
    :return: A tuple of tensors with added noise
    """
    return tuple(w + torch.randn_like(w) * noise_scale for w in weights)

def train(model, train_loader, criterion, optimizer, num_iterations):
    start_time = time.time()
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for _ in range(num_iterations):
        try:
            data, target = next(iter(train_loader))
        except StopIteration:
            train_loader = iter(train_loader)
            data, target = next(train_loader)

        data, target = data.float(), target.long()

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()
        total += target.size(0)

    train_time = time.time() - start_time
    return train_loss / num_iterations, correct / total, train_time


def validation(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float(), target.long()

            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total

    return test_loss, accuracy


def create_model_optimizer_criterion_dict(number_of_samples, learning_rate, momentum):
    model_dict = {}
    optimizer_dict = {}
    criterion_dict = {}

    for i in range(number_of_samples):
        model_name = f"model{i}"
        model_info = Net2nn()
        model_dict[model_name] = model_info

        optimizer_name = f"optimizer{i}"
        ##优化器
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)

        optimizer_dict[optimizer_name] = optimizer_info

        criterion_name = f"criterion{i}"
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict[criterion_name] = criterion_info

    return model_dict, optimizer_dict, criterion_dict


def get_weighted_averaged_weights_Q(model_dict, data_proportions, participating_clients, q=1.0):
    """
    Implement q-FFL weighted averaging of model parameters.

    :param model_dict: Dictionary of client models
    :param data_proportions: List of data proportions for each client
    :param participating_clients: List of indices of participating clients
    :param q: The q parameter for q-FFL (default is 1.0)
    :return: Tuple of averaged weight tensors
    """
    # Initialize tensors for the weighted average
    first_model = next(iter(model_dict.values()))
    fc1_mean_weight = torch.zeros_like(first_model.fc1.weight)
    fc1_mean_bias = torch.zeros_like(first_model.fc1.bias)
    fc2_mean_weight = torch.zeros_like(first_model.fc2.weight)
    fc2_mean_bias = torch.zeros_like(first_model.fc2.bias)
    fc3_mean_weight = torch.zeros_like(first_model.fc3.weight)
    fc3_mean_bias = torch.zeros_like(first_model.fc3.bias)

    # Calculate F_i (loss) for each participating client
    losses = []
    for i in participating_clients:
        model = model_dict[f'model{i}']
        # Assuming you have a way to calculate loss for each model
        # This is a placeholder and should be replaced with actual loss calculation
        loss = calculate_model_loss(model)  # You need to implement this function
        losses.append(loss)

    # Calculate weights based on q-FFL formula
    weights = []
    for i, loss in zip(participating_clients, losses):
        p_i = data_proportions[i]
        weight = p_i * (loss ** (q + 1))
        weights.append(weight)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Calculate weighted average
    with torch.no_grad():
        for i, weight in zip(participating_clients, normalized_weights):
            model = model_dict[f'model{i}']
            fc1_mean_weight += model.fc1.weight.data * weight
            fc1_mean_bias += model.fc1.bias.data * weight
            fc2_mean_weight += model.fc2.weight.data * weight
            fc2_mean_bias += model.fc2.bias.data * weight
            fc3_mean_weight += model.fc3.weight.data * weight
            fc3_mean_bias += model.fc3.bias.data * weight

    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias


def calculate_model_loss(model):
    """
    Calculate the loss for a given model.
    This is a placeholder function and should be implemented based on your specific setup.
    """
    # Placeholder implementation
    # You should replace this with actual loss calculation using your validation data
    return torch.rand(1).item()  # Random loss for demonstration

def get_weighted_averaged_weights(model_dict, data_proportions,participating_clients):
    first_model = next(iter(model_dict.values()))
    fc1_mean_weight = torch.zeros_like(first_model.fc1.weight)
    fc1_mean_bias = torch.zeros_like(first_model.fc1.bias)
    fc2_mean_weight = torch.zeros_like(first_model.fc2.weight)
    fc2_mean_bias = torch.zeros_like(first_model.fc2.bias)
    fc3_mean_weight = torch.zeros_like(first_model.fc3.weight)
    fc3_mean_bias = torch.zeros_like(first_model.fc3.bias)

    valid_clients = [participating_clients[i] for i in range(len(participating_clients))]
    #print(valid_clients)
    #print(data_proportions)
    # Calculate total proportion for normalization
    total_proportion = [data_proportions[i] for i in range(len(valid_clients))]

    total_proportion = sum(total_proportion)
    #print(total_proportion)
    if total_proportion == 0:
        print("Warning: No valid participating clients. Using uniform weights.")
        total_proportion = len(valid_clients)
        weights = [1 / len(valid_clients)] * len(valid_clients)
    else:
        weights = [data_proportions[i] / total_proportion for i in range(len(valid_clients))]

    with torch.no_grad():
        for i, weight in zip(valid_clients, weights):
            fc1_mean_weight += model_dict[f'model{i}'].fc1.weight.data * weight
            fc1_mean_bias += model_dict[f'model{i}'].fc1.bias.data * weight
            fc2_mean_weight += model_dict[f'model{i}'].fc2.weight.data * weight
            fc2_mean_bias += model_dict[f'model{i}'].fc2.bias.data * weight
            fc3_mean_weight += model_dict[f'model{i}'].fc3.weight.data * weight
            fc3_mean_bias += model_dict[f'model{i}'].fc3.bias.data * weight

    return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias

#%%
def calculate_transfer_time(model_size, network_speed, packet_loss, latency):
    # Calculate effective speed considering packet loss
    effective_speed = network_speed * (1 - packet_loss)

    # Calculate transfer time
    transfer_time = (model_size / effective_speed) + latency

    # Add some randomness to simulate network fluctuations
    jitter = random.uniform(-0.1, 0.1) * transfer_time

    return transfer_time + jitter


def simulate_network_conditions():
    # Network speed in MB/s
    speed = random.uniform(0.5, 10)

    # Packet loss as a percentage
    packet_loss = random.uniform(0, 0.05)

    # Latency in seconds
    latency = random.uniform(0.05, 0.5)

    return speed, packet_loss, latency


# Then in the training loop:

def get_model_size(model):
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # Size in MB

# Then in the training loop:
def ensure_tensor(data):
    if isinstance(data, int):
        return torch.tensor([data])
    elif isinstance(data, list):
        return torch.tensor(data)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, data_proportions,
                                                                     participating_clients, noise_scale=0.01):
    #averaged_weights = get_weighted_averaged_weights_Q(model_dict, data_proportions, participating_clients, q=100.0)
    averaged_weights = get_weighted_averaged_weights(model_dict, data_proportions, participating_clients)
    # Add noise to the averaged weights
    noisy_weights = add_noise_to_weights(averaged_weights, noise_scale)

    # Unpack the noisy weights
    fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = noisy_weights

    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc2.weight.data = fc2_mean_weight.data.clone()
        main_model.fc3.weight.data = fc3_mean_weight.data.clone()
        main_model.fc1.bias.data = fc1_mean_bias.data.clone()
        main_model.fc2.bias.data = fc2_mean_bias.data.clone()
        main_model.fc3.bias.data = fc3_mean_bias.data.clone()

    return main_model


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):
            model_dict[f'model{i}'].load_state_dict(main_model.state_dict())
    return model_dict
# Example usage
number_of_clients = 10
max_wait_time = 1
# change get weight to getweiigt_Q
#


epochs = 50
batch_size = 32
learning_rate = 0.01
momentum = 0.9



#


# federated_learning_process(x_train, y_train, x_test, y_test, number_of_clients, epochs, batch_size, learning_rate,
#                                momentum, max_wait_time):
# Initialize main model and criterion
main_model = Net2nn()
main_criterion = nn.CrossEntropyLoss()
x_train = ensure_tensor(x_train)
y_train = ensure_tensor(y_train)
x_test = ensure_tensor(x_test)
y_test = ensure_tensor(y_test)
# Split data among clients
client_data = []
data_proportions = []
for i in range(number_of_clients):
    start_idx = i * len(x_train) // number_of_clients
    end_idx = (i + 1) * len(x_train) // number_of_clients
    client_x = x_train[start_idx:end_idx]
    client_y = y_train[start_idx:end_idx]
    if client_x.dim() == 1:
        client_x = client_x.unsqueeze(1)
    if client_y.dim() > 1:
        client_y = client_y.squeeze()

    client_data.append((client_x, client_y))
    data_proportions.append((end_idx - start_idx) / len(x_train))
# Create model, optimizer, and criterion dictionaries
model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_clients, learning_rate,
                                                                                   momentum)

# Prepare test data loader
test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

# Estimate model size (in MB)
model_size = sum(p.numel() for p in main_model.parameters()) * 4 / (1024 * 1024)

Acc_list = []
Time_list = []
Participating_clients_list = []

index_range =3
for i in range(index_range):
    for epoch in range(epochs):
        epoch_start_time = time.time()
        client_times = []
        participating_clients = []

        for i in range(number_of_clients):
            # Simulate network conditions for this client
            speed, packet_loss, latency = simulate_network_conditions()

            # Calculate transfer times
            download_time = calculate_transfer_time(model_size, speed, packet_loss, latency)
            upload_time = calculate_transfer_time(model_size, speed, packet_loss, latency)

            # Check if client can participate based on network conditions
            if download_time + upload_time > max_wait_time:
                #print(f"Client {i} excluded due to poor network conditions")
                continue
            else:
                participating_clients.append(i)

                # Prepare client's data loader
                train_ds = TensorDataset(client_data[i][0], client_data[i][1])
                train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

                # Calculate number of iterations for this client
                num_iterations = max(1, int(len(train_dl) * data_proportions[i]))

                # Train client model
                model = model_dict[f'model{i}']
                criterion = criterion_dict[f'criterion{i}']
                optimizer = optimizer_dict[f'optimizer{i}']
                _, _, train_time = train(model, train_dl, criterion, optimizer, num_iterations)

                client_times.append(train_time + download_time + upload_time)

        # Update main model with weighted average of participating client models
        if participating_clients:
            participating_proportions = [data_proportions[i] for i in participating_clients]
            total_proportion = sum(participating_proportions)
            normalized_proportions = [p / total_proportion for p in participating_proportions]

            participating_models = {f'model{i}': model_dict[f'model{i}'] for i in participating_clients}
            ##Update
            main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(
                main_model, model_dict, data_proportions, participating_clients, noise_scale=0.5
            )

            # Evaluate main model
            test_loss, test_accuracy = validation(main_model, test_dl, main_criterion)

            # Calculate total epoch time
            epoch_time = time.time() - epoch_start_time + max(client_times)

            print(
                f"Epoch {epoch + 1}: main_model accuracy on all test data: {test_accuracy:.4f}, Time: {epoch_time:.2f}s, Participating clients: {len(participating_clients)}/{number_of_clients}")

            Acc_list.append(test_accuracy)
            Time_list.append(epoch_time)
            Participating_clients_list.append(len(participating_clients))

            # Send updated main model back to all clients
            model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_clients)
        else:
            print(f"Epoch {epoch + 1}: No clients participated due to poor network conditions")
            Acc_list.append(None)
            Time_list.append(None)
            Participating_clients_list.append(0)

        #return Acc_list, Time_list, Participating_clients_list


     # maximum wait time in seconds

    # Assuming x_train, y_train, x_test, y_test are already defined
    #Acc_list, Time_list, Participating_clients_list = federated_learning_process(x_train, y_train, x_test, y_test,
                                                                                 # number_of_clients, epochs, batch_size,
                                                                                 # learning_rate, momentum, max_wait_time)
    Time_list = [time for time in Time_list if time is not None]



    import pickle
    with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_1_Time'+str(i)+'.pkl', 'wb') as file:
        pickle.dump(Time_list, file)

    with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_1_Acc' + str(i) + '.pkl', 'wb') as file:
        pickle.dump(Acc_list, file)

    # with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_2_Time' + str(i) + '.pkl', 'wb') as file:
    #     pickle.dump(Time_list, file)
    # with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_2_Acc' + str(i) + '.pkl', 'wb') as file:
    #     pickle.dump(Acc_list, file)


    # Print final results
    print("\nFinal Results:")
    print(f"Final Accuracy: {Acc_list[-1]:.4f}")
    print(f"Total Time: {sum(Time_list):.2f}s")
    print(f"Average number of participating clients per epoch: {sum(Participating_clients_list) / len(Participating_clients_list):.2f}")

#%%
plt.figure()
plt.plot(Acc_list)
plt.show()


# #%%
# #%%
# with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_1_Time'+str(i)+'.pkl', 'rb') as file:
#     Time_list1 = pickle.load(file)
# with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_1_Time'+str(i)+'.pkl', 'rb') as file:
#     Time_list2 = pickle.load(file)
# with open('/Users/sonmjack/Downloads/figure_compare/asy/Type_1_Time'+str(i)+'.pkl', 'rb') as file:
#     Time_list3 = pickle.load(file)
#
# # with open('/Users/sonmjack/Downloads/figure_compare/asy/T2_Time'+str(i)+'.pkl', 'rb') as file:
# #     T2_Time_list1 = pickle.load(file)
# # with open('/Users/sonmjack/Downloads/figure_compare/asy/T2_'+str(i)+'.pkl', 'rb') as file:
# #     T2_Time_list2 = pickle.load(file)
# # with open('/Users/sonmjack/Downloads/figure_compare/asy/T2_'+str(i)+'.pkl', 'rb') as file:
# #     T2_Time_list3 = pickle.load(file)
# #
# # with open('/Users/sonmjack/Downloads/figure_compare/asy/T2_Time'+str(i)+'.pkl', 'rb') as file:
# #     T3_Time_list1 = pickle.load(file)
# # with open('/Users/sonmjack/Downloads/figure_compare/asy/T2_'+str(i)+'.pkl', 'rb') as file:
# #     T3_Time_list2 = pickle.load(file)
# # with open('/Users/sonmjack/Downloads/figure_compare/asy/T2_'+str(i)+'.pkl', 'rb') as file:
# #     T3_Time_list3 = pickle.load(file)
#
# All_time_list1 = [Time_list1, Time_list2, Time_list3]
# def prepare_data(data, group_name):
#     dimensions = np.tile(np.arange(0,epochs), len(data))
#     df = pd.DataFrame({'time': dimensions, 'ACC': data.reshape(-1,1).flatten()})
#     df['Group'] = group_name
#     return df
#
# df1 = prepare_data(np.array(All_time_list1), 'T1(client = 10)')
# df2 = prepare_data(np.array(All_time_list2), 'T2(client = 20)')
# df3 = prepare_data(np.array(All_time_list3), 'T3(client = 30)')
#
# df = pd.concat([df1, df2, df3])
#
# import seaborn as sns
# fig, ax = plt.subplots()
# sns.lineplot(data=df, x='time',
#              y='Acc', hue='Group', style='Group', markers=True, dashes=False)
#
#
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.title("",fontsize=13)
# plt.xlabel('',fontsize=13)
# plt.ylabel("",fontsize=13)
# plt.tick_params(axis='y', labelsize=13)
# plt.tick_params(axis='x', labelsize=13)
# plt.savefig('/Users/sonmjack/Downloads/figure_compare/'+''+'.pdf')
# #plt.savefig('/Users/sonmjack/Downloads/figure_compare/'+'Whole asy corr WT'+'.svg')
# plt.show()