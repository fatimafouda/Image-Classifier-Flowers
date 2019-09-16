import argparse
from utilis import read_labels, load_data, create_model, train_model, save_checkpoint
'''
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --network"vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_nodes 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
'''
parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", default="flowers", help="path to data directory")
parser.add_argument("--labels", default="cat_to_name.json", help="path to json file containing labels")
parser.add_argument("--network", default="densenet161", help="choose a pretrained model from torchvision.models")
parser.add_argument("--learning_rate", type=float, default=0.001, help="set learning rate")
parser.add_argument("--dropout", type=float, default=0.15, help="set dropout rate")
parser.add_argument("--hidden_nodes", type=int, default=300, help="set number of nodes in the hidden layer")
parser.add_argument("--epochs", type=int, default=10, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="path to save model checkpoint")

args = parser.parse_args()

print("Loading labels..")
labels = read_labels(args.labels)

print("Preprocessing dataset..")
trainloader, vloader, testloader, train_data = load_data(args.data_dir)

print("creating NN model..")
model, criterion, optimizer = create_model(args.network, args.hidden_nodes, args.learning_rate, args.dropout)

print("training the model with your defined hyperparameters..")
model, epoch, accuracy, optimizer = train_model(model, criterion, optimizer, args.epochs, trainloader, vloader, args.gpu, train_data)

print("saving checkpoint..")
if args.save_dir:
    save_checkpoint(model, args.network, epoch, accuracy, criterion, optimizer, args.save_dir)