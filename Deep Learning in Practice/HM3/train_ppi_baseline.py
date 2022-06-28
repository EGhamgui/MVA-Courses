import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import GATConv
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader


MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


class BasicGraphModel(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)

        return outputs

class GAT(nn.Module):

    def __init__(self, g, n_layers, input_size, hidden_size, output_size, num_heads, nonlinearity,dropout = 0):
        super().__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.nonlinearity = nonlinearity

        # Replace GraphConv layers with GATConv layers
        self.layers.append(GATConv(input_size, hidden_size , num_heads ,dropout))
        for i in range(n_layers - 1):
            self.layers.append(GATConv(hidden_size*num_heads, hidden_size , num_heads,dropout))

        # Add a fully connected layer for classification task
        self.layers.append(nn.Linear(in_features= hidden_size*num_heads, out_features=output_size))

    def forward(self, inputs):
        outputs = inputs
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                outputs = self.layers[i](outputs).flatten(1)
            else :
                # Add nonlinearity function to the last layer 
                outputs = self.nonlinearity(self.layers[i](self.g, outputs).flatten(1))
        return outputs

def main(args):

    # load dataset and create dataloader
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))

    ########### Replace this model with your own GNN implemented class ################################

    # Define variables 
    n_layers = 3
    hidden_size = 350
    num_heads = 4
    lr = 0.005
    nonlinearity = F.leaky_relu

    model = GAT(g=train_dataset.graph,n_layers=n_layers,input_size=n_features,hidden_size=hidden_size,output_size=n_classes,
                            num_heads=num_heads,nonlinearity=nonlinearity).to(device)

    #model = BasicGraphModel(g=train_dataset.graph, n_layers=2, input_size=n_features,
                            #hidden_size=256, output_size=n_classes, nonlinearity=F.elu).to(device)

    ###################################################################################################

    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset)
        torch.save(model.state_dict(), MODEL_STATE_FILE)

    # import model from file
    model.load_state_dict(torch.load(MODEL_STATE_FILE))

    # test the model
    test(model, loss_fcn, device, test_dataloader)

    return model

def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset):

    f1_score_list = []
    epoch_list = []

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            subgraph = subgraph.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 5 == 0:
            scores = []
            for batch, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                subgraph = subgraph.clone().to(device)
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn)
                scores.append(score)
                f1_score_list.append(score)
                epoch_list.append(epoch)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))

    #plot_f1_score(epoch_list, f1_score_list)

def test(model, loss_fcn, device, test_dataloader):
    test_scores = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        subgraph = subgraph.to(device)
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores

def evaluate(features, model, subgraph, labels, loss_fcn):
    with torch.no_grad():
        model.eval()
        model.g = subgraph
        for layer in model.layers:
            layer.g = subgraph
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()

def collate_fn(sample) :
    # concatenate graph, features and labels w.r.t batch size
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels

def plot_f1_score(epoch_list, f1_score_list) :

    plt.plot(epoch_list, f1_score_list)
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.title("Evolution of f1 score w.r.t epochs")
    name = 'fig.png'
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":

    # PARSER TO ADD OPTIONS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    
    args = parser.parse_args()

    # READ MAIN
    main(args)
