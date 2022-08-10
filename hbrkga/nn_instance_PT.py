import torch
import torch.nn as nn
import torch.nn.functional as F
import hbrkga.data_utils as du
from sklearn.model_selection import train_test_split

class NNInstance():
    
    def __init__(self, train_path, test_path,
        epochs_no = 300,
        batch_size = 50000,
        patience = 50,
        min_delta = 0.003,
        prior = float("inf")):
    
        self._train_path = train_path
        self._test_path = test_path
        self._epochs_no = epochs_no #Number of epochs
        self._batch_size = batch_size
        self._patience = patience #Number of trials to reduce epoch loss based on the min_delta
        self._min_delta = min_delta #min MSE reduction in each epoch
        self._prior = prior
        self._num_nodes = 5 # Because there are 5 hyperparameters right now this will be 5, but might change in the future I assume
        
        self._trainX, self._trainY, self._predX, self._predY, self._n_classes = du.csv_to_numpy_array(self._train_path, self._test_path)
        self._num_x = self._trainX.shape[1]
        self._num_y = self._trainY.shape[1]
        self._trainX = torch.Tensor(self._trainX)
        self._trainY = torch.Tensor(self._trainY)
        self._predX = torch.Tensor(self._predX)
        self._predY = torch.Tensor(self._predY)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class NeuralNet(nn.Module):
        def __init__(self, input_size, layer1, layer2, layer3, n_classes):
            super(NNInstance.NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_size, layer1) 
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(layer1, layer2)
            self.fc3 = nn.Linear(layer2, layer3)  
            self.fc4 = nn.Linear(layer3, n_classes)  

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            out = self.relu(out)
            out = self.fc4(out)
            return F.log_softmax(out, dim=1)

        def f1(self, predX, y_true):

            y_pred = self(predX)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = F.one_hot(y_pred, y_true.shape[1])
            
            tp = (y_true * y_pred).sum().to(torch.float32)
            tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
            fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
            fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            f1 = 2* (precision*recall) / (precision + recall)
            return f1

    ###########################################################################

    def train(self, layer1, layer2, layer3, learning_rate, rr, percentage = 1.0):

        if percentage != 1.0:
            _, trainX, _, trainY = train_test_split(self._trainX, self._trainY, test_size=float(percentage), stratify=self._trainY)
        else:
            trainX, trainY = self._trainX, self._trainY

        model = self.NeuralNet(self._num_x, layer1, layer2, layer3, self._n_classes).to(self.device)

        # Loss and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=rr)

        best_f1 = 0

        # Train the model
        for epoch in range(self._epochs_no):
            for i in range(0, len(trainX), self._batch_size): 
                # Move tensors to the configured device
                batchX = trainX[i:i+self._batch_size].to(self.device)
                batchY = trainY[i:i+self._batch_size].to(self.device)
                
                # Forward pass
                outputs = model(batchX)
                loss = loss_function(outputs, batchY)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Best f1
                f1 = model.f1(self._predX.to(self.device), self._predY.to(self.device))
                if f1 > best_f1:
                    best_f1 = f1
            if self.early_stop(epoch, loss.item()):
                #print(f"Stopped at epoch {epoch}/{self._epochs_no}")
                break
        #if epoch == self._epochs_no-1:
            #print("Didn't stop")

        return best_f1
          
    ###########################################################################

    def early_stop(self, epoch, epoch_loss):
        flag = False
        #print(f"self._prior - epoch_loss = {self._prior - epoch_loss}")
        if epoch == 0:
            self._patience_cnt = 0
        elif self._prior - epoch_loss > self._min_delta:
            self._patience_cnt = 0
        else:
            self._patience_cnt += 1
        if self._patience_cnt > self._patience:
            flag = True
            self._patience_cnt = 0
        self._prior = epoch_loss
        return flag