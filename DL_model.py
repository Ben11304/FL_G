import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import collections
from collections import OrderedDict
import torch.nn.functional as F
import utils
import wandb

class Net(nn.Module):
    def __init__(self, dropout_rate,device):
        self.device = device
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(64,8)
        # self.sm = nn.Softmax()
        self.params_key=[]
        params=self.state_dict()
        keys=[]
        for key,_ in params.items():
            keys.append(key)
        self.params_key=keys
        self.to(device)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        # x = self.sm(x)
        return x

    def Quick_evaluate(self, outputs, y_test, criterion):
        # Chắc chắn rằng mọi thứ đều trên cùng một thiết bị
        outputs = outputs.to(self.device)
        y_test = y_test.long().to(self.device)
        
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).float().mean().item()  # Cải tiến hiệu suất
        
        loss = criterion(outputs, y_test)
        return loss, accuracy

    

    def evaluate(self, X_in, y_in, criterion):
        self.eval()
        if not isinstance(X_in, torch.Tensor):
            X_test = torch.tensor(X_in.values, dtype=torch.float32).to(self.device)
        else:
            X_test = X_in.float().to(self.device)
            
        if not isinstance(y_in, torch.Tensor):
            y_test = torch.tensor(y_in.values).long().squeeze().to(self.device)
        else:
            y_test = y_in.long().squeeze().to(self.device)

        with torch.no_grad():
            outputs = self(X_test).to(self.device)
            y_test = y_test.clamp(0, 7)
            y_test=y_test.to(self.device)
            loss, accuracy = self.Quick_evaluate(outputs, y_test,criterion)
        return loss.item(), accuracy
    
    def fit(self, train_loader ,learning_rate, criterion, num_epochs=10, name_log="DeepLearning_Model", note=""):
        optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, alpha=0.9)
        # Lịch sử huấn luyện
        # wandb.watch(self,criterion, log="all", log_freq=10)
        history = {'loss': [], 'accuracy': []}
        for epoch in range(num_epochs):
            # Tính toán đầu ra mô hình
            for batch_idx, (X, y) in enumerate(train_loader):
                y=y.squeeze().to(self.device)
                optimizer.zero_grad()
                outputs = self(X.float()).to(self.device)
                loss,accuracy=self.Quick_evaluate(outputs,y.long(),criterion)
                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy)
                loss.backward()
                optimizer.step()
            print(f"accuracy for epoch { epoch} : ",accuracy)
            # wandb.define_metric(f"{name_log}/ {note} DL train accuracy", step_metric="epochs")
            # wandb.define_metric(f"{name_log}/ {note} DL train loss", step_metric="epochs")
            # wandb.log({f"{name_log}/ {note} DL train accuracy": accuracy, f"{name_log}/ {note} DL train loss": loss, "epochs": epoch+1})
        return history
    def get_parameters(self):
        params=self.state_dict()
        parameters=[] 
        keys=[]
        for key,tensor in params.items():
            parameters.append(tensor)
            keys.append(key)
        self.params_key=keys
        return parameters

    
    def load_parameters(self, parameters_tensor):
        if isinstance(parameters_tensor, OrderedDict):
            self.load_state_dict(parameters_tensor)
        else:
            tensor=[]
            for par in parameters_tensor:
                tensor.append(torch.tensor(par))
            params = collections.OrderedDict(zip(self.params_key,tensor))
            self.load_state_dict(params)


    def get_weigthdivegence(self, par):
        t=float(0)
        m=float(0)
        param=self.get_parameters()
        for i in range(0,len(param),2):
            size=param[i].size()
            if len(size)==1:
                for k in range(size[0]):
                    m_model=param[i][k].item()
                    m_SGD=par[i][k].item()
                    t =t+ (m_model-m_SGD)*(m_model-m_SGD)
                    m = m+abs(m_SGD)
            else:
                for k in range(size[0]):
                    for j in range(size[1]):
                        m_model=param[i][k][j].item()
                        m_SGD=par[i][k][j].item()
                        t =t+ (m_model-m_SGD)*(m_model-m_SGD)
                        m = m+abs(m_SGD)
        return float(t/m)
    


    def Cross_validation(self,X,y,k,learning_rate: float=0.001):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        z=1
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = X_train.float()
            X_test = X_test.float()
            y_train = y_train.long()
            y_test = y_test.float()
            print(f"length of trainset {len(y_train)}, length of testset{len(y_test)}")
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = self(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                outputs = self(X_test)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_test.long()).sum().item() / y_test.size(0)
                print(f'Accuracy for fold {z} : {accuracy}')
                z=z+1
        return self



