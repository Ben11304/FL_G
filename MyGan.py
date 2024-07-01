import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import collections
from collections import OrderedDict
import pandas as pd
import wandb
from tqdm import tqdm 


class Gen(nn.Module):
    def __init__(self, args):
        super(Gen, self).__init__()
        self.input_dim = args.noise_size
        self.output_dim = args.n_features
        self.class_num = args.n_classes
        self.label_emb = nn.Embedding(self.class_num,self.class_num)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
    
        self.model = nn.Sequential(
            *block(self.input_dim + self.class_num, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1028),
            *block(1028,self.output_dim)
        )

    def forward(self, noise ,label):
        x = torch.cat((self.label_emb(label).squeeze(),noise), 1)
        x = self.model(x)
        return x

class Dis(nn.Module):
    def __init__(self,args):
        super(Dis, self).__init__()
        self.input_dim = args.n_features
        self.output_dim = args.n_features
        self.class_num = args.n_classes
        self.label_emb = nn.Embedding(self.class_num,self.class_num)

        self.model = nn.Sequential(
            nn.Linear((self.class_num + self.input_dim), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input ,label):
        # Concatenate label embedding and image to produce input
        x = torch.cat((self.label_emb(label).squeeze(), input), 1)
        x = self.model(x)
        return x
    
class CGAN():
    def __init__(self, args,device):
        # parameters
        self.device = device
        self.epoch = args.Gan_epochs
        self.batch_size = args.batch_size
        self.z_dim = args.noise_size
        self.class_num = args.n_classes
        #location of Gen and Dis parameters keys
        self.D_params_key=[]
        self.G_params_key=[]


        # networks init
        self.G = Gen(args).to(self.device)
        self.D = Dis(args).to(self.device)


        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG)
        self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=args.lrD, alpha=0.9)
        self.MSE_loss = torch.nn.MSELoss()
        self.BCE_loss=nn.BCELoss()
        self.CE_loss=nn.CrossEntropyLoss()
    def to(self,device):
        self.G.to(device)
        self.D.to(device)
    def freedata_train(self,m_model,criterion,round="0",name_log="Generator" ):
        # wandb.watch(self.G,criterion,log="all",log_freq=10)
        history = {'loss': [],'accuracy':[]}
        print('training start!!')
        torch.autograd.set_detect_anomaly(True)
        for epoch in tqdm(range(self.epoch), desc="Freedata Training progress"):
            self.G.train()
            y_ = torch.randint(0, 8, (self.batch_size,))
            y_=y_.squeeze()
            y_=torch.tensor(y_).long().to(self.device)
            z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.z_dim)))).to(self.device)
            self.G_optimizer.zero_grad()
            z_=torch.rand((self.batch_size, self.z_dim)).to(self.device)
            G_ = self.G(z_, y_).to(self.device)
            M = m_model(G_).to(self.device)
            G_loss,G_accuracy=m_model.Quick_evaluate(M,y_,criterion)
            G_loss.backward()
            self.G_optimizer.step()
            # print(f"epoch {epoch} G loss : {G_loss}")
            history["loss"].append(G_loss)
            history["accuracy"].append(G_accuracy)
            # wandb.define_metric(f"{name_log}/ round {round} loss", step_metric="epochs")
            # wandb.define_metric(f"{name_log}/ round {round} accuracy", step_metric="epochs")
            # wandb.log({f"{name_log}/ round {round} accuracy": G_accuracy, "epochs":epoch})
            # wandb.log({f"{name_log}/ round {round} loss":G_loss, "epochs":epoch})
        return history
                

    def train(self,data):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        if len(data)!=0:
            self.data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)

        self.D.train()
        print('training start!!')
        start_time = time.time()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, da in enumerate(self.data_loader):
                x_=da[:,1:].float()
                y_=da[:,:1].int()
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (self.batch_size, self.z_dim))))
                
                # update D network
                self.D_optimizer.zero_grad()
                D_real = self.D(x_, y_)
                D_real_loss = self.MSE_loss(D_real, self.y_real_)
                G_ = self.G(z_, y_)
                D_fake = self.D(G_, y_)
                D_fake_loss = self.MSE_loss(D_fake, self.y_fake_)
                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())
                D_loss.backward()
                self.D_optimizer.step()

                 # update G network
                self.G_optimizer.zero_grad()
                z_=torch.rand((self.batch_size, self.z_dim))
                G_ = self.G(z_, y_)
                D_fake = self.D(G_, y_)
                G_loss = self.MSE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())
                G_loss.backward()
                self.G_optimizer.step()
                
               
                if (iter + 1) == self.data_loader.dataset.__len__() // self.batch_size:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
                

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!")
        return self.train_hist
        
    def get_parameters(self):
        params=self.D.state_dict()
        D_parameters=[]
        D_keys=[]
        for key,tensor in params.items():
            D_parameters.append(tensor)
            D_keys.append(key)
        self.D_params_key=D_keys

        params=self.G.state_dict()
        G_parameters=[]
        G_keys=[]
        for key,tensor in params.items():
            G_parameters.append(tensor)
            G_keys.append(key)
        self.G_params_key=G_keys

        return D_parameters,G_parameters
        
    def load_parameter(self,D_parameters,G_parameters):
        #load Dis parameter
        if isinstance(D_parameters, OrderedDict):
            self.D.load_state_dict(D_parameters)
        else:
            tensor=[]
            for par in D_parameters:
                tensor.append(torch.tensor(par))
            params = collections.OrderedDict(zip(self.D_params_key,tensor))
            self.D.load_state_dict(params)
        
        #load Gen parameters
        if isinstance(G_parameters, OrderedDict):
            self.G.load_state_dict(G_parameters)
        else:
            tensor=[]
            for par in G_parameters:
                tensor.append(torch.tensor(par))
            params = collections.OrderedDict(zip(self.G_params_key,tensor))
            self.G.load_state_dict(params)
            
    def sample(self,y,n_samples): #chưa logic lắm n_samples=len(y)
        if isinstance(y, pd.DataFrame):
            y=torch.tensor(y.values).to(self.device)
        
        z= torch.rand(n_samples,self.z_dim).to(self.device)
        fdata=self.G(z,y)
        y = y.unsqueeze(1) 
        fdata=torch.cat((y,fdata),dim=1).to(self.device)
        return fdata
