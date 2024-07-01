import torch
import DL_model
import sys
import torch.nn as nn
import MyGan as Gan
import os
from torch.utils.data import TensorDataset, DataLoader
import wandb
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # p_t
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


'''Client'''
class client():
    def __init__(self, cid, config, trainloader, testloader,device):
       self.cid=cid
       self.trainset=trainloader.to(device) #tensor
       # self.testset=testloader.to(device) #tensor
       self.labels = torch.unique(self.trainset[:,:1].squeeze())
       self.Gan=Gan.CGAN(config,device)
       self.model=DL_model.Net(config.Dropout_rate,device).to(device)
       self.model_config=config
       
    def update_model(self,params):
        self.model.load_parameters(params)
    def update_Gan(self,params):
        self.Gan.load_parameters(params)
    def Gan_fit(self,data):
        hist=self.Gan.train(data)
        for G_loss in hist['G_loss']:
            wandb.log({f"client {self.cid}/ G_loss":G_loss})
        for D_loss in hist['D_loss']:
            wandb.log({f"client {self.cid}/ D_loss":D_loss})    
        #code Generator fit data
    def model_fit(self,data,criterion, round=""):
        if len(data)!=0:
            X=data[:,1:].float()
            y=data[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        else:
            X=self.trainset[:,1:].float()
            y=self.trainset[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        dataset=TensorDataset(X,y)
        dataloader=DataLoader(dataset, batch_size=128)
        his=self.model.fit(dataloader,self.model_config.learning_rate,criterion,self.model_config.epochs,f"client {self.cid}",f"round {round}" )
        for loss in his["loss"]:
            wandb.log({f"client {self.cid}/model_loss": loss})
        # wandb.log({f"test_local_model_{self.cid}":his["loss"][-1]})
        #model classify fit
    def get_parameters(self):
        D_parameters,G_parameters=self.Gan.get_parameters()
        M_parameters=self.model.get_parameters()
        return M_parameters,D_parameters, G_parameters
    def evaluate(self,X,y,criterion):
        y=y.squeeze()
        y=y.long()
        loss,accuracy=self.model.evaluate(X,y,criterion)
        # wandb.log({f"client {self.cid}/model_accuracy":accuracy })
        return loss,accuracy
    def Gen_synthetic(self,required):
        required=required.squeeze()
        print(f"generating {len(required)} synthetic data")
        wandb.log({"quantity_of_syntheticData": len(required)})
        return self.Gan.sample(required,len(required))
    def Gen_fake(self,n_samples):
        y=torch.randint(0, 8, (n_samples,))
        y=y.squeeze().to(self.device)
        wandb.log({"quantity_of_syntheticData":n_samples})
        return self.Gan.sample(y,n_samples)
    
def fn_client(cid,config, trainloader, testloader,device)->client:
    return client(cid, config, trainloader, testloader,device)

'''server'''
class server():
    def __init__(self,config, trainloader,testloader,device):
       self.trainset= trainloader.to(device)#tensor
       self.testset= testloader.to(device) #tensor
       self.Gan=Gan.CGAN(config,device)
       self.model=DL_model.Net(config.Dropout_rate,device).to(device)
       self.model_config=config
       self.device=device
        
    def update_model(self,params):
        self.model.load_parameters(params)
    def update_Gan(self,params):
        self.Gan.load_parameters(params)
    def Gan_freedata_fit(self,criterion,round="0"):
        his=self.Gan.freedata_train(self.model,criterion,round,"server_generator")
        for loss in his["loss"]:
            wandb.log({"server/Generator_loss":loss})
        for accuracy in his["accuracy"]:
            wandb.log({"server/Generator_accuracy":accuracy})     
        #code Generator fit data
    def model_fit(self,data,criterion):
        if len(data)!=0:
            X=data[:,1:].float()
            y=data[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        else:
            X=self.trainset[:,1:].float()
            y=self.trainset[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        dataset=TensorDataset(X,y)
        dataloader=DataLoader(dataset, batch_size=128)
        his=self.model.fit(dataloader,self.model_config.learning_rate,criterion,self.model_config.epochs, " ","server_DeepLearningModel")
        for loss in his["loss"]:
            wandb.log({f"server/model_loss": loss})
        for accuracy in his["accuracy"]:
            wandb.log({f"server/model_accuracy": accuracy})
        
    def get_parameters(self):
        D_parameters,G_parameters=self.Gan.get_parameters()
        M_parameters=self.model.get_parameters()
        return M_parameters,D_parameters, G_parameters
    def evaluate(self,criterion):
        X=self.testset[:,1:].float().to(self.device)
        y=self.testset[:,:1].to(self.device)
        y=y.squeeze().tolist()
        y=torch.tensor(y).long()
        loss,accuracy=self.model.evaluate(X,y,criterion)
        return loss,accuracy
    def Gen_synthetic(self,required):
        required=required.squeeze()
        print(f"generating {len(required)} synthetic data")
        wandb.log({"quantity_of_syntheticData": len(required)})
        return self.Gan.sample(required,len(required))
    def Gen_fake(self,n_samples):
        y=torch.randint(0, 8, (n_samples,))
        y=y.squeeze().to(self.device)
        wandb.log({"quantity_of_syntheticData":n_samples})
        return self.Gan.sample(y,n_samples)


'''Frame work'''
class Federated_Learning():
    def __init__(self,config, trainloaders, testloaders, serverdata, testdata,device ):
        self.server=server(config, serverdata, testdata,device)
        self.testset=testdata.to(device)
        self.clients=[]
        self.n_clients=4
        for i in range(self.n_clients):
            cl=fn_client(i,config, trainloaders[i], testloaders[i],device)
            self.clients.append(cl)
        self.criterion=torch.nn.CrossEntropyLoss()
        # self.criterion=FocalLoss()
        self.device=device
        self.synthetic_start_round=config.synthetic_start_round
    def client_M_update(self):
        M_params=self.server.model.get_parameters()
        for i in range(self.n_clients):
            self.clients[i].update_model(M_params)
    def server_M_update(self):
        params=self.clients[0].model.get_parameters()
        for i in range(len(params)):
            for k in range(1,self.n_clients):
                params[i]=params[i]+self.clients[k].model.get_parameters()[i]
            params[i]=params[i]/self.n_clients
        self.server.update_model(params)
        print("finished AVG model")
    
    def normal_FL(self,rounds):
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.client_M_update()
            for i in range(self.n_clients):
                fit_data=torch.cat((self.clients[i].trainset,round_fake_data),dim=0).to(self.device)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
            self.server_M_update()
            loss,accuracy=self.server.evaluate(self.criterion)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round })
            print(f"round {round} accuracy for server: {accuracy}")
        return accuracy_hist
        
    def free_data_simulation_v2(self,rounds):
            #sinh dữ liệu random 
            accuracy_hist=[]
            print(f"initial setup for free data training")
            loss,accuracy=self.server.evaluate(self.criterion)
            print(f"----------initial accuracy {accuracy} ----------")
            round_fake_data=torch.empty(1, 29).to(self.device)
            for round in range(rounds):
                self.client_M_update()
                for i in range(self.n_clients):
                    fit_data=torch.cat((self.clients[i].trainset,round_fake_data),dim=0).to(self.device)
                    print(f"processing client {i}")
                    fit_data=fit_data.detach()
                    self.clients[i].model_fit(fit_data,self.criterion,round)
                    self.clients[i].evaluate(self.server.testset[:,1:].float(),self.server.testset[:,:1].float(),self.criterion)
                self.server_M_update()
                loss,accuracy=self.server.evaluate(self.criterion)
                accuracy_hist.append(accuracy)
                print(f"round {round} accuracy for server: {accuracy}")
                print(f"starting training for generator")
                self.server.Gan_freedata_fit(self.criterion,round)
                if round>self.synthetic_start_round-1:
                    f=self.server.Gen_fake(10000)
                    print(f"-------------ready for round {round}-------------")
                    # self.client_M_update()
                    round_fake_data=f.to(self.device)
                    round_fake_data.detach()
                wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
                wandb.log({f"framework/server_accuracy": accuracy, "rounds": round})
            return accuracy_hist
    def free_data_simulation_v4(self,rounds):
        #Gan chỉ tham gia khi chuẩn bị đánh giá
        accuracy_hist=[]
        print(f"initial setup for freedata version 3 training")
        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ------------")
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds": 0})
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.client_M_update()
            print(f"completed to synchronized server parameters to all clients")
            if round> self.synthetic_start_round-1:
                print(f"starting training for generator")
                self.server.Gan_freedata_fit(self.criterion,round)
            for i in range(self.n_clients):
                fit_data=self.clients[i].trainset.to(self.device)
                if round> self.synthetic_start_round-1:
                    full_range = list(range(8))
                    value_list = [x.item() for x in list(self.clients[i].labels)]
                    missing_values = [x for x in full_range if x not in value_list]
                    tensor_fake=torch.tensor(missing_values)
                    tensor_fake=tensor_fake.repeat(1,10000 )
                    print(tensor_fake)
                    tensor_fake=tensor_fake.long().to(self.device)
                    f=self.server.Gen_synthetic(tensor_fake)
                    f=f.detach()
                    fit_data=torch.cat((self.clients[i].trainset,f),dim=0).to(self.device)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
                # self.clients[i].evaluate(self.server.testset[:,1:].float(),self.server.testset[:,:1].float(),self.criterion)
            self.server_M_update()
            loss,accuracy=self.server.evaluate(self.criterion)
            accuracy_hist.append(accuracy)
            print(f"round {round} accuracy for server: {accuracy}")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds": round+1})
        return accuracy_hist

    
    def free_data_simulation_v3(self,rounds):
        #sinh đúng dữ liệu thiếu
        accuracy_hist=[]
        print(f"initial setup for freedata version 3 training")
        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ------------")
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds": 0})
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.client_M_update()
            print(f"completed to synchronized server parameters to all clients")
            for i in range(self.n_clients):
                fit_data=self.clients[i].trainset.to(self.device)
                if round> self.synthetic_start_round-1:
                    full_range = list(range(8))
                    value_list = [x.item() for x in list(self.clients[i].labels)]
                    missing_values = [x for x in full_range if x not in value_list]
                    tensor_fake=torch.tensor(missing_values)
                    tensor_fake=tensor_fake.repeat(1,10000 )
                    print(tensor_fake)
                    tensor_fake=tensor_fake.long().to(self.device)
                    f=self.server.Gen_synthetic(tensor_fake)
                    f=f.detach()
                    fit_data=torch.cat((self.clients[i].trainset,f),dim=0).to(self.device)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
                # self.clients[i].evaluate(self.server.testset[:,1:].float(),self.server.testset[:,:1].float(),self.criterion)
            self.server_M_update()
            loss,accuracy=self.server.evaluate(self.criterion)
            accuracy_hist.append(accuracy)
            print(f"round {round} accuracy for server: {accuracy}")
            print(f"starting training for generator")
            self.server.Gan_freedata_fit(self.criterion,round)
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds": round+1})
        return accuracy_hist
        
    def free_data_simulation_v1(self,rounds):
        # sử dụng dữ liệu của server để có thể huấn luyện ban đầu cho GAN
        accuracy_hist=[]
        print(self.server.Gan.G)
        print(f"initial setup for free data training")
        self.server.model_fit([],self.criterion)
        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"initial server model, accuracy: {accuracy}, loss: {loss}")
        total_syntheticdata=self.server.trainset
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.server.Gan_freedata_fit(self.criterion,round )
            if round>5:
                f=self.server.Gen_fake(10000)
                total_syntheticdata=torch.cat((total_syntheticdata,f),dim=0).detach().to(self.device)
                print(f"-------------ready for round {round}-------------")
                self.client_M_update()
                round_fake_data=f.to(self.device)
                round_fake_data.detach()
            print(f"complete to update client's M model")
            for i in range(self.n_clients):
                fit_data=torch.cat((self.clients[i].trainset,round_fake_data),dim=0)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
            self.server_M_update()
            loss,accuracy=self.server.evaluate(self.criterion)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds": round})
            print(f"round {round} accuracy for server: {accuracy}")

        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"----------last accuracy {accuracy} ----------")
        return accuracy_hist
    def Gan_for_all_clients(self,rounds):
        print(f"initial setup for training")
        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ------------")
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds": 0})
        for round in range(rounds):
            synthetic_data=torch.empty(1, 29).to(self.device)
            for i in range(self.n_clients):
                self.client_M_update()
                self.clients[i].Gan_fit(self.clients[i].trainset)
                required=torch.tensor(self.clients[i].labels)
                required=required.repeat(1,10000)
                required=required.long().to(self.device)
                f=self.clients[i].Gen_synthetic(required)
                f=f.detach()
                synthetic_data=torch.cat((synthetic_data,f),dim=0).to(self.device)
                self.clients[i].model_fit(self.clients[i].trainset,self.criterion,round)
            self.server_M_update()
            self.server.model_fit(synthetic_data,self.criterion)
            loss,accuracy=self.server.evaluate(self.criterion)
            print(f"round {round} accuracy for server: {accuracy}")
            print(f"starting training for generator")
            self.server.Gan_freedata_fit(self.criterion,round)
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds": round+1})
        
