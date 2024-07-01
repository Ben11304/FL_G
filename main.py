import pandas as pd 
import FL
import data_work
import torch
from sklearn.model_selection import train_test_split
import wandb
import os
# Đặt GPU mà bạn muốn sử dụng
# Thay '0' bằng số index của GPU mà bạn muốn sử dụng
gpu_index = 1
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_index)
    print(f"Đã chọn GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
else:
    print("Không có GPU nào có sẵn hoặc PyTorch không được cài đặt với hỗ trợ CUDA.")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
# device="cpu"
print(f"device : {device}")

def main():
    config=dict(
    Gan_epochs=500,
    noise_size=5,
    n_features=28,
    n_classes=8,
    lrG=0.003,
    lrD=0.003,
    epochs=15,
    classes=8,
    batch_size=128,
    learning_rate=0.0001,
    architechture="DL_model",
    IID=False,
    n_clients=4,
    criterion=torch.nn.CrossEntropyLoss(),
    Dropout_rate=0.1,
    synthetic_start_round= 4
   )
    server_data = pd.read_csv("./data/server_data.csv", index_col=0)
    test_data=pd.read_csv("./data/test_data.csv", index_col=0)
    server_data=torch.tensor(server_data.values).to(device)
    test_data=torch.tensor(test_data.values).to(device)
    trainloaders=[]
    testloaders=[]
    for i in range(4):
        temp=pd.read_csv(f"./data/trainloader_{i}.csv", index_col=0)
        trainloaders.append(temp)
        trainloaders[i]=torch.tensor(trainloaders[i].values).to(device)

        temp=pd.read_csv(f"./data/testloader_{i}.csv", index_col=0)
        testloaders.append(temp)
        testloaders[i]=torch.tensor(testloaders[i].values).to(device)
        print(trainloaders[i][:,:1].min(), trainloaders[i][:,:1].max())
    print(f"number of clients : {len(trainloaders)}")
    with wandb.init(project="FL_normal", name="V3", config=config):
        conf=wandb.config
        ser_test=FL.Federated_Learning(conf,trainloaders,testloaders,server_data,test_data,device)
        ac=ser_test.free_data_simulation_v3(15)
        wandb.finish()
for i in range(2):
    main()