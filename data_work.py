
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter


def get_sublabel(df, label_name: str):
    unique_labels = df[label_name].unique()
    small_dataframes = {}
    for label in unique_labels:
        small_dataframes[label] = df[df[label_name] == label]
    return small_dataframes


def data_split(data,label_name: str, num_partitions: int, val_ratio: float = 0.1, IID: bool=True , clusted_in_subdata: int=1):
    '''
    data: input data
    label_name: name of target label in data
    num_partitions: number of Subdataset
    IID: (False for non iid data split)
    val_ratio: rate of validation data
    test_ratio: rate of test data
    '''
    if not IID :
        print("prepairing non IID dataset")
        length=len(data)
        partition_size=int(length/num_partitions)
        partition_len_list = [partition_size] * num_partitions
        print(f"len of client data :{partition_size * num_partitions}, len of each subdata : {partition_size}")
        data = data[:int(partition_size * num_partitions)]
        df_sorted = data.sort_values(by=label_name)
        df=df_sorted.reset_index(drop=True)

        partitions=[]
        start=0
        for num in partition_len_list:
            partitions.append(df[start:start+num])
            start=start+num     
        num=1
        trainloaders = []
        valloaders = []
        for client in partitions:
            len_val = int(val_ratio * len(client))
            len_train = len(client) - len_val
            print(f"client number {num} : train({len_train}), val({len_val})")
            num=num+1
            client = client.sample(frac=1).reset_index(drop=True)
            val_data=client[0:len_val]
            train_data=client[len_val:]
            train_data = train_data.reset_index(drop=True)
            val_data=val_data.reset_index(drop=True)
            trainloaders.append(train_data)
            valloaders.append(val_data)

    if IID:
        print("prepairing IID dataset")
        partition_size = int(len(data) // num_partitions)
        partition_len_list = [partition_size] * num_partitions
        print(f"len of client data :{len(data)}, len of each subdata : {partition_size}")
        df = data[:int(partition_size * num_partitions)]
        partitions=[]
        start=0
        for num in partition_len_list:
            partitions.append(df[start:start+num])
            start=start+num

        trainloaders = []
        valloaders = []
        num=1
        for client in partitions:
            len_val = int(val_ratio * len(client))
            len_train = len(client) - len_val
            print(f"client number {num} : train({len_train}), val({len_val})")
            num=num+1
            val_data = client[0:len_val]
            train_data=client[len_val:]
            train_data = train_data.reset_index(drop=True)
            val_data=val_data.reset_index(drop=True)
            trainloaders.append(train_data)
            valloaders.append(val_data)
    return trainloaders, valloaders

def identify_correlated(df, threshold):
    """
    A function to identify highly correlated features.
    """
    # Compute correlation matrix with absolute values
    #df=df.drop("category",axis=1)
    matrix = df.corr().abs()

    # Create a boolean mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Subset the matrix
    reduced_matrix = matrix.mask(mask)

    # Find cols that meet the threshold
    to_drop = [c for c in reduced_matrix.columns if any(reduced_matrix[c] > threshold)]

    return to_drop

# def processed(data,label_name: str,threshold: int=0.01, corr_threshold=1):
#     label=data[label_name]
#     data=data.drop(label_name,axis=1)
#     data=pd.concat([label,data],axis=1)
#     data.astype(str)
#     df=data.dropna()
#     label_encoder = LabelEncoder()
#     encoded_df = df.apply(label_encoder.fit_transform)
#     #feature selection
#     selector = VarianceThreshold(threshold=threshold)
#     filtered_df = selector.fit_transform(encoded_df)
#     filtered_df = pd.DataFrame(filtered_df, columns=df.columns[selector.get_support()])
#     #normalized
#     normalized_df=filtered_df.mean()
#     normalized_df=filtered_df.drop(identify_correlated(pd.DataFrame(normalized_df),corr_threshold),axis=1)
#     processed_data=normalized_df
#     processed_data=processed_data.sample(frac=1).reset_index(drop=True)
#     return processed_data

def processed(data, label_name: str, threshold: float=0.01, corr_threshold=1):
    # Tách label
    label = data[label_name]
    data = data.drop(label_name, axis=1)
    
    # Tính giá trị trung bình cho cột số và mode cho cột categorical
    mean_values = data.select_dtypes(include=[np.number]).mean()
    mode_values = data.select_dtypes(exclude=[np.number]).mode().iloc[0]
    
    # Thay thế NaN bằng giá trị trung bình hoặc mode
    data.fillna(mean_values, inplace=True)
    data.fillna(mode_values, inplace=True)
    
    # Đưa label trở lại vào DataFrame
    data = pd.concat([label, data], axis=1)
    
    # Chuyển đổi các cột sang kiểu dữ liệu số để encoding
    df = data.apply(lambda x: x.astype(str) if x.dtype == 'object' else x)
    
    # Encoding dữ liệu
    label_encoder = LabelEncoder()
    encoded_df = df.apply(label_encoder.fit_transform)
    
    # Feature selection
    selector = VarianceThreshold(threshold=threshold)
    filtered_df = selector.fit_transform(encoded_df)
    filtered_df = pd.DataFrame(filtered_df, columns=df.columns[selector.get_support()])
    
    # Xử lý correlated features (giả định bạn đã định nghĩa hàm identify_correlated)
    normalized_df = filtered_df.mean()
    normalized_df = filtered_df.drop(identify_correlated(pd.DataFrame(normalized_df), corr_threshold), axis=1)
    
    # Xáo trộn dữ liệu
    processed_data = normalized_df.sample(frac=1).reset_index(drop=True)
    
    return processed_data



def count_label_data(labels):
    '''
    this step use for analyse label
    '''
    label_counts = Counter(labels)
    return label_counts

def analyse_dataset(data,names):
    list=[]
    for name in names:
        label_data_count = count_label_data(data[name])
        print(f"thống kê nhãn {name}:")
        for label, count in label_data_count.items():
            list.append(label)
            print(label, count)
    return list


def group_labels(label_counts, n):
    grouped_labels = {}
    labels = list(label_counts.keys())
    # Duyệt qua từng cặp nhãn
    for i in range(0, len(labels), n):
        group =labels[i:i + n]
        group_key = tuple(group)
        group_count = sum(label_counts[label] for label in group)
        grouped_labels[group_key] = group_count

    return grouped_labels

def set_data(loaders):
    list=[]
    for loader in loaders:
        y=loader["subcategory "]
        X=loader.drop("subcategory ",axis=1)
        print(f"len X {len(X)}, len y {len(y)}")
        list.append((X,y))
    return list
