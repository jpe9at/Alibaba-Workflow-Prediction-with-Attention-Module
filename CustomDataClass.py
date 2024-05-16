import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import date2num


#for timeseries predictions: 
def dataset_for_time_series(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+lookback:i+lookback+1]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)


def merge_dfs(df_1,df_2):
    merged_df = pd.concat([df_1, df_2], axis=0, ignore_index=True)
    shuffled_df = merged_df.sample(frac=1).reset_index(drop=True)
    return shuffled_df

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class DataModule: 
    """The base class of data."""
    def __init__(self, X,y):
        self.dataset = Data(X,y)

    def get_dataloader(self, batch_size, num_workers=4):
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=None, pin_memory=False)

    def train_dataloader(self, batch_size):
        return self.get_dataloader(batch_size)

    def val_dataloader(self, batch_size):
        #here say something about gradients not required ? 
        return self.get_dataloader(batch_size)

def custom_timestamp_to_datetime(timestamp):
    if np.isnan(timestamp) != True:
        custom_epoch = datetime(2023, 1, 1, 0, 0, 0)
        # Convert the custom timestamp to datetime object
        return custom_epoch + timedelta(seconds=timestamp)
    else: 
        return pd.NaT

def custom_datetime_to_timestep2(datetimeob):
    if type(datetimeob) != type(pd.NaT):
        if datetimeob.second > 30: 
            datetimeob2 = datetimeob.replace(second=30, microsecond=0)
        else: 
            datetimeob2 = datetimeob.replace(second=0, microsecond=0)
        # Convert the custom timestamp to datetime object
        number = datetimeob2.timestamp() // 30
        return number
    else: 
        return pd.NaT

def custom_datetime_to_timestep(datetimeob):
    if type(datetimeob) != type(pd.NaT):
        replace_value = (datetimeob.minute // 5  ) * 5
        datetimeob2 = datetimeob.replace(minute = replace_value, second=0, microsecond=0)
        number = datetimeob2.timestamp() // (60 * 5)
        return number
    else: 
        return pd.NaT

def moving_average(data, window_size):
    """Calculate the moving average of a time series."""
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return moving_avg


#this works only because every timestep has a workload
def calc_workload_per_timestep(workload_type):
    start_time = time.time()
   
   #initialise workload dictionary with key= timestep, value= vector of current and new workload
    workload_dict = {}
    for item in workload_type:
        key_for_new_res = 'new_' + item
        workload_dict[item] = {}
        workload_dict[key_for_new_res] = {}

    for index, row in df.iterrows():
        if np.isnan(row['scheduled_time']) !=True: 
            for key in workload_type:
                beginning = int(row['scheduled_time'])
                duration = int(row['duration'])
                key_for_new_res = 'new_' + key
                workload_dict[key_for_new_res][beginning] = workload_dict[key_for_new_res].get(beginning, 0) + row[key]
                for timestep in range(beginning, beginning + duration):
                    workload_dict[key][timestep] = workload_dict[key].get(timestep, 0) + row[key]
   
    mediate_time = time.time()
    print(mediate_time - start_time)
    
    for i in range(len(workload_dict['gpu_milli'])):
        for key in workload_dict.keys():
            if 'new' in key:
                   workload_dict[key][i] = workload_dict[key].get(i,0)  

    end_time = time.time()
    computing_time = end_time - mediate_time
    print(computing_time)

    return workload_dict


if __name__ == '__main__': 

    df = pd.read_csv('~/Alibaba Programs/openb_pod_list_default.csv')
                     
    df['creation_time'] = df['creation_time'].apply(custom_timestamp_to_datetime)
    df = df[df['creation_time'].dt.dayofyear > 115]
    df['deletion_time'] = df['deletion_time'].apply(custom_timestamp_to_datetime)
    df['scheduled_time'] = df['scheduled_time'].apply(custom_timestamp_to_datetime)
    
    #prepare data for choosen interval
    df['creation_time'] = df['creation_time'].apply(custom_datetime_to_timestep)
    df['deletion_time'] = df['deletion_time'].apply(custom_datetime_to_timestep)
    df['scheduled_time'] = df['scheduled_time'].apply(custom_datetime_to_timestep)

    #set starting point to 1
    norm = df['creation_time'].iloc[0] - 1
    df['creation_time'] = df['creation_time'] - norm 
    df['deletion_time'] = df['deletion_time']  - norm
    df['scheduled_time'] = df['scheduled_time']  - norm


    df['duration'] = df.apply(lambda row: row['deletion_time'] - row['scheduled_time'] if pd.isna(row['scheduled_time']) != True else pd.NA, axis = 1)

    workload_dict = calc_workload_per_timestep(['cpu_milli', 'gpu_milli', 'memory_mib']) 
    
    workload_dataframe = pd.DataFrame(workload_dict)
    
    end_time = time.time()

    window_avg = 15 
    workload_dataframe['avg_cpu'] = moving_average(workload_dataframe.cpu_milli, window_avg)
    workload_dataframe['avg_gpu'] = moving_average(workload_dataframe.gpu_milli, window_avg)
    workload_dataframe['avg_memory'] = moving_average(workload_dataframe.memory_mib, window_avg)
    
    window_avg = 15
    workload_dataframe['avg_new_cpu'] = moving_average(workload_dataframe.new_cpu_milli, window_avg)
    workload_dataframe['avg_new_gpu'] = moving_average(workload_dataframe.new_gpu_milli, window_avg)
    workload_dataframe['avg_new_memory'] = moving_average(workload_dataframe.new_memory_mib, window_avg)

    workload_dataframe.to_csv('~/Datasets/Alibaba_workload_data.csv', index=False)

    x_axis= range(len(workload_dataframe))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, workload_dataframe['avg_new_cpu'], color='red', label='avg_new_cpu' , linestyle='-')
    plt.plot(x_axis, workload_dataframe['new_cpu_milli'], color='cyan', label='new_cpu_workload' , linestyle='-', alpha = 0.3)
    plt.title('New CPU Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(x_axis,  workload_dataframe['avg_new_gpu'], color='red', label='avg_new_gpu' , linestyle='-')
    plt.plot(x_axis,  workload_dataframe['new_gpu_milli'], color='blue', label='new_gpu_workload' , linestyle='-', alpha = 0.3)
    plt.title('New GPU Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(x_axis,  workload_dataframe['new_memory_mib'], color='purple', label='new_memory_workload' , linestyle='-', alpha = 0.3)
    plt.plot(x_axis,  workload_dataframe['avg_new_memory'], color='red', label='avg_new_mib' , linestyle='-')
    plt.title('New Memory Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, workload_dataframe['cpu_milli'], color='blue', label='cpu_workload' , linestyle='-', alpha = 0.3)
    plt.plot(x_axis, workload_dataframe['avg_cpu'], color='red', label='avg_cpu' , linestyle='-')
    plt.title('CPU Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(x_axis,  workload_dataframe['gpu_milli'], color='green', label='gpu_workload' , linestyle='-', alpha = 0.3)
    plt.plot(x_axis,  workload_dataframe['avg_gpu'], color='red', label='avg_gpu' , linestyle='-')
    plt.title('GPU Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(10,6))
    plt.plot(x_axis,  workload_dataframe['avg_memory'], color='red', label='avg_mib' , linestyle='-')
    plt.plot(x_axis,  workload_dataframe['memory_mib'], color='cyan', label='memory_workload' , linestyle='-', alpha = 0.3)
    plt.title('Memory Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    #plt.show()


    ####################################
    #Summary Statistics
    ####################################
    df = pd.read_csv('~/Alibaba Programs/openb_pod_list_default.csv')
    job_counts = {} 
    workload_per_minute= {}
    cpu_workload_per_minute = {}
    num_of_gpu_per_minute = {}
    memory_mib_per_minute = {}

    df['creation_time'] = df['creation_time'].apply(custom_timestamp_to_datetime)
    
    for index, row in df.iterrows():
        item = row['creation_time']
        workload = row['gpu_milli']
        cpu_workload = row['cpu_milli']
        num_of_gpu = row['num_gpu']
        memory_mib = row['memory_mib']

        item_minute = item.replace(microsecond=0)
        job_counts[item_minute] = job_counts.get(item_minute, 0) + 1
        num_of_gpu_per_minute[item_minute] = num_of_gpu_per_minute.get(item_minute, 0) + num_of_gpu
        memory_mib_per_minute[item_minute] = memory_mib_per_minute.get(item_minute, 0) + memory_mib
        workload_per_minute[item_minute] = workload_per_minute.get(item_minute, 0) + workload
        cpu_workload_per_minute[item_minute] = cpu_workload_per_minute.get(item_minute, 0) + cpu_workload

    # Extract data for plotting
    minutes = date2num(list(job_counts.keys()))
    counts = list(job_counts.values())
    workload = list(workload_per_minute.values())
    cpu_workload = list(cpu_workload_per_minute.values())
    memory_mib = list(memory_mib_per_minute.values())
    num_of_gpu = list(num_of_gpu_per_minute.values())


    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(minutes, workload, linestyle='-', label = 'gpu')
    plt.plot(minutes, memory_mib, linestyle='-', color = 'yellow', label = 'cpu')
    plt.plot(minutes, cpu_workload, color = 'red')
    plt.title('Workload per minute')
    plt.xlabel('Time')
    plt.ylabel('Workload')
    plt.grid(True)
    plt.legend()

    plt.figure(figsize=(15, 6))
    plt.scatter(workload, cpu_workload)
    plt.xlabel('gpu')
    plt.ylabel('cpu')
    plt.title('Correlation between work gpu and cpu workload')

    plt.figure(figsize=(15, 6))
    plt.scatter(num_of_gpu, workload)
    plt.xlabel('num_gpu')
    plt.ylabel('gpu')
    plt.title('Correlation between work num_gpu and gpu workload')

    plt.figure(figsize=(15, 6))
    plt.plot(minutes, counts, 'o', color = 'green')
    plt.plot(minutes, num_of_gpu, 'o', color = 'blue')
    plt.title('jobs per minute')
    plt.xlabel('Time')
    plt.ylabel('jobs')
    plt.grid(True)


    ############################
    #Average stats per hour
    ############################

    df['minute'] = df.creation_time.dt.minute
    df['hour'] = df.creation_time.dt.hour
    df['num_gpu'] = df['num_gpu'].fillna(0)
    hourly_num_gpu_stats = df.groupby('hour')['num_gpu'].agg(['mean', 'std'])
    minutly_num_gpu_stats = df.groupby('minute')['num_gpu'].agg(['mean', 'std'])
    hourly_gpu_stats = df.groupby('hour')['gpu_milli'].agg(['mean', 'std'])


    plt.figure(figsize=(15,6))
    plt.boxplot(df.groupby('hour')['gpu_milli'].apply(list), positions=hourly_gpu_stats.index, showmeans=True)
    plt.errorbar(hourly_gpu_stats.index, hourly_gpu_stats['mean'], yerr=hourly_gpu_stats['std'], fmt='o', markersize=4, capsize=5)
    plt.xlabel('Hour of the day')
    plt.ylabel('Average Value')
    plt.title('Average Value with Standard Deviation by Hour')
    plt.grid(True)

    plt.figure(figsize=(15,6))
    plt.boxplot(df.groupby('hour')['num_gpu'].apply(list), positions=hourly_num_gpu_stats.index, showmeans=True)
    plt.errorbar(hourly_num_gpu_stats.index, hourly_num_gpu_stats['mean'], yerr=hourly_num_gpu_stats['std'], fmt='o', markersize=4, capsize=5)
    plt.xlabel('Hour of the day')
    plt.ylabel('Average Value')
    plt.title('Average Value with Standard Deviation by Hour')
    plt.grid(True)
    plt.show()


