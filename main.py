from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import time
from Module import EncoderDecoderMaster
from Trainer import Trainer
from CustomDataClass import dataset_for_time_series, DataModule
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import json
import torch
import pickle

######################################
#Set an option whether Hyperparameter 
#Optimization schould be included
######################################

parser = argparse.ArgumentParser()
parser.add_argument("--hyper", help="Use Hyperparameter optimization", action="store_true" )
parser.add_argument("--saved", help="Use saved model", action="store_true")

args = parser.parse_args()
    
hyperparameter_optimization = args.hyper
use_savedm_model = args.saved
if hyperparameter_optimization == True:
  print("Hyperparameter optimization:", hyperparameter_optimization)
elif use_savedm_model == True:
    print("Use saved model:", use_savedm_model)



##################################


print('Load the dataframe')

workload_dataframe = pd.read_csv('~/Datasets/alibaba_workload_data.csv')

workload_dataframe = workload_dataframe.fillna(0)


#standardize by substracting the mean and dividing by sdt. deviation. 
standard_scaler = StandardScaler()
standardizedm_data = standard_scaler.fit_transform(workload_dataframe)
workload_data = pd.DataFrame(standardizedm_data, columns=workload_dataframe.columns)

#to do reverse the compression
#original_data_standardized = standard_scaler.inverse_transform(df_standardized)

#get the numbers a bit bigger to avoid dealing with too small values. 



'''
#This does the Min Max Saling
#substract by min value, divide by max value minus in value.
min_max_scaler = MinMaxScaler()
workload_data = pd.DataFrame(min_max_scaler.fit_transform(workload_dataframe), columns=workload_dataframe.columns)

#to reverse the compression
#original_data_minmax = min_max_scaler.inverse_transform(df_normalized)
'''

workload_data = workload_data * 10

workload_data = workload_data[['avg_cpu', 'avg_gpu','avg_memory','avg_new_cpu', 'avg_new_gpu', 'avg_new_memory']]

input_size = workload_data.shape[1]
output_size = 3

############################################
#Do the optimization loop:

if hyperparameter_optimization == True:
    print('Hyperparameter Optimization Loop')
    best_params, best_accuracy = Trainer.hyperparameter_optimization(workload_data, input_size, 3)
    edm_model= EncoderDecoderMaster(input_size, best_params['hidden_size'], 3, best_params['hidden_size_2'], output_size, num_layers = best_params['num_layers'],optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate'], loss_function = best_params['loss'], clip_val = best_params['gradient_clip'], scheduler = best_params['scheduler'])
    trainer = Trainer(60, best_params['batch_size'], early_stopping_patience = 10, window_size = best_params['window_size'])
    trainer.fit(edm_model,workload_data)

    torch.save({
        'state_dict': edm_model.state_dict(),
        'input_size': input_size,
        'hidden_size':best_params['hidden_size'],
        'hidden_size_2':best_params['hidden_size_2'],
        'num_layers': best_params['num_layers'],
        'learning_rate': best_params['learning_rate'], 
        'loss_function': best_params['loss'], 
        'clip_val': best_params['gradient_clip'],
        'optimizer': best_params['optimizer'],
        'scheduler': best_params['scheduler']}, 'eda_state_and_attributes.pth')

    # Serialize with pickle
    with open('EncoderDecoderMasterattention.pkl', 'wb') as f:
        pickle.dump(edm_model, f)



elif use_savedm_model == True:
    print('Use saved model')
    # Load the model's state dictionary

    best_params = torch.load('edm_state_and_attributes.pth')
    edm_model= EncoderDecoderMaster(input_size, best_params['hidden_size'],3, best_params['hidden_size_2'], output_size, num_layers = best_params['num_layers'],optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate'], loss_function = best_params['loss'], clip_val = best_params['gradient_clip'], scheduler = best_params['scheduler'])

    edm_model.load_state_dict(best_params['state_dict'])

    # Deserialize with pickle
    with open('EncoderDecoderMaster.pkl', 'rb') as f:
        edm_model = pickle.load(f)
    trainer = Trainer(window_size=18)
    trainer.prepare_training_data(workload_data)


else:
    print('Train custom model')
    hidden_size = 128
    hidden_size_2 = 8
    num_layers = 1
    optimizer = 'SGD'
    learning_rate = 0.003259
    loss_function = 'Huber'
    scheduler = 'OnPlateau'
    clip_val = 0.0
    l1 = 0.0
    l2 = 0.0
    edm_model= EncoderDecoderMaster(input_size, hidden_size, 3, hidden_size_2, output_size, num_layers, optimizer, learning_rate, loss_function, l1, l2, clip_val , scheduler )
    trainer = Trainer(50, 32, early_stopping_patience = 10, window_size = 15)
    trainer.fit(edm_model,workload_data)
    torch.save({
        'state_dict': edm_model.state_dict(),
        'input_size': input_size,
        'output_size':output_size,
        'hidden_size': hidden_size,
        'hidden_size_2': hidden_size_2,
        'num_layers': num_layers,
        'learning_rate': learning_rate, 
        'loss_function': loss_function, 
        'clip_val': clip_val,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'l1' : l1,
        'l2' : l2}, 'custom_state_and_attributes.pth')

    # Serialize with pickle
    with open('customEncoderDecoderMaster.pkl', 'wb') as f:
        pickle.dump(edm_model, f)



######################################
#Visualize training progress
######################################

if use_savedm_model == False:

    n_epochs = range(trainer.max_epochs)
    train_loss = trainer.train_loss_values
    nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
    train_loss = np.concatenate([train_loss,nan_values])

    val_loss = trainer.val_loss_values
    nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
    val_loss = np.concatenate([val_loss,nan_values])

    plt.figure(figsize=(10,6))
    plt.plot(n_epochs, train_loss, color='blue', label='train_loss' , linestyle='-')
    plt.plot(n_epochs, val_loss, color='orange', label='val_loss' , linestyle='-')
    plt.title("Train and Validation Loss")
    plt.legend()


##############################
#Test the model
##############################


def choose_same_value_again(dataset):
    baseline_predictions = torch.zeros(1,output_size)
    for x, _ in test_data:
        next_value = x[:,-1,0:3]   
        baseline_predictions = torch.cat((baseline_predictions,next_value), dim=0)
    return baseline_predictions


    return y_0, y_1, y_2


y_hat, y = trainer.test(edm_model)

print(f"Test loss is: {float(trainer.test_loss)}")

y_0 = y[:,0]
y_1 = y[:,1]
y_2 = y[:,2]
y_hat_0 = y_hat[:,0]
y_hat_1 = y_hat[:,1]
y_hat_2 = y_hat[:,2]

test_data = trainer.test_dataloader

baseline_predictions = choose_same_value_again(test_data)
m_0 = baseline_predictions[:,0]
m_1 = baseline_predictions[:,1]
m_2 = baseline_predictions[:,2]


x_values = range(len(y_0))

plt.figure()
plt.plot(x_values, y_0, color='blue', label='Test Data' , linestyle='-')
plt.plot(x_values, y_hat_0, color='green', label='Prediction' , linestyle='-')
#plt.plot(x_values, y_0[:-1], color='orange', label='Same value again' , linestyle='-')
plt.title("Avg CPU")
plt.legend()

plt.figure()
plt.plot(x_values, y_1, color='red', label='Test Data' , linestyle='-')
plt.plot(x_values, y_hat_1, color='green', label='Prediction' , linestyle='-')
#plt.plot(x_values, y_1[:-1], color='yellow', label='Same value again' , linestyle='-')
plt.title("Avg GPU")
plt.legend()

plt.figure()
plt.plot(x_values, y_2, color='cyan', label='Test Data' , linestyle='-')
plt.plot(x_values, y_hat_2, color='green', label='Prediction' , linestyle='-')
#plt.plot(x_values, y_2[:-1], color='orange', label='Same value again' , linestyle='-')
plt.title("Avg Memory")
plt.legend()

plt.show()

'''
##################################
#do some iterated predicitons
##################################


iteratedm_data = iter(test_data)
first_batch , _ = next(iteratedm_data)
start = first_batch[0,:,:].unsqueeze(0)
edm_predictions = start[0,:,0:3].squeeze(0)
y = torch.cat((edm_predictions, y), dim=0).detach().numpy()
y_0 = y[:,0]
y_1 = y[:,1]
y_2 = y[:,2]

edm_model.eval()
for i in range(test_data.dataset.len):
    next_prediction = edm_model(start)
    edm_predictions = torch.cat((edm_predictions,next_prediction),dim=0)
    start = torch.cat((start,next_prediction.unsqueeze(0)),dim=1)[:,1:,:]

p_0 = edm_predictions[:,0].detach().numpy()
p_1 = edm_predictions[:,1].detach().numpy()
p_2 = edm_predictions[:,2].detach().numpy()

x_values = range(len(y_0))

plt.figure()
plt.plot(x_values, y_0, color='blue', label='test_data' , linestyle='-')
plt.plot(x_values, p_0, color='green', label='prediction' , linestyle='-')
plt.title("Iterated Predictions CPU")
plt.legend()

plt.figure()
plt.plot(x_values, y_1, color='red', label='test_data' , linestyle='-')
plt.plot(x_values, p_1, color='green', label='prediction' , linestyle='-')
plt.title("Iterated Predictions GPU")
plt.legend()

plt.figure()
plt.plot(x_values, y_2, color='cyan', label='test_data' , linestyle='-')
plt.plot(x_values, p_2, color='green', label='prediction' , linestyle='-')
plt.title("Iterated Predictions Memory")
plt.legend()
plt.show()
'''
