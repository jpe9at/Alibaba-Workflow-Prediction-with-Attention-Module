import torch
import torch.nn as nn
import time
from CustomDataClass import dataset_for_time_series, DataModule
import optuna
from Module import EncoderDecoderMaster


class Trainer: 
    """The base class for training models with data."""
    def __init__(self, max_epochs = 50, batch_size = 8, early_stopping_patience=6, min_delta = 0.09, num_gpus=0, window_size = 10):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.num_epochs_no_improve = 0
        self.min_delta = min_delta
        assert num_gpus == 0, 'No GPU support yet'
        self.window_size = window_size

    def prepare_training_data(self, workload_data):
        
        #create the training/val/testsplit
        split_index = int(len(workload_data) * 0.8)  # Split at 80% of the data
        split_index2 = int(split_index + ((len(workload_data) - split_index)/2))

        #do not use the full dataset if the sizes are too big.
        #h = int(split_index * 0.78)
        #h2 = split_index + int(split_index2 * 0.7)


        #Split the time series data into training and testing sets
        print('Prepare the labels')
        train_data_X, train_data_y = dataset_for_time_series(workload_data[:split_index], self.window_size)
        val_data_X, val_data_y = dataset_for_time_series(workload_data[split_index:split_index2], self.window_size)
        test_data_X, test_data_y = dataset_for_time_series(workload_data[split_index2:-2], self.window_size)

        print('Prepare the train_dataset')
        data_train = DataModule(train_data_X, train_data_y)

        print('Prepare the val_dataset')
        data_val = DataModule(val_data_X, val_data_y) 

        print('Prepare the test_dataset')
        data_test = DataModule(test_data_X, test_data_y)

        self.train_dataloader = data_train.get_dataloader(self.batch_size)
        self.val_dataloader = data_val.get_dataloader(self.batch_size)
        self.test_dataloader = data_test.get_dataloader(self.batch_size)
    
    def prepare_test_data(self, data_test):
        #if test data is added after the training 
        test_data_X, test_data_y = dataset_for_time_series(data_test, self.window_size)
        data_test = DataModule(test_data_X, val_data_y)
        self.test_dataloader = data_test.get_dataloader(batch_size)
    
    def prepare_model(self, model):
        model.trainer = self
        self.model = model
    
    def fit(self, model, dataset):
        self.train_loss_values = []
        self.val_loss_values = []
        self.prepare_training_data(dataset)
        self.prepare_model(model)
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss, val_loss = self.fit_epoch()
            if (epoch+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.max_epochs}], Train_Loss: {train_loss:.4f}, Val_Loss: {val_loss: .4f}, LR = {self.model.scheduler.get_last_lr() if self.model.scheduler is not None else self.model.learning_rate}')
            self.train_loss_values.append(train_loss)
            self.val_loss_values.append(val_loss)

            #########################################
            #Early Stopping Monitor
            #instead, we can also use the early stopping monitor class below. 
            if (self.best_val_loss - val_loss) > self.min_delta:
                self.best_val_loss = val_loss
                self.num_epochs_no_improve = 0
            else:
                self.num_epochs_no_improve += 1
                if self.num_epochs_no_improve == self.early_stopping_patience:
                    print("Early stopping at epoch", epoch)
                    break
            ########################################

            ########################################
            #Scheduler for adaptive learning rate
            if self.model.scheduler is not None:
                self.model.scheduler.step(val_loss)
            ########################################


    def fit_epoch(self):
        train_loss = 0.0
        total_batches = len(self.train_dataloader)
        #torch.autograd.set_detect_anomaly(True)
        for idx, (x_batch, y_batch) in enumerate(self.train_dataloader):
            additional = y_batch[:,-1,3:]
            output = self.model(x_batch, additional)
            loss = self.model.loss(output, y_batch[:,-1,0:3])
            self.model.optimizer.zero_grad()
            loss.backward()
            
            ######################################
            #L1 Loss
            if self.model.l1_rate != 0: 
                loss = self.model.l1_regularization(self.model.l2_rate)
            ######################################
            
            ######################################
            #Gradient Clipping
            if self.model.clip_val !=0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.clip_val)  # Gradient clipping
            ######################################

            self.model.optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
            time.sleep(0.1)  # Simulate batch processing time
            
            # Calculate progress
            progress = (idx + 1) / total_batches * 100
            print(f"\rBatch {idx + 1}/{total_batches} completed. Progress: {progress:.2f}%", end='', flush=True)

        train_loss /= len(self.train_dataloader.dataset)
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in self.val_dataloader:
                additional = y_batch[:,-1,3:]
                val_output = self.model(x_batch, additional)
                loss = self.model.loss(val_output, y_batch[:,-1,0:3])
                val_loss += loss.item() * x_batch.size(0) #why multiplication with 0?
            val_loss /= len(self.val_dataloader.dataset)
        return train_loss, val_loss

    def test(self, model, data_test = None):
        model.eval()
        self.test_loss = 0.0
        if data_test == None:
            try:
                self.test_dataloader is not None
            except NameError:
                print('No test dataset specified')
        else: 
            self.prepare_test_data(data_test)

        y_hat_total = torch.zeros(1,3)
        y_total = torch.zeros(1,3)
        with torch.no_grad():
            for X,y in self.test_dataloader:
                additional = y[:,-1,3:]
                y_hat = model(X, additional)  # Choose the class with highest probability
                y_total = torch.cat((y_total,y[:,-1,0:3]), dim=0)
                y_hat_total = torch.cat((y_hat_total,y_hat), dim=0)
                loss = model.loss(y_hat, y[:,-1,0:3])
                self.test_loss += loss * X.size(0)
        self.test_loss /= len(self.test_dataloader.dataset)

        return y_hat_total[1:], y_total[1:]

    def calculate_accuracy(self,predictions, labels):
        # Get the predicted classes by selecting the index with the highest probabilityc
        _, predicted_classes = torch.max(predictions, 0)
        # Compare predictions with ground truth
        correct_predictions = torch.eq(predicted_classes, labels).sum().item()
        # Calculate accuracy
        accuracy = correct_predictions / labels.size(0)
        return accuracy

    
    @classmethod
    def Optuna_objective(cls, trial, workload_data, input_size, output_size):
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        learning_r = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [32,128,256])
        hidden_size = trial.suggest_categorical('hidden_size',[32,64,128])
        l2_rate = trial.suggest_categorical('l2_rate', [0.0,0.0001,0.005])
        loss_function = trial.suggest_categorical('loss', ['MSE','Huber'])
        window_size = trial.suggest_categorical('window_size', [10,15,20])
        gradient_clip = trial.suggest_categorical('gradient_clip', [0.0,1.0])
        scheduler = trial.suggest_categorical('scheduler', [None, 'OnPlateau'])
        num_layers = trial.suggest_categorical('num_layers', [1,2])

        model = EncoderDecoder(input_size, hidden_size, output_size, num_layers, learning_rate = learning_r, loss_function = loss_function, clip_val = gradient_clip, scheduler = scheduler)
        trainer = cls(30,  batch_size, window_size = window_size)
        trainer.fit(model, workload_data)

        return  trainer.val_loss_values[-1]

    @classmethod
    def hyperparameter_optimization(cls, workload_data, input_size,output_size):
        study = optuna.create_study(direction='minimize')
        objective_func = lambda trial: cls.Optuna_objective(trial, workload_data, input_size, output_size)
        study.optimize(objective_func, n_trials=30)

        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value

        return best_params, best_accuracy



