#!/usr/bin/env python
"""
ML based Crop Yield Prediction By Country
ECE 597/697 ML - UMass Amherst, Spring 2023

@author Domenic McArthur<dmcarthur@umass.edu>
@author Eric Webster<ewebster@umass.edu>
@author John Murray<jomurray@umass.edu>
"""

import re
import os
import sys
import logging
import numpy as np
from datetime import datetime

resdir = "Results/"+str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

os.mkdir(resdir)
logging.basicConfig(filename=resdir+'/local_run.log', encoding='utf-8', level=logging.DEBUG)
#logging.disable('DEBUG') # If this line is uncommented, logging will occur, otherwise nothing will be logged
logging.debug('\n\n\n\n--------------------STARTING NEW RUN - '+ str(datetime.now()) + ' --------------------')


# ------------------- CUDA CHECK SECTION -------------------
if gpu.is_available():
    logging.debug("[GPU Devices Available]")
    for dev in range(gpu.device_count()):
        logging.debug("Name: %s %s", gpu.get_device_name(dev),
                      "(current)" if (gpu.current_device() == dev) else None)
elif useCUDA:
    logging.error("CUDA ACCELERATION SELECTED, BUT FAILED. If you do not want to use CUDA acceleration set useCUDA = False in main.py")
else:
    logging.debug("Using Torch CPU version: %s", torch.__version__)

# ----------------------------------------------------------



import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import KFold


#print(torch.randn(4, 3, 5))

# R  = Rainfall
# P = pesticides
# T = Temperature

# [ R, P, T ]

#[
# [ R, P, T ]
# [ P, R, T ]     Possible input modifications
# [ T, R, P ] 
# ]


mode="???"

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    logging.debug("Reset trainable parameters of layer = " + str(layer))
    layer.reset_parameters()


class LoadDataset1(Dataset):
    # Load dataset for a list of 1 or more countries, for 1 or more years, for a specific crop
    def __init__(self, countries=None, years=None, crops=None):
        # Load and prune dataset of unnessesary data

        if countries==None and years==None and crops==None:
            logging.error("Please specify what country/countries, year/years, and crop to use for dataset")
            return False

        self.datain = pd.read_csv("Datasets/Dataset1/yield_df.csv", header=None, sep=',')
        self.datain = self.datain.drop(columns=[0], axis=1)
        logging.debug("All Matching Dataset Format Referece (pay attention to indicies) - \n" + str(self.datain.loc[self.datain.index[0]]))
        logging.debug("All Matching Input Dataset \n" + str(self.datain) + "\n")
        self.datain = self.datain.drop(index=0)

        if years == "all":
            if crops == "all":
                # TODO - Domenic, add the case where any crop, for any year, for a country in countries makes up the dataset
                print("to impliment")
            else:
                self.datain = self.datain[(self.datain[1].isin(countries)) & (self.datain[2].isin(crops))].reset_index().drop(columns=['index'], axis=1).set_axis(range(self.datain.shape[1]), axis=1)
        else:
            if crops == "all":
                # TODO - Domenic, add the case where any crop, for any set of years, for a country in countries makes up the dataset
                print("to impliment")
            else:
                # select sets for years, crops, and countries
                self.datain = self.datain[
                    (self.datain[1].isin(countries)) &
                    (self.datain[2].isin(crops)) &
                    (self.datain[3].isin(years))
                ].reset_index().drop(columns=['index'], axis=1).set_axis(range(self.datain.shape[1]), axis=1)

        logging.debug("Parsed Input Dataset \n" + str(self.datain) + "\n")


    def __len__(self):
        return len(self.datain)


    # For now, I've excluded year from the inputs, if we want to expand the dataset then we could start with country 
    # GDP and bring year back in (Hypothesis being, higher GDP, better farming equipment and management, higher yield)
    def __getitem__(self, indx):
        x = torch.tensor([float(self.datain[5][indx]), float(self.datain[6][indx]), float(self.datain[7][indx])], dtype=torch.float)
        y = torch.tensor(float(self.datain[4][indx]), dtype=torch.float)
        return x, y

        # region Oldstuff

        # DONT NEED THIS PROCESSING, IT TAKES A LONG TIME AND YOU CAN JUST FORMAT AND PASS THINGS WITH THE __get_item__ function and pandas .loc/.loci
        # self.structured_data_pairs = {}
        # countries = self.datain[1].unique()
        # n = 1
        # for country in countries:
        #     full_country_production = self.datain.loc[self.datain[1] == country]
        #     country_years = full_country_production[3].unique()
        #     self.structured_data_pairs[country] = {}
        #     for year in country_years:
        #         full_country_production_year = full_country_production.loc[self.datain[3] == year]
        #         country_crop = full_country_production_year[2].unique()
        #         self.structured_data_pairs[country][year] = {}
        #         for crop in country_crop:
        #             full_country_production_year_crop = full_country_production_year.loc[self.datain[2] == crop]
        #             self.structured_data_pairs[country][year][crop] = {"average_rain_fall_mm_per_year":full_country_production_year_crop[5][n], "pesticides_tonnes":full_country_production_year_crop[6][n], "avg_temp":full_country_production_year_crop[7][n], "hg/ha_yield":full_country_production_year_crop[4][n]}
        #             n+=full_country_production_year_crop.shape[0]
        #             #print(str(self.structured_data_pairs))
        # logging.debug("Loaded Dataset to Dictionary containing... \n \
        #              # Countries = " + str(len(self.structured_data_pairs)))
        # There are countries without rain levels in here, could maybe be used as a test set, weakly relating similar nearby countries with a model
        # self.extra = pd.read_csv("Datasets/Dataset1/yield.csv", header=None, sep=',')
        # self.extra = self.extra.drop(columns=[0,1,2,4,5,6,7,8], axis=1)
        # logging.debug("Non-Mathcing Dataset Results Format Referece (pay attention to indicies) - \n" + str(self.extra.loc[self.extra.index[0]]))
        # logging.debug("Non-Mathcing Dataset Results \n" + str(self.extra)+ "\n")
        # self.extra = self.extra.drop(index=0)

        # endregion Oldstuff

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        MSE_loss = nn.MSELoss()
        RMSE_loss = torch.sqrt(MSE_loss(x, y))
        return RMSE_loss

class CropYieldPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=420, kernel_size=3, bias=True).cuda()
        self.zeropad2 = nn.ZeroPad2d((1, 1, 0, 0)).cuda()
        self.conv2 = nn.Conv1d(in_channels=420, out_channels=120, kernel_size=3, bias=True).cuda()
        self.conv3 = nn.Conv1d(in_channels=120, out_channels=60, kernel_size=3, bias=True).cuda()
        self.conv4 = nn.Conv1d(in_channels=60, out_channels=1, kernel_size=3, bias=True).cuda()

    # Provide model path
    def forward(self, x):
        #print("Layer 1 input" + str(x))
        x = self.zeropad2(torch.nn.functional.relu(self.conv1(x)))
        #print("Layer 2 input" + str(x))
        x = self.zeropad2(torch.nn.functional.relu(self.conv2(x)))
        #print("Layer 3 input" + str(x))
        x = self.zeropad2(torch.nn.functional.relu(self.conv3(x)))
        #print("Layer 4 input" + str(x))
        x = torch.nn.functional.relu(self.conv4(x))
        #print(x)
        return x



if __name__ == "__main__":
    logging.debug("Running program as standalone")
    
    # K-folds cross-validation implimentation referenced from
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
    k_folds = 4
    test_split = 0.15
    training_epochs = 400
    results = {}
    torch.manual_seed(420) # Might want to comment this, as it will have an effect on k-folds data
    loss_function = RMSELoss()

    DS = LoadDataset1(['Bahamas', 'Bangladesh', 'Brazil','Guatemala','Germany'], "all", ["Maize"])
    # print(DS.__getitem__(5))

    # region TODO - MOVE INTO DATASET1 CLASS

    training_set, test_set =  np.split(DS.datain.sample(frac=1, random_state=68), [int((1.0-test_split)*len(DS))])
    training_set = training_set.reset_index().drop(columns=['index'], axis=1).set_axis(range(training_set.shape[1]), axis=1)
    test_set     = test_set.reset_index().drop(columns=['index'], axis=1).set_axis(range(test_set.shape[1]), axis=1)
    logging.debug("Training Set with labels \n" + str(training_set) + "\n")
    logging.debug("Test Set with labels \n" + str(test_set) + "\n")

    training_set = training_set.reset_index().drop(columns=[0,1,2,'index'], axis=1)
    test_set     = test_set.reset_index().drop(columns=[0,1,2, 'index'], axis=1)
    training_set = training_set.set_axis(range(training_set.shape[1]), axis=1)
    test_set     = test_set.set_axis(range(test_set.shape[1]), axis=1)
    logging.debug("Training Set raw \n" + str(training_set) + "\n")
    logging.debug("Test Set raw \n" + str(test_set) + "\n")

    training_set_xs = training_set[[training_set.columns[1],training_set.columns[2],training_set.columns[3]]].to_numpy().astype(np.float32)
    training_set_ys = training_set[training_set.columns[0]].to_numpy().astype(np.float32)
    # region Expanded Loop now uses list comnprehention
    trainingdict = {}
    # for row in range(0,len(training_set_xs)):
    #     trainingdict[row] = {"x": training_set_xs[row], "y":training_set_ys[row]}
    # endregion Expanded Loop now uses list comnprehention
    trainingsetdict = {row: {"x": training_set_xs[row], "y": training_set_ys[row]} for row in range(len(training_set_xs))}
    logging.debug("Training Set dictionary \n" + re.sub("(.{82})", "\\1\n", str(trainingsetdict), 0, re.DOTALL) + "\n")
    test_set_xs     = test_set[[test_set.columns[1],test_set.columns[2],test_set.columns[3]]].to_numpy().astype(np.float32)
    test_set_ys     = test_set[test_set.columns[0]].to_numpy().astype(np.float32)
    testsetdict     = {row: {"x": test_set_xs[row], "y": test_set_ys[row]} for row in range(len(test_set_xs))}
    logging.debug("Test Set dictionary \n" + re.sub("(.{82})", "\\1\n", str(testsetdict), 0, re.DOTALL) + "\n")
    
    # endregion TODO - MOVE INTO DATASET1 CLASS

    combinationset = ConcatDataset([trainingdict, testsetdict])

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Starting Model Training and Evaluation Here
    logging.debug("Starting model training and evaluation over " + str(kfold) + " cross-validation folds.")

    for fold, (train, test) in enumerate(kfold.split(combinationset)):
        
        # Batch size = 
        train_subsampler = torch.utils.data.SubsetRandomSampler(train)
        trainloader = DataLoader(combinationset, batch_size=12, sampler=train_subsampler)

        test_subsampler = torch.utils.data.SubsetRandomSampler(test) 
        testloader = DataLoader(combinationset, batch_size=12, sampler=test_subsampler)

        model = CropYieldPredictionModel()
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("Training Model for Fold "+ str(fold))
        logging.debug("Training Model for Fold "+ str(fold))

        for epoch in range(0, training_epochs):
            print("Fold - " + str(fold) +  " Epoch - " + str(epoch+1))
            logging.debug("########## Fold - " + str(fold) +  " Epoch - " + str(epoch+1) + " ##########")
            current_loss = 0.0

            for i, data in enumerate(trainloader, 0): # THIS LINE CAUSING AN ERROR FROM TRAINLOADER

                inputs = data["x"]
                inputs = inputs.unsqueeze(1)
                targets = data["y"]
                # print(i)
                # print(data)
                # print(inputs)
                # print(targets)
                if useCUDA:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                
                #print(targets)

                output = model(inputs)
                #print(output)

                loss = loss_function(output, targets)

                loss.backward()

                optimizer.step()

                current_loss += loss.item()
                if i % 5 == 0:
                    logging.debug("RMSE Loss after batch member " + str(i+1) + " = " + str(current_loss / 5))
                    current_loss = 0.0

        print("Training Model for Fold "+ str(fold)+" completed, saving model.")
        logging.debug("Training Model for Fold "+ str(fold)+" completed, saving model.")

        save_path = resdir+"/model-fold-"+str(fold)+".pth"
        torch.save(model.state_dict(), save_path)
        
        print("Testing Model for Fold "+ str(fold))
        logging.debug("Testing Model for Fold "+ str(fold))

        # TODO - John, Eric, Dominic - finish K-fold cross-validation 05/18/2023 DO IT HERE BOI fix it up

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):

                # Get inputs
                inputs = data["x"][0].unsqueeze(0)
                inputs = inputs.unsqueeze(1)
                targets = data["y"]

                if useCUDA:
                    inputs, targets = inputs.cuda(), targets.cuda()

                # Generate outputs
                outputs = model(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)

    # Print fold results
  
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
    
    # Test model
    #model.eval()
    #y_pred = model(X_test)
    #acc = (y_pred.round() == y_test).float().mean()


else:
   logging.debug("File imported.")

