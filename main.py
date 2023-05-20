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
from datetime import datetime

import numpy as np
import torch
import torch.cuda as gpu

# ------------------- LOGGING  -------------------
# By default, log output writes to ./Results/MM-DD-YYY-HH-MM-SS/local_run.log only.
# To output to standard error regardless, specify the STDERR environment variable.
STDERR = os.environ.get('STDERR', '0')  # by default, we don't output to STDERR.

# Change to logging.CRITICAL and only critical errors will be logged:
# See: https://docs.python.org/3/library/logging.html#logging-levels
LOGLVL = logging.DEBUG

resdir = os.path.join(os.getcwd(), "Results", str(datetime.now().strftime("%m-%d-%Y-%H-%M-%S")))
os.makedirs(resdir)

logging.basicConfig(filename=os.path.join(resdir, 'local_run.log'), encoding='utf-8', level=LOGLVL)
#logging.disable('DEBUG') # If this line is uncommented, logging will occur, otherwise nothing will be logged

if STDERR == '1':  # Setup the optional additional logger
    logging.getLogger().addHandler(logging.StreamHandler())

logging.debug('\n\n\n\n--------------------STARTING NEW RUN - %s --------------------', str(datetime.now()))
# ------------------------------------------------

# ------------------- CUDA CHECK SECTION -------------------
useCUDA = True  # CHANGE THIS TO FALSE TO USE TORCH CPU
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

mode="???"

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    logging.debug("Reset trainable parameters of layer = %s", str(layer))
    layer.reset_parameters()


class LoadDataset1(Dataset):
    # Load dataset for a list of 1 or more countries, for 1 or more years, for a specific crop
    def __init__(self, countries=None, years=None, crops=None):
        """
        Load dataset1 for
        @param countries is a list, or "all"
        @param years is a list, a range (represented as a string), an integer, or "all"
        @param crops is a list, or "all"
        """

        if countries==None and years==None and crops==None:
            logging.error("Please specify what country/countries, year/years, and crop to use for dataset")
            return False
        # Imports a Pandas DataFrame object:
        data = pd.read_csv("Datasets/Dataset1/yield_df.csv", header=None, sep=',')
        # NOTE: DataFrame objects (like this one) are referenced by Column then Row, unlike a Matrix
        
        # Drop 1st column from DataFrame, which is index:
        data.drop(columns=0, axis='columns', inplace=True)
        
        logging.info("All Matching Dataset Format Referece (pay attention to indicies) - \n%s", str(data.loc[data.index[0]]))
        logging.info("All Matching Input Dataset \n%s\n", str(data))
        
        # Drop 1st row from DataFrame, which holds labels:
        data.drop(index=0, inplace=True)

        if years == "all":
            if crops == "all":
               
                data=data[
                    (data[1].isin(countries='Albania')) 
                   ]
                    
                print("to impliment")
            else:
                data = data[(data[1].isin(countries)) & (data[2].isin(crops))].reset_index().drop(columns=['index'], axis=1)
        else:
            # ----- get 'years' -----
            if isinstance(years, str) and re.fullmatch(r'\d{4}-\d{4}', years):
                yearsRange = re.fullmatch(r'(\d{4})-(\d{4})', years)
                # years given as an *inclusive* range, e.g. '2004-2016' is start of 2004 to end of 2016
                # yearsRange is a new variable representing the match, or None if no match
                rangeObj = range(int(yearsRange.group(1)), int(yearsRange.group(2))+1)
                # remember that DataFrame object matches on row/col *values*, so you must be careful
                # with indicies, and you cannot pass any int as a row or col label.
                years = [str(x) for x in rangeObj]
            else:
                years = [str(years)]
            # ----- end get 'years' -----
            
            if crops == "all":
                # The case where any crop, for any set of years, for a country in countries makes up the dataset
                data=data[
                    (data[1].isin(countries='Albania')) &
                    (data[3].isin(years=[1996,2004]))]
                logging.warning("to impliment")
            else:
                # select sets for years, crops, and countries
                data = data[(data[1].isin(countries)) &
                            (data[2].isin(crops)) &
                            (data[3].isin(years))].reset_index().drop(columns=['index'], axis='columns').set_axis(range(data.shape[1]), axis='columns')

        logging.info("Parsed Input Dataset \n%s\n", str(data))
        self.datain = data  # finally set self.datain

        logging.debug(f"self.datain[5] => {self.datain[5]}")
        logging.debug(f"self.datain[5][20] => {self.datain[5][20]}")

    def __len__(self):
        return len(self.datain)


    # For now, I've excluded year from the inputs, if we want to expand the dataset then we could start with country 
    # GDP and bring year back in (Hypothesis being, higher GDP, better farming equipment and management, higher yield)
    def __getitem__(self, index):
        x = torch.tensor([
            float(self.datain[5][index]), # rainfall
            float(self.datain[6][index]), # pesticide
            float(self.datain[7][index])  # temp
        ], dtype=torch.float)
        y = torch.tensor(float(self.datain[4][index]), dtype=torch.float) # yield
        return x, y

    def split_test_training(self, test_split):
        # Splitting the datain
        
        self.training_set, self.test_set = np.split(
            self.datain.sample(frac=1, random_state=68), # the data to be split up
            [int((1.0-test_split)*len(self))]            # indices or sections at which array is split
        )
        logging.debug(f"Original data: \n{str(self.datain.sample(frac=1, random_state=68))}")
        logging.debug(f"Training set: \n{self.training_set}\n")
        logging.debug(f"Test set: \n{self.test_set}\n")

    @staticmethod
    def reset_index(dataframe):
        """ Do a proper reset of indices so that the labels on the table match up to where the entries are positionally. """
        return(dataframe.reset_index().drop(columns=['index'], axis=1).set_axis(range(dataframe.shape[1]), axis=1))

    def add_data_labels(self):
        self.training_set = LoadDataset1.reset_index(self.training_set)
        logging.debug("Training Set with labels \n%s\n", str(self.training_set))
        self.test_set = LoadDataset1.reset_index(self.test_set)
        logging.debug("Test Set with labels \n%s\n", str(self.test_set))
        return

    @staticmethod
    def drop_cols(dataframe, cols):
        """ Fully drop columns and subsequently adjust the index """
        return(dataframe.reset_index().drop(columns=cols, axis=1))

    def drop_area_item(self):
        self.training_set = LoadDataset1.drop_cols(self.training_set, [0,1,2, 'index'])
        self.test_set     = LoadDataset1.drop_cols(self.test_set,     [0,1,2, 'index'])
        return

    
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



class CropYieldPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # After super initialize model structures
        self.layers = nn.Sequential(
        nn.Conv1d(3, 6, 5),
        nn.MaxPool2d(2, 2),
        nn.Conv1d(6, 12, 5),
        nn.ReLU(),
        nn.Linear(16 * 5, 120),
        nn.Linear(120, 84),
        nn.Linear(84, 10),
        nn.Linear(10, 1)
        )

    # Provide model path
    def forward(self, x):
        return self.layers(x)



if __name__ == "__main__":
    logging.debug("Running program as standalone")
    
    # K-folds cross-validation implimentation referenced from
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
<<<<<<< HEAD
    k_folds         = 5
    test_split      = 0.15
    training_epochs = 40
    results         = {}
    torch.manual_seed(420) # Might want to comment this, as it will have an effect on k-folds data
    loss_function   = torch.nn.MSELoss() # torch.sqrt() for RMSE

    DS = LoadDataset1(['Bahamas', 'Bangladesh', 'Brazil','Guatemala','Germany'], "1990-2004", ["Maize"])
    # print(DS.__getitem__(5))

    # ----- BEGIN SECTION MIGRATING TO LOADDATASET1 CLASS -----
    # Operations on Test and Training data:
    DS.split_test_training(test_split)  # test and training are in the class now
    DS.add_data_labels()                # Reset the index on test and training so it counts up normally
    
    DS.drop_area_item()                 # Drop columns 0 (index), 1 (area), and 2 (item)
    
    logging.debug("----- RAW DATA -----")
    DS.add_data_labels()                # We reset the index again
    
    # NOTE: Here's where Eric left off. -----
    training_set = DS.training_set
    test_set = DS.test_set

    logging.debug("Training Set \n" + str(training_set) + "\n")
    logging.debug("Test Set \n" + str(test_set) + "\n")
    
    # NOTE: Saw this dissappear when merging. Are we going to forego the training set dict?
    # ----- BEGIN TRAINING SET DICT -----
    #training_set_xs = training_set[[training_set.columns[1],training_set.columns[2],training_set.columns[3]]].to_numpy().astype(np.float32)
    #training_set_ys = training_set[training_set.columns[0]].to_numpy().astype(np.float32)
    # region Expanded Loop now uses list comnprehention
    #trainingdict = {}
    # for row in range(0,len(training_set_xs)):
    #     trainingdict[row] = {"x": training_set_xs[row], "y":training_set_ys[row]}
    # endregion Expanded Loop now uses list comnprehention
    #trainingsetdict = {row: {"x": training_set_xs[row], "y": training_set_ys[row]} for row in range(len(training_set_xs))}
    #logging.debug("Training Set dictionary \n" + re.sub("(.{82})", "\\1\n", str(trainingsetdict), 0, re.DOTALL) + "\n")
    #test_set_xs     = test_set[[test_set.columns[1],test_set.columns[2],test_set.columns[3]]].to_numpy().astype(np.float32)
    #test_set_ys     = test_set[test_set.columns[0]].to_numpy().astype(np.float32)
    #testsetdict     = {row: {"x": test_set_xs[row], "y": test_set_ys[row]} for row in range(len(test_set_xs))}
    #logging.debug("Test Set dictionary \n" + re.sub("(.{82})", "\\1\n", str(testsetdict), 0, re.DOTALL) + "\n")
    # ----- END TRAINING SET DICT -----
    
    combinationset = ConcatDataset([training_set, test_set])

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Starting Model Training and Evaluation Here
    logging.debug("Starting model training and evaluation over %s cross-validation folds.", str(kfold))

    # TODO - John, Eric, Dominic - finish K-fold cross-validation 

    for fold, (train, test) in enumerate(kfold.split(combinationset)):
        print("to impliment")
        print(train)
        print(test)
        logging.debug("Fold %s", str(fold))

        # Batch size = 
        train_subsampler = torch.utils.data.SubsetRandomSampler(train)
        trainloader = DataLoader(combinationset, batch_size=1, sampler=train_subsampler)

        test_subsampler = torch.utils.data.SubsetRandomSampler(test) 
        testloader = DataLoader(combinationset, batch_size=1, sampler=test_subsampler)

        model = CropYieldPredictionModel()
        model.apply(reset_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

        for epoch in range(0, training_epochs):
            logging.debug("########## Epoch " + str(epoch+1) + " ##########")

            current_loss = 0.0

            print(len(combinationset))
            print(len(trainloader))
            print(len(testloader))

            print(next(iter(trainloader)))

            for i, data in enumerate(trainloader, 0): # THIS LINE CAUSING AN ERROR FROM TRAINLOADER

                print(i)

                inputs, target = data
                if useCUDA:
                    inputs, target = inputs.cuda(), target.cuda()

                optimizer.zero_grad()

                output = model(inputs)

                loss = loss_function(output, target)

                loss.backward()

                optimizer.step()

                current_loss += loss.item()
                if i % 5 == 0:
                    print("Loss after mini-batch" + str(i+1) + "-" + str(current_loss / 5))
                    current_loss = 0.0





    exit()

    # Training Set
    DLTrain = torch.utils.data.DataLoader(DS, batch_size=12, shuffle=True)

    # Test Set
    DLTest = torch.utils.data.DataLoader(DS, batch_size=12, shuffle=True)

    
    
    

    # Model Training
    model.train()
    for epoch in range(training_epochs):
        for X_batch, y_batch in DLT:
            if useCUDA:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
    # Test model
    model.eval()
    y_pred = model(X_test)
    acc = (y_pred.round() == y_test).float().mean()


else:
   logging.debug("File imported.")

