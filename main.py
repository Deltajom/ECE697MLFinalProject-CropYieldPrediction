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
import matplotlib.pyplot as pl

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

        logging.debug(f"self.datain[5] => \n{self.datain[5]}")
        logging.debug(f"self.datain[5][20] => \n{self.datain[5][20]}")

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

    def add_data_labels(self, log=False):
        self.training_set = LoadDataset1.reset_index(self.training_set)
        self.test_set = LoadDataset1.reset_index(self.test_set)
        if(log):
            logging.debug("Training Set \n%s\n", str(self.training_set))
            logging.debug("Test Set \n%s\n", str(self.test_set))
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
        if(useCUDA):
            # self.zeropad2 = nn.ZeroPad2d((1, 1, 0, 0)).cuda()
            # self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, bias=True).cuda()
            # self.conv2 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=3, bias=True).cuda()
            # self.l1 = nn.Linear(in_features=96*3, out_features=240, bias=True).cuda()
            # self.l2 = nn.Linear(in_features=240, out_features=120, bias=True).cuda()
            # self.l3 = nn.Linear(in_features=120, out_features=60, bias=True).cuda()
            # self.l4 = nn.Linear(in_features=60, out_features=30, bias=True).cuda()
            # self.l5 = nn.Linear(in_features=30, out_features=1, bias=True).cuda()
            self.zeropad2 = nn.ZeroPad2d((1, 1, 0, 0)).cuda()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, bias=True).cuda()
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=42, kernel_size=3, bias=True).cuda()
            self.l1 = nn.Linear(in_features=42*3, out_features=120, bias=True).cuda()
            self.l2 = nn.Linear(in_features=120, out_features=60, bias=True).cuda()
            self.l3 = nn.Linear(in_features=60, out_features=30, bias=True).cuda()
            self.l4 = nn.Linear(in_features=30, out_features=15, bias=True).cuda()
            self.l5 = nn.Linear(in_features=15, out_features=1, bias=True).cuda() 

        else:
            self.zeropad2 = nn.ZeroPad2d((1, 1, 0, 0))
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=60, kernel_size=3, bias=True)
            self.conv2 = nn.Conv1d(in_channels=60, out_channels=20, kernel_size=3, bias=True)
            self.conv3 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, bias=True)
            self.conv4 = nn.Conv1d(in_channels=10, out_channels=5, kernel_size=3, bias=True)
            self.conv5 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3, bias=True)

    # Provide model path
    def forward(self, x):
        #print("Layer 1 input" + str(x))
        x = self.zeropad2(torch.nn.functional.relu(self.conv1(x)))
        #print("Layer 2 input" + str(x))
        x = self.zeropad2(torch.nn.functional.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        #print("Layer 3 input" + str(x))
        x = torch.nn.functional.relu(self.l1(x))
        #print("Layer 4 input" + str(x))
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))
        x = torch.nn.functional.relu(self.l4(x))
        x = self.l5(x)
        #print("Layer 5 input" + str(x))
        #x = torch.nn.functional.relu(self.conv5(x))
        #print(x)
        return x



if __name__ == "__main__":
    logging.debug("Running program as standalone")
    
    # K-folds cross-validation implimentation referenced from
    # https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
    k_folds = 6
    test_split = 0.1
    training_epochs = 2500
    results = {}
    torch.manual_seed(420) # Might want to comment this, as it will have an effect on k-folds data
    loss_function = RMSELoss()

    DS = LoadDataset1(['Argentina', 'Colombia', 'Brazil','Chile','El Salvador'], "all", ["Maize"])
    # print(DS.__getitem__(5))

    # ----- BEGIN SECTION MIGRATING TO LOADDATASET1 CLASS -----
    # Operations on Test and Training data:
    DS.split_test_training(test_split)  # test and training are in the class now
    DS.add_data_labels()                # Reset the index on test and training so it counts up normally
    
    DS.drop_area_item()                 # Drop columns 0 (index), 1 (area), and 2 (item)
    
    logging.debug("----- RAW DATA -----")
    DS.add_data_labels(log=True)                # We reset the index again & log
    
    # NOTE: Here's where Eric left off. -----
    training_set = DS.training_set
    test_set = DS.test_set
    
    # NOTE: Saw this dissappear when merging. Are we going to forego the training set dict? 
    # NOTE: ANSWER - NO we have to use the dictionaries as the concat dataset on raw DS objects is causing key errors in our case
    # ----- BEGIN TRAINING SET DICT -----
    training_set_xs = training_set[[training_set.columns[1],training_set.columns[2],training_set.columns[3]]].to_numpy().astype(np.float32)
    training_set_ys = training_set[training_set.columns[0]].to_numpy().astype(np.float32)
    #region Expanded Loop now uses list comnprehention
    #trainingsetdict = {}
    #for row in range(0,len(training_set_xs)):
    #    trainingdict[row] = {"x": training_set_xs[row], "y":training_set_ys[row]}
    #endregion Expanded Loop now uses list comnprehention
    trainingsetdict = {row: {"x": training_set_xs[row], "y": training_set_ys[row]} for row in range(len(training_set_xs))}
    logging.debug("Training Set dictionary \n" + re.sub("(.{82})", "\\1\n", str(trainingsetdict), 0, re.DOTALL) + "\n")
    test_set_xs     = test_set[[test_set.columns[1],test_set.columns[2],test_set.columns[3]]].to_numpy().astype(np.float32)
    test_set_ys     = test_set[test_set.columns[0]].to_numpy().astype(np.float32)
    testsetdict     = {row: {"x": test_set_xs[row], "y": test_set_ys[row]} for row in range(len(test_set_xs))}
    logging.debug("Test Set dictionary \n" + re.sub("(.{82})", "\\1\n", str(testsetdict), 0, re.DOTALL) + "\n")

    # ----- END TRAINING SET DICT -----
    combinationset = ConcatDataset([trainingsetdict, testsetdict])

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Starting Model Training and Evaluation Here
    logging.debug("Starting model training and evaluation over %s cross-validation folds.", str(kfold))
    
    for fold, (train, test) in enumerate(kfold.split(combinationset)):
        # Batch size = 
        lossarray=[]
        train_subsampler = torch.utils.data.SubsetRandomSampler(train)
        trainloader = DataLoader(combinationset, batch_size=8, sampler=train_subsampler)

        test_subsampler = torch.utils.data.SubsetRandomSampler(test) 
        testloader = DataLoader(combinationset, batch_size=8, sampler=test_subsampler)

        model = CropYieldPredictionModel()
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.6e-4)

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

                if useCUDA:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()

                outputs = model(inputs)
                
                # print("Flattened out" + str(outputs.flatten()))

                # print("Targets" + str(targets))

                loss = loss_function(outputs.flatten(), targets)
                
                

                loss.backward()
                
                optimizer.step()

                current_loss += loss.item()
            
                if i % 5 == 0:
                    logging.debug("RMSE Loss after batch member " + str(i+1) + " = " + str(current_loss / 5))
                    current_loss = 0.0
            lossarray.append(current_loss)
        epochs=np.arange(0,len(lossarray))
        pl.plot(epochs,lossarray,'r')
        pl.ylabel('Loss')
        pl.xlabel('Epoch')
        pl.title('Loss vs Epoch Fold - '+str(fold))
        pl.savefig(os.path.join(resdir,'Lossgraphfold'+str(fold)+".png"))
        print("Training Model for Fold "+ str(fold)+" completed, saving model.")
        logging.debug("Training Model for Fold "+ str(fold)+" completed, saving model.")
        pl.cla()
        save_path = resdir+"/model-fold-"+str(fold)+".pth"
        torch.save(model.state_dict(), save_path)
        
        print("Testing Model for Fold "+ str(fold))
        logging.debug("Testing Model for Fold "+ str(fold))

        # Evaluation for this fold
        allelementpercenterrors = []
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            for i, data in enumerate(testloader, 0):
                
                inputs = data["x"]
                inputs = inputs.unsqueeze(1)
                targets = data["y"]

                if useCUDA:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = model(inputs)

                # Find average percengate error, where %error = (| Experimental measurements - Actual measurements | / Actual measurements)*100 (element-wise) sum these and 
                # divide by number of inputs
                runavgerror = torch.mul(torch.div(torch.abs(torch.sub(outputs.flatten(), targets)), targets),100)
                allelementpercenterrors.extend(runavgerror)


            print('Overall average percent error %d: %d %%' % (fold, (torch.sum(torch.tensor(allelementpercenterrors))/len(allelementpercenterrors))))
            print('--------------------------------')
            logging.debug('Overall average percent error %d: %d %%' % (fold, (torch.sum(torch.tensor(allelementpercenterrors))/len(allelementpercenterrors))))
            logging.debug('--------------------------------')
            results[fold] = (torch.sum(torch.tensor(allelementpercenterrors))/len(allelementpercenterrors))
   
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')

    logging.debug(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    logging.debug('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Multirun average percent error: {sum/len(results.items())} %')
    logging.debug(f'Multirun average percent error: {sum/len(results.items())} %')

else:
   logging.debug("File imported.")
