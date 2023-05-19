import sys
import logging
import numpy as np
from datetime import datetime


logging.basicConfig(filename='local.log', encoding='utf-8', level=logging.DEBUG)
#logging.disable('DEBUG') # If this line is uncommented, logging will occur, otherwise nothing will be logged
logging.debug('\n\n\n\n--------------------STARTING NEW RUN - '+ str(datetime.now()) + ' --------------------')

# ------------------- CUDA CHECK SECTION -------------------
useCUDA = False # CHANGE THIS TO FALSE TO USE TORCH CPU
import torch
import testtorch


if useCUDA:
    res = testtorch.test_CUDA()
    logging.debug(res[1])
    if not res[0]:
        sys.exit("ERROR - CUDA ACCELERATION SELECTED, BUT FAILED. If you do not want to use CUDA acceleration set useCUDA = False in main.py")
else:
    logging.debug("Using Torch CPU")
logging.debug(testtorch.test_version())
# ----------------------------------------------------------

import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import KFold

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
            sys.exit("ERROR, please specify what country/countries, year/years, and crop to use for dataset")

        self.datain = pd.read_csv("Datasets/Dataset1/yield_df.csv", header=None, sep=',')
        self.datain = self.datain.drop(columns=[0], axis=1)
        logging.debug("All Matching Dataset Format Referece (pay attention to indicies) - \n" + str(self.datain.loc[self.datain.index[0]]))
        logging.debug("All Matching Input Dataset \n" + str(self.datain) + "\n")
        self.datain = self.datain.drop(index=0)

        if years == "all":
            if crops == "all":
               
                self.datain=self.datain[
                    (self.datain[1].isin(countries='Albania')) 
                   ]
                    
                print("to impliment")
            else:
                self.datain = self.datain[(self.datain[1].isin(countries)) & (self.datain[2].isin(crops))].reset_index().drop(columns=['index'], axis=1)
        else:
            if crops == "all":
               
                self.datain=self.datain[
                    (self.datain[1].isin(countries='Albania')) &
                    (self.datain[3].isin(years=[1996,2004]))]
                print("to impliment")
            else:
                # select sets for years, crops, and countries
                self.datain = self.datain[
                    (self.datain[1].isin(countries)) &
                    (self.datain[2].isin(crops)) &
                    (self.datain[3].isin(years))
                ].reset_index().drop(columns=['index'], axis=1)

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
    k_folds = 5
    test_split = 0.15
    training_epochs = 40
    results = {}
    torch.manual_seed(420) # Might want to comment this, as it will have an effect on k-folds data
    loss_function = torch.nn.MSELoss() # torch.sqrt() for RMSE

    DS = LoadDataset1(['Albania'], "all", ["Maize"])
    # print(DS.__getitem__(5))

    training_set, test_set =  np.split(DS.datain.sample(frac=1, random_state=68), [int((1.0-test_split)*len(DS))])
    training_set = training_set.reset_index().drop(columns=['index'], axis=1).set_axis(range(training_set.shape[1]), axis=1)
    test_set     = test_set.reset_index().drop(columns=['index'], axis=1).set_axis(range(test_set.shape[1]), axis=1)
    logging.debug("Training Set \n" + str(training_set) + "\n")
    logging.debug("Test Set \n" + str(test_set) + "\n")
    
    combinationset = ConcatDataset([training_set, test_set])

    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Starting Model Training and Evaluation Here
    logging.debug("Starting model training and evaluation over " + str(kfold) + " cross-validation folds.")

    # TODO - John, Eric, Dominic - finish K-fold cross-validation 

    for fold, (train, test) in enumerate(kfold.split(combinationset)):
        print("to impliment")
        print(train)
        print(test)
        logging.debug("Fold " + str(fold))

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

