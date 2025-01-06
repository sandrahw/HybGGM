import os
import glob
import numpy as np

os.getcwd()
# Define the directory path to check
directory_path = "/eejit/home/hausw001/HybGGM/hybGGM_test/results/testing/random_sampling_180/"  # Replace with the path to your main directory

# List to store folders that do not meet the criteria
folders_to_nopred = []
folders_twoevents = []
folders_nomodel = []

folderlist = glob.glob(directory_path + '/*')
# remove folders with 'plots' in the name
folderlist = [f for f in folderlist if 'plots' not in f]
# remove folder that don't have UNet in the name
folderlist = [f for f in folderlist if 'UNet' in f]

for folder in folderlist[:]:
    print(folder)
    if not os.path.exists(folder + '/y_pred_denorm_new.npy'):
        folders_to_nopred.append(folder)
    if len(glob.glob(folder + '/events.out*')) > 1:
        folders_twoevents.append(folder)
    if not os.path.exists(folder + '/best_model.pth'):
        folders_nomodel.append(folder)

# get the folder names and hyperparameter specifics
folder_names = [os.path.basename(f) for f in folders_to_nopred]
hyperparams = [f.split('/')[-1].split('_') for f in folders_to_nopred]
np.save('/eejit/home/hausw001/HybGGM/hybGGM_test/src/model/folders_to_nopred_hyperparams.npy', hyperparams)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:25:10 2024

@author: niko
"""
import csv
import pandas as pd
import glob
import numpy as np

def checkCSV(fileName):
  cnt = 0
  with open(fileName) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='$')
    for row in csv_reader:
      try:
        vals = row[0].split(" ")
        if vals[7] == "done" and vals[8] == "at":
          cnt += 1
      except:
        pass
  modelCheck = "Best model saved!"
  validationCheck = "Val Loss"
  predictionCheck = "Hyperparameter tuning prediction finished"
  modPass = [False] * cnt
  valPass = [False] * cnt
  predPass = [False] * cnt
  model = []
  sample = []
  epoch = []
  learning = []
  batch = []
  size = []
  cnt = 0
  with open(fileName) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='$')
    for row in csv_reader:
      try:
        if row[0] == modelCheck:
          modPass[cnt] = True
      except:
        pass
      try:
        if row[0][:8] == validationCheck:
          valPass[cnt] = True
      except:
        pass
      try:
        if row[0] == predictionCheck:
          predPass[cnt] = True
      except:
        pass
      try:
        vals = row[0].split(" ")
        if vals[7] == "done" and vals[8] == "at":
          model.append(vals[0])
          sample.append(vals[2])
          epoch.append(vals[3])
          learning.append(vals[4])
          batch.append(vals[5])
          size.append(vals[6])
          cnt += 1
      except:
        pass
  df = pd.DataFrame({'modelCheck':modPass, 'validationCheck':valPass, 'predictionCheck':predPass
                     , 'model':model, 'sample': sample, 'epoch': epoch
                     , 'learning': learning, 'batch': batch, 'size': size},
                    columns=['modelCheck','validationCheck','predictionCheck', 'model', 'sample','epoch','learning','batch','size'])
  return(df)

files = glob.glob("/Users/niko/Downloads/outputfiles/*.out")

totalOut = checkCSV(files[0])
totalOut.at[0, "modelCheck"] = False

for fileName in files[1:]:
  out = checkCSV(fileName)
  start = totalOut.index.values[-1] + 1
  end = len(out) + start
  out = out.set_index(np.arange(start,end))
  totalOut = pd.concat([totalOut, out])