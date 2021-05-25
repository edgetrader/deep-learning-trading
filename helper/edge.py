import os, zipfile
import datetime
import joblib

from datetime import date
from tensorflow import keras
from dateutil.relativedelta import relativedelta

def getFilelist(foldername):
    filelist = []

    for dirname, _, filenames in os.walk(foldername):
        for filename in filenames:
            filelist.append(os.path.join(dirname, filename))

    return filelist

def createfolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)    

def unzipfolder(folder, req_ext='.csv'):
    zipfilelist = [filename for filename in getFilelist(folder) if filename.endswith('.zip')]

    for filename in zipfilelist: # loop through items in dir
        zip_ref = zipfile.ZipFile(filename) # create zipfile object
        zip_ref.extractall(folder) # extract file to dir
        zip_ref.close() # close file

    removeList = [filename for filename in getFilelist(folder) if not filename.endswith(req_ext)]
    for filename in removeList:
        os.remove(filename) # delete zipped file    

def getTimeStamp():
    # Get Timestamp as a String
    out = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return out

def save_ml_model(model, filename):
    # Save to file in the current working directory
    joblib.dump(model, filename)

def load_ml_model(filename):    
    # Load from file
    joblib_model = joblib.load(filename)
    return joblib_model

def save_dl_model(model, name):
    model.save(name)
    model.save_weights(name + "_wgts.hdf5")

def load_dl_model(name):
    model = keras.models.load_model(name)
    model.load_weights(name + "_wgts.hdf5")
    return model    
