import vtk
import os
import pickle
import utils
import numpy as np
from bson import ObjectId
import pymongo
import gridfs

db = pymongo.MongoClient().maxilafacial
fileDB = gridfs.GridFS(db)
sample_size = 4096


if __name__ == "__main__":
    #Generate Training Data Here!
    patients = db.patient.find({})

    for element in patients:
        patientID = element["_id"]

        if not element['status']: continue
        
        train_data = utils.make_training_data(cl, size=100, sample_size = sample_size)
        
        input_data = []
        gt_data = []

        for data in train_data:
            input_data.append(data['input'])
            gt_data.append(data['gt'])
        
        #Save By Patient ID
        np.savez_compressed(os.path.join('processed',str(patientID)), input=input_data, gt=gt_data)


