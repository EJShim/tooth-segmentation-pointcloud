import vtk
import os
import pickle
import utils
import numpy as np
import network, tf_util
import tensorflow as tf
from datetime import datetime
from bson import ObjectId
import pymongo
import gridfs
import random
import pickle


db = pymongo.MongoClient().maxilafacial
fileDB = gridfs.GridFS(db)



#Graph Definition
num_point = 1024 
input_tensor = tf.placeholder(tf.float32, shape=(1, num_point, 3), name="target_input")
gt_tensor = tf.placeholder(tf.int32, shape=(1, num_point))
is_training = tf.placeholder(tf.bool, name="target_isTraining")
output_tensor = network.get_model(input_tensor, is_training)

output_data_tensor = tf.argmax(output_tensor, 2, name="target_output")
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=output_tensor)
loss_op = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(1e-4, 0.5)
train_op = optimizer.minimize(loss_op)




if __name__ == "__main__":

    utils.make_training_data(ObjectId("5cd38c139b1fd2e2fcb3a61f"))
    exit()
    patients = db.patient.find({})


    count = 0


    #Make Training Data
    input_patient_id = []
    for patient in patients:
        input_patient_id.append(patient['_id'])

    input_patient_id = [input_patient_id[1]]
    

    max_epoch = 500




    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    for epoch in range(max_epoch):

        #Save
        if epoch % 10 == 0:
            save_path = "./weights/epoch_" + str(epoch)
            builder = tf.saved_model.builder.SavedModelBuilder(save_path)
            builder.add_meta_graph_and_variables(sess, ['ejshim'])
            builder.save()


        #Shuffle input data
        random.shuffle(input_patient_id)

        for patientID in input_patient_id:

            single_batch = utils.make_training_data(patientID)
            
            input_data = single_batch['input']
            gt_data = single_batch['gt']


            [loss, _] = sess.run([loss_op, train_op], feed_dict={input_tensor:[input_data], gt_tensor:[gt_data], is_training:True})

            print("epoch", epoch, ", Loss :", loss, "ID : ", patientID)