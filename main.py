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
num_point = 4096 
batch_size = 4

input_tensor = tf.placeholder(tf.float32, shape=(None, num_point, 3), name="target_input")
gt_tensor = tf.placeholder(tf.int32, shape=(None, num_point))
is_training = tf.placeholder(tf.bool, name="target_isTraining")
output_tensor = network.get_model(input_tensor, is_training)

output_data_tensor = tf.argmax(output_tensor, 2, name="target_output")
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=output_tensor)
loss_op = tf.reduce_mean(loss)
#Add Regularization
regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
loss_op = loss_op + 0.001*regularizer
optimizer = tf.train.AdamOptimizer(1e-4, 0.5)
train_op = optimizer.minimize(loss_op)


def apply_random_rotation(input_batch):
    print(input_batch.shape)


    transform = vtk.vtkTransform()
    print(transform.GetMatrix())
    exit()



if __name__ == "__main__":

    data_path_list = os.listdir("processed")


    input_data = []
    gt_data = []
    
    for data_name in data_path_list:
        data_path = os.path.join("processed", data_name)
        data_loaded = np.load(data_path)
        input_data.append(data_loaded['input'])
        gt_data.append(data_loaded['gt'])

    input_data = np.concatenate(input_data)
    gt_data = np.concatenate(gt_data)        

    print(input_data.shape, gt_data.shape)
    



    sess = tf.InteractiveSession()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    

    max_epoch = 100

    for epoch in range(max_epoch):

        #Save
    
        save_path = "./weights/epoch_" + str(epoch)
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(sess, ['ejshim'])
        builder.save()
        
        #Make Shuffle
        p = np.random.permutation(input_data.shape[0])
        input_data = input_data[p]
        gt_data = gt_data[p]


        #for idx, input_batch in enumerate(input_data):
        idx = 0
        #for data in train_set:
        while idx < len(input_data):
            
            input_batch = input_data[idx:idx+batch_size]
            gt_batch = gt_data[idx:idx+batch_size]
            idx = idx+batch_size


            [loss, _] = sess.run([loss_op, train_op], feed_dict={input_tensor:input_batch, gt_tensor:gt_batch, is_training:True})

            print("epoch", epoch, ", Loss :", loss)