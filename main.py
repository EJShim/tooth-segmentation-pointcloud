import vtk
import os
import pickle
import utils
import numpy as np
import network, tf_util
import tensorflow as tf
from datetime import datetime

# Initialize RenderWIndow
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1200, 1200)
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)



if __name__ == "__main__":


    #Import Input Data
    input_poly = utils.ReadSTL('./processed/input.stl')
    input_data = []

    for idx in range(input_poly.GetNumberOfPoints()):
        position = input_poly.GetPoint(idx)
        input_data.append(position)
    input_data = np.array(input_data)
    

    #Import Ground-truth data
    with open('./processed/groundtruth', 'rb') as filehandler:
        # read the data as binary data stream
        ground_truth_data = np.array(pickle.load(filehandler))\
    #########################################################################################
    #Subsasmple module
    
    print(input_data.shape, ground_truth_data.shape)

    subsample_idx = np.random.choice( np.arange(input_data.shape[0]), 32768 )

    subsample_input = []
    subsample_gt = []

    for idx in subsample_idx:
        subsample_input.append( input_data[idx] )
        subsample_gt.append( ground_truth_data[idx] )

    subsample_input = np.array(subsample_input)
    subsample_gt = np.array(subsample_gt)

    subsample_polydata = utils.PoitncloudToMesh(subsample_input)
    ##########################################################################################

    #Make Actor, Visualize
    input_actor = utils.MakeActor(subsample_polydata)
    input_actor.GetProperty().SetPointSize(1)
    renderer.AddActor(input_actor)
 
    print("start training")

    num_point = subsample_input.shape[0]
    input_tensor = tf.placeholder(tf.float32, shape=(1, num_point, 3))
    gt_tensor = tf.placeholder(tf.int32, shape=(1, num_point))
    is_training = tf.placeholder(tf.bool)
    output_tensor = network.get_model(input_tensor, is_training)
    output_data_tensor  = tf.cast(tf.greater(output_tensor[:,:,1], output_tensor[:,:,0]), tf.float32, name="output_ejshim")
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=output_tensor)
    loss_op = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(1e-4, 0.9)
    train_op = optimizer.minimize(loss_op)
    


    ############################################################
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    starttime = datetime.now()

    for i in range(100):
        [output_data, loss, _] = sess.run([output_data_tensor, loss_op, train_op], feed_dict={input_tensor:[subsample_input], gt_tensor:[subsample_gt], is_training:True})


        print(loss)
        utils.Visualize_segmentation(subsample_polydata, output_data[0])
        renderWindow.Render()




    
    



    iren.Start()


    







    
    

