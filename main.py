import vtk
import os
import pickle
import utils
import numpy as np
import network, tf_util
import tensorflow as tf

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
    input_actor = utils.MakeActor(input_poly)
    input_data = []

    for idx in range(input_poly.GetNumberOfPoints()):
        position = input_poly.GetPoint(idx)
        input_data.append(position)
    input_data = np.array(input_data)
    

    renderer.AddActor(input_actor)
    renderWindow.Render()


    #Import Ground-truth data
    with open('./processed/groundtruth', 'rb') as filehandler:
        # read the data as binary data stream
        ground_truth_data = np.array(pickle.load(filehandler))
    
    utils.Visualize_segmentation(input_poly, ground_truth_data)

    print(input_data.shape, ground_truth_data.shape)

    renderWindow.Render()
    print("this is ground-truth")
    iren.Start()

    print("start training")

    num_point = input_data.shape[0]
    input_tensor = tf.placeholder(tf.float32, shape=(1, num_point, 3))
    gt_tensor = tf.placeholder(tf.int32, shape=(1))
    is_training = tf.placeholder(tf.bool)


    output_tensor = network.get_model(input_tensor, is_training)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    output = sess.run(output_tensor, feed_dict={input_tensor:[input_data], is_training:False})

    
    

