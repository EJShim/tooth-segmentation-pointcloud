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
renderWindow.SetSize(1000, 1000)
renderWindow.SetFullScreen(True)
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)

txtActor = vtk.vtkTextActor()
txtActor.SetInput("Preparing...")
txtActor.GetTextProperty().SetFontFamilyToArial()
txtActor.GetTextProperty().SetFontSize(46)
txtActor.GetTextProperty().SetColor(1,1,0)
txtActor.SetDisplayPosition(100,900)
renderer.AddActor(txtActor)
#renderWindow.Render()



if __name__ == "__main__":


    #Import Input Data
    input_poly = utils.ReadSTL('./processed/input.stl')
    original_data = []

    for idx in range(input_poly.GetNumberOfPoints()):
        position = input_poly.GetPoint(idx)
        original_data.append(position)
    original_data = np.array(original_data)
    

    #Import Ground-truth data
    with open('./processed/groundtruth', 'rb') as filehandler:
        # read the data as binary data stream
        original_ground_truth = np.array(pickle.load(filehandler))\
    #########################################################################################
    #Subsasmple module




    input_actor = utils.MakeActor(input_poly)
    renderer.AddActor(input_actor)
    
    renderer.GetActiveCamera().Pitch(-30)
    renderer.ResetCamera()
    renderWindow.Render()
    #  iren.Start()



 
    print("start training")


    #make 10 training data
    train_set = utils.make_subsample_data(original_data, original_ground_truth, size=10)
    test_data = utils.make_subsample_data(original_data, original_ground_truth)



    num_point = train_set[0]['input'].shape[0]
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

    for epoch in range(11):

        for data in train_set:

            input_data = data['input']
            gt_data = data['gt']

            [output_data, loss, _] = sess.run([output_data_tensor, loss_op, train_op], feed_dict={input_tensor:[input_data], gt_tensor:[gt_data], is_training:True})

            
            log = str(epoch) + "/" + "10, Loss : " +  str(loss)
            txtActor.SetInput(log)
            utils.update_segmentation(input_poly, output_data[0], data['idx'])
            renderWindow.Render()


        # #run test
        # for data in test_data:
        #     output_data = sess.run(output_data_tensor, feed_dict={input_tensor:[data['input']], is_training:False})                 
        #     utils.update_segmentation(input_poly, output_data[0], data['idx'])
        #     renderWindow.Render()
    
    txtActor.SetInput("Finished")
    txtActor.GetTextProperty().SetColor(0, 1, 0)
    renderWindow.Render()

    iren.Start()