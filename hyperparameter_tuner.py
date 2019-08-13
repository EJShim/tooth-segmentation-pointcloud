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
    input_poly = utils.ReadSTL('./processed/temp.stl')
    utils.sort_pointIndex(input_poly)


    original_data = []

    for idx in range(input_poly.GetNumberOfPoints()):
        position = np.array(input_poly.GetPoint(idx))
        original_data.append(position)
    original_data = np.array(original_data)

    original_data = utils.normalize_input_data(original_data)


    #Import Ground-truth data
    with open('./processed/temp_gt', 'rb') as filehandler:
        # read the data as binary data stream
        original_ground_truth = np.array(pickle.load(filehandler))
    #########################################################################################
    #Subsasmple module


 

    input_actor = utils.MakeActor(input_poly)
    renderer.AddActor(input_actor)
    
    renderer.GetActiveCamera().Pitch(-45)
    renderer.ResetCamera()
    renderWindow.Render()
    #  iren.Start()


    sample_size = 32768

    print("Initialization!")
 


    #make 10 training data
    train_set = utils.make_subsample_data(original_data, original_ground_truth, size=100, sample_size=sample_size)
    test_data = utils.make_subsample_data(original_data, original_ground_truth, sample_size=sample_size)

        

    num_point = train_set[0]['input'].shape[0]
    input_tensor = tf.placeholder(tf.float32, shape=(1, num_point, 3), name="target_input")
    gt_tensor = tf.placeholder(tf.int32, shape=(1, num_point))
    is_training = tf.placeholder(tf.bool, name="target_isTraining")
    output_tensor = network.get_model(input_tensor, is_training)

    output_data_tensor = tf.argmax(output_tensor, 2, name="target_output")

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=output_tensor)
    loss_op = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(1e-4, 0.5)
    train_op = optimizer.minimize(loss_op)
    


    ############################################################
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    
    print("start training")

    max_epoch = 31

    for epoch in range(max_epoch):
        #Save
        save_path = "./weights/epoch_" + str(epoch)
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(sess, ['ejshim'])
        builder.save()
        #while True:
        for data in train_set:
            
            input_data = data['input']
            gt_data = data['gt']

            [output_data, loss, _] = sess.run([output_data_tensor, loss_op, train_op], feed_dict={input_tensor:[input_data], gt_tensor:[gt_data], is_training:True})

            
            log = str(epoch) + "/" + str(max_epoch-1) + ", Loss : " +  str(loss)
            txtActor.SetInput(log)
            #utils.update_segmentation(input_poly, output_data[0], data['idx'])
            renderWindow.Render()


        #run test
        for data in test_data:
            output_data = sess.run(output_data_tensor, feed_dict={input_tensor:[data['input']], is_training:False})                 
            utils.update_segmentation(input_poly, output_data[0], data['idx'])
            renderWindow.Render()
    
    txtActor.SetInput("Finished")
    txtActor.GetTextProperty().SetColor(0, 1, 0)
    renderWindow.Render()

    iren.Start()