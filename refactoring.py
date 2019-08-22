import vtk
import os
import pickle
import utils
import numpy as np
import network
import tensorflow as tf
from datetime import datetime
import temp_util

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




def get_model(input_tensor, is_training, bn_decay = None):    
    weight_decay = 0.0
    num_point = input_tensor.get_shape()[1].value
    k = 10


    #Transform Net
    adj_matrix = temp_util.pairwise_distance(input_tensor)    
    nn_idx = temp_util.knn(adj_matrix, k=k)
    
    edge_feature = temp_util.get_edge_feature(input_tensor, nn_idx=nn_idx, k=k)

    


    out1_1 = temp_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training, weight_decay=weight_decay,
                       scope='one/adj_conv1', bn_decay=bn_decay, is_dist=True)
    
    
    out1_2 = temp_util.conv2d(out1_1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='one/adj_conv2', bn_decay=bn_decay, is_dist=True)

        
    out1_3 = temp_util.conv2d(out1_2, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='one/adj_conv3', bn_decay=bn_decay, is_dist=True)

    net_1 = tf.reduce_max(out1_3, axis=-2, keepdims=True)

    




    adj = temp_util.pairwise_distance(net_1)
    nn_idx = temp_util.knn(adj, k=k)

    
    edge_feature = temp_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)    

    

    

    out2_1 = temp_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='two/adj_conv1', bn_decay=bn_decay, is_dist=True)

    out2_2 = temp_util.conv2d(out2_1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='two/adj_conv2', bn_decay=bn_decay, is_dist=True)

    out2_3 = temp_util.conv2d(out2_2, 64, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training, weight_decay=weight_decay,
                            scope='two/adj_conv3', bn_decay=bn_decay, is_dist=True)
                            
    net_2 = tf.reduce_max(out2_3, axis=-2, keepdims=True)
      

    adj = temp_util.pairwise_distance(net_2)
    nn_idx = temp_util.knn(adj, k=k)
    edge_feature = temp_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

    out3_1 = temp_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='three/adj_conv1', bn_decay=bn_decay, is_dist=True)


    out3_2 = temp_util.conv2d(out3_1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training, weight_decay=weight_decay,
                        scope='three/adj_conv2', bn_decay=bn_decay, is_dist=True)


    net_3 = tf.reduce_max(out3_2, axis=-2, keepdims=True)



    out7 = temp_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], 
                        padding='VALID', stride=[1,1],
                        bn=True, is_training=is_training,
                        scope='adj_conv7', bn_decay=bn_decay, is_dist=True)


    out_max = temp_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')    


    expand = tf.tile(out_max, [1, num_point, 1, 1])    

    concat = tf.concat(axis=3, values=[expand, 
                                        net_1,
                                        net_2,
                                        net_3])

    # CONV 
    net = temp_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1],
                bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
    net = temp_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    
    net = temp_util.conv2d(net, 16, [1,1], padding='VALID', stride=[1,1],
                activation_fn=None, scope='seg/output', is_dist=True)


    net = tf.squeeze(net, [2])


    return net


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
    train_set = utils.make_subsample_data(original_data, original_ground_truth, size=1, sample_size=sample_size)

    sample_data = train_set[0]['input']

    print(sample_data.shape)

    
    input_tensor = tf.placeholder(tf.float32, shape=(None, sample_data.shape[0], sample_data.shape[1]), name="target_input")
    gt_tensor = tf.placeholder(tf.int32, shape=(None, sample_data.shape[0]))
    print(input_tensor, gt_tensor)
    
    is_training = tf.placeholder(tf.bool, name="target_isTraining")
    output_tensor = get_model(input_tensor, is_training)
    

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
        # save_path = "./weights/epoch_" + str(epoch)
        # builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        # builder.add_meta_graph_and_variables(sess, ['ejshim'])
        # builder.save()
        #while True:
        for data in train_set:
            
            input_data = data['input']
            gt_data = data['gt']

            [output_data, loss, _] = sess.run([output_data_tensor, loss_op, train_op], feed_dict={input_tensor:[input_data], gt_tensor:[gt_data], is_training:True})

            
            log = str(epoch) + "/" + str(max_epoch-1) + ", Loss : " +  str(loss)
            txtActor.SetInput(log)
            utils.update_segmentation(input_poly, output_data[0], data['idx'])
            renderWindow.Render()


        #run test
        # for data in test_data:
        #     output_data = sess.run(output_data_tensor, feed_dict={input_tensor:[data['input']], is_training:False})                 
        #     utils.update_segmentation(input_poly, output_data[0], data['idx'])
        #     renderWindow.Render()
    
    txtActor.SetInput("Finished")
    txtActor.GetTextProperty().SetColor(0, 1, 0)
    renderWindow.Render()

    iren.Start()