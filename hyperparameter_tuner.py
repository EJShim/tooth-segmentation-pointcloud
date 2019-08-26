import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import os
import sys
import pickle
import utils
import numpy as np
import network, tf_util
import tensorflow as tf
from datetime import datetime
from threading import Thread
from time import sleep
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *





sample_size = 4096
batch_size = 4
    
#Import Input Data
input_poly = utils.ReadSTL('./temp/temp.stl')
utils.sort_pointIndex(input_poly)


original_data = []

for idx in range(input_poly.GetNumberOfPoints()):
    position = np.array(input_poly.GetPoint(idx))
    original_data.append(position)
original_data = np.array(original_data)

original_data = utils.normalize_input_data(original_data)


#Import Ground-truth data
with open('./temp/temp_gt', 'rb') as filehandler:
    # read the data as binary data stream
    original_ground_truth = np.array(pickle.load(filehandler))
#########################################################################################






class TrainingThread(QThread):

    signal_log = pyqtSignal(str)

    def __init__(self, parent=None):
        super(TrainingThread, self).__init__(parent)

        
        self.max_epoch = 31

    
    def run(self):
        
        self.signal_log.emit("Making training Data")
        #make 10 training data
        train_set = utils.make_subsample_data(original_data, original_ground_truth, size=100, sample_size=sample_size)
        test_data = utils.make_subsample_data(original_data, original_ground_truth, sample_size=sample_size)

            
        self.signal_log.emit("Initializing Graph...")
        num_point = train_set[0]['input'].shape[0]
        input_tensor = tf.placeholder(tf.float32, shape=(None, num_point, 3), name="target_input")
        gt_tensor = tf.placeholder(tf.int32, shape=(None, num_point))
        is_training = tf.placeholder(tf.bool, name="target_isTraining")
        output_tensor = network.get_model(input_tensor, is_training)

        output_data_tensor = tf.argmax(output_tensor, 2, name="target_output")

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=output_tensor)
        loss_op = tf.reduce_mean(loss)

        regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])#
        loss_op = loss_op + 0.001*regularizer

        optimizer = tf.train.AdamOptimizer(1e-4, 0.5)
        train_op = optimizer.minimize(loss_op)

        
        self.signal_log.emit("Initializing weights...")
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())


        self.signal_log.emit("Start Training...")
        for epoch in range(self.max_epoch):
            #Save
            # save_path = "./weights/epoch_" + str(epoch)
            # builder = tf.saved_model.builder.SavedModelBuilder(save_path)
            # builder.add_meta_graph_and_variables(sess, ['ejshim'])
            # builder.save()
            #while True:


            idx = 0
            #for data in train_set:
            while idx < len(train_set):
                
                batch = train_set[idx:idx+batch_size]
                idx = idx+batch_size

                input_data = []
                gt_data = []

                for data in batch:
                    input_data.append(data['input'])
                    gt_data.append(data['gt'])


                # input_data = batch['input']
                # gt_data = batch['gt']

                [output_data, loss, _] = sess.run([output_data_tensor, loss_op, train_op], feed_dict={input_tensor:input_data, gt_tensor:gt_data, is_training:True})

                log = str(epoch) + "/" + str(self.max_epoch-1) + ", Loss : " +  str(loss)                  
                for i in range(batch_size):
                    utils.update_segmentation(input_poly, output_data[i], batch[i]['idx'])
                self.signal_log.emit(log)              


            #run test
            # for data in test_data:
            #     output_data = sess.run(output_data_tensor, feed_dict={input_tensor:[data['input']], is_training:False})                 
            #     utils.update_segmentation(input_poly, output_data[0], data['idx'])
            #     renderWindow.Render()
        
        self.signal_log.emit("Finished")


class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setCentralWidget(QVTKRenderWindowInteractor())
        self.renderWindow = self.centralWidget().GetRenderWindow()

        #Set REnderer        
        self.renderer = vtk.vtkRenderer()        
        self.renderWindow.GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.renderWindow.AddRenderer(self.renderer)


        self.txtActor = vtk.vtkTextActor()
        self.txtActor.SetInput("Preparing...")
        self.txtActor.GetTextProperty().SetFontFamilyToArial()
        self.txtActor.GetTextProperty().SetFontSize(46)
        self.txtActor.GetTextProperty().SetColor(1,1,0)
        self.txtActor.SetDisplayPosition(100,900)
        self.renderer.AddActor(self.txtActor)
        

        input_actor = utils.MakeActor(input_poly)
        self.renderer.AddActor(input_actor)    
        self.renderer.GetActiveCamera().Pitch(-45)
        self.renderer.ResetCamera()



        self.renderWindow.Render()

        
        #Thread
        self.thread = TrainingThread()
        self.thread.signal_log.connect(self.onLog)
        self.thread.start()
        

    def onLog(self, log):
            self.txtActor.SetInput(log)
            # utils.update_segmentation(input_poly, output_data[0], data['idx'])
            self.renderWindow.Render()





if __name__ == "__main__":
    
    app = QApplication([])
    window = MainWindow()
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())

