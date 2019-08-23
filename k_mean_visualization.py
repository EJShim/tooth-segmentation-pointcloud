## 각 train set loss 의 평균치와 각 test set loss의 평균치 비교


import tensorflow as tf
import numpy as np
import vtk

# Renderer
renderer = vtk.vtkRenderer()


# Render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(1000, 1000)

# Render window interactor
iren = vtk.vtkRenderWindowInteractor()

interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)
iren.SetRenderWindow(renWin)


def clustering_network_tensorflow(shape):
    
    input_tensor = tf.placeholder(dtype=tf.float32, shape=shape)
    centroid_index_tensor = tf.placeholder(dtype = tf.int32)
    k_tensor = tf.placeholder(dtype = tf.int32)

    #Centroid 위치
    centroid_tensor = input_tensor[centroid_index_tensor]

    #Calculate distance between basis and all the pointclouds
    distance_tensor = tf.reduce_sum(tf.square(tf.subtract( input_tensor, centroid_tensor )), 1) * -1.0

    # print(distance_tensor)
    k_means_clustering = tf.nn.top_k(distance_tensor, k_tensor)

    output_tensor = k_means_clustering.indices



    


    return input_tensor, centroid_index_tensor, k_tensor, output_tensor


    ## Get stl file's point size number function
def get_point_from_stl(path):
    
    file_data = vtk.vtkSTLReader()
    file_data.SetFileName(path)
    file_data.Update()
    
    polydata = file_data.GetOutput()

    result = []

    for i in range (polydata.GetNumberOfPoints()):        
        result.append(polydata.GetPoint(i))

    return result



#Jansen-Shannen divergence
#WGAN


def make_point_actor(point_data):

    output_points = vtk.vtkPoints()
    output_vertices = vtk.vtkCellArray()
    output_colors = vtk.vtkUnsignedCharArray()
    output_colors.SetNumberOfComponents(3)
    output_colors.SetName("Colors")

    for point in point_data:
        idx = output_points.InsertNextPoint(point[:3])
        output_vertices.InsertNextCell(1)
        output_vertices.InsertCellPoint(idx)
        output_colors.InsertNextTuple([0, 0, 255])
        
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(output_points)  # Point data를 입력
    polydata.SetVerts(output_vertices)  # vertex 정보를 입력
    polydata.GetPointData().SetScalars(output_colors)  # 위에서 지정한 색 입력
    polydata.Modified()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(10)

    return actor


def subsample(data, size = 32768):
    index_list = np.arange(data.shape[0])
    print(index_list)

    subsample_idx = np.random.choice( index_list, size, replace=False )


    #Sort order
    subsample_idx = np.sort(subsample_idx)
    

    result = []
    

    for idx in subsample_idx:
        result.append( data[idx] )        

    result = np.array(result)

    return result
    




if __name__ == "__main__":

    #STL 데이터 불러와서 point data 로 정리하기
    vessel_data = get_point_from_stl('processed/temp.stl')

    point_data = np.array(vessel_data)
    point_data = subsample(point_data)
    
    print(point_data.shape)
    
    # Rendering 위해서 actor 만들기
    actor = make_point_actor(point_data)



    # Tensorflow 로 K-means clustering 정의
    input_tensor, centroid_index_tensor, k_tensor, output_tensor = clustering_network_tensorflow(point_data.shape)

    # Initialization
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    # output 에서 top-50 개의 후보군 index 를 만들어놓음
    output = sess.run(output_tensor, feed_dict={input_tensor : point_data, 
                                                centroid_index_tensor : 0,
                                                k_tensor : 40})
    


    #렌더링
    scalar = actor.GetMapper().GetInput().GetPointData().GetScalars()

    
    #Update polydata
    for idx in output:
        scalar.SetTuple(idx, [255, 255, 0])
    
    #Show centroid
    scalar.SetTuple(0, [255, 0, 0])
    #Visualize

    renderer.AddActor(actor)
    renWin.Render()

    iren.Start()



    

