import vtk
import os
import random
import numpy as np
import utils

################################################### Rendering ##########################################################
## Initialize RenderWIndow
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(500, 500)


## renderWindow.SetFullScreen(True)
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)



## Actor function
def makePointCloudActor(pointcloud, gt=[], pointsize=6):

    output_points = vtk.vtkPoints()
    output_vertices = vtk.vtkCellArray()
    output_colors = vtk.vtkFloatArray()
    output_colors.SetNumberOfComponents(1)
    output_colors.SetName("Colors")
    num_points = pointcloud.shape[0]



    for idx, point in enumerate(pointcloud):
        id = output_points.InsertNextPoint(point)
        output_vertices.InsertNextCell(1)
        output_vertices.InsertCellPoint(id)

        if len(gt) == 0:
            output_colors.InsertNextTuple([idx/num_points])
        else:
            output_colors.InsertNextTuple([gt[idx]/16])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(output_points)  # Point data를 입력
    polydata.SetVerts(output_vertices)  # vertex 정보를 입력
    polydata.GetPointData().SetScalars(output_colors)  # 위에서 지정한 색 입력

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(pointsize)

    return actor


def apply_random_rotation(input_batch, gt_batch):    

    #Add Transform
    transform = vtk.vtkTransform()
    transform.RotateX(random.randrange(-45, 45))
    transform.RotateY(random.randrange(-45, 45))
    transform.RotateZ(random.randrange(-45, 45))
    transform.Translate(2, 0, 0)
    transform.Update()

    
    result_batch = []
    result_gt_batch = []
    for idx, batch in enumerate(input_batch):

        gt = gt_batch[idx]
        result = np.array([*map(transform.TransformPoint, batch.tolist() )])
        #Rearrange
        result, gt_result = utils.ArrangeNormalizedPointData(result, gt)
        result_batch.append(result)
        result_gt_batch.append(gt_result)


    return result_batch, result_gt_batch



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

    test_data = input_data[0]
    test_gt = gt_data[0]

    
    test_actor = makePointCloudActor(test_data, test_gt)

    test_data_transformed, test_data_transformed_gt = apply_random_rotation([test_data], [test_gt])
    transformed_actor = makePointCloudActor(test_data_transformed[0], test_data_transformed_gt[0])




    renderer.AddActor(test_actor)
    renderer.AddActor(transformed_actor)
    renderWindow.Render()

    iren.Start()
    exit()
