import vtk
import os
import pickle
import utils
import numpy as np

#Initialize Database - maxilafacial


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


    data_list = os.listdir('./stl')
    tooth_data = []
    for mesh in data_list:
        dataIndex = int(mesh)
        if dataIndex < 100 and dataIndex > 14: tooth_data.append( os.path.join('./stl',mesh) )


    print(tooth_data)

    tooth_data.sort()

    #This will be input data
    polydata = utils.ReadSTL('./stl/137')
    utils.sort_pointIndex(polydata)
    
    #Decimation Test
    # decimation = vtk.vtkDecimatePro()
    # decimation.SetInputData(polydata)
    # decimation.SetTargetReduction(0.8)
    # decimation.Update()
    # polydata = decimation.GetOutput()

    
    #Chagne Vertex Color
    actor = utils.MakeActor(polydata)
    renderer.AddActor(actor)

    

    toothpoly_position = []
    #This will be output data
    for idx, filepath in enumerate(tooth_data):
        toothpoly = utils.ReadSTL(filepath, [255, 0, 0])
        # toothactor = MakeActor(toothpoly)
        # renderer.AddActor(toothactor)
        for i in range(toothpoly.GetNumberOfPoints()):
             toothpoly_position.append([toothpoly.GetPoint(i), idx+1])


    print(len(toothpoly_position), polydata.GetNumberOfPoints())


    result_ids = []

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(polydata)

    for index, position in enumerate(toothpoly_position):
        temp_list = vtk.vtkIdList()
        locator.FindClosestNPoints(1, position[0], temp_list)

        result_ids.append( [temp_list.GetId(0), position[1]] )
    



    #Save Ground Truth Data
    gt_data = np.zeros((polydata.GetNumberOfPoints(),))


    for idx in result_ids:
        polydata.GetPointData().GetScalars().SetTuple(idx[0], utils.color_preset[idx[1]])
        gt_data[idx] = idx[1]

    with open('./processed/groundtruth_temp', 'wb') as filehandler:
        # store the data as binary data stream
        pickle.dump(gt_data, filehandler)

    



    renderWindow.Render()

    
    iren.Start()