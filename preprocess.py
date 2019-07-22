import vtk
import os
import pickle
import utils

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
        

    #This will be input data
    polydata = utils.ReadSTL('./stl/137')
    
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
             toothpoly_position.append(toothpoly.GetPoint(i))


    print(len(toothpoly_position), polydata.GetNumberOfPoints())


    result_ids = []

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(polydata)

    for index, position in enumerate(toothpoly_position):
        temp_list = vtk.vtkIdList()
        locator.FindClosestNPoints(1, position, temp_list)

        result_ids.append( temp_list.GetId(0) )
    
    
    for idx in result_ids:
        polydata.GetPointData().GetScalars().SetTuple(idx, [0, 255, 0])        


    #Save Ground Truth Data
    gt_data = []

    for i in range(polydata.GetNumberOfPoints()):
        polydataTuple = polydata.GetPointData().GetScalars().GetTuple(i)
        color = [polydataTuple[0], polydataTuple[1], polydataTuple[2]]

        if color == [0, 255, 0]: gt_data.append(1)
        else : gt_data.append(0)

    with open('./processed/groundtruth', 'wb') as filehandler:
        # store the data as binary data stream
        pickle.dump(gt_data, filehandler)

    



    renderWindow.Render()

    
    iren.Start()