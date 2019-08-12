import vtk
import os
import pickle
import utils
import numpy as np
from datetime import datetime
import vtk.util.numpy_support as vtk_np

# Initialize RenderWIndow
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1000, 1000)
#renderWindow.SetFullScreen(True)
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)



def sort_pointIndex(polydata):

    result = []


    bounds = polydata.GetBounds() 
    grid_locator = np.empty(shape=(100,100,100, 0)).tolist()
    num_points = polydata.GetNumberOfPoints()

    for i in range(num_points):

        position = polydata.GetPoint(i)
        position = [position[0], position[1], position[2]]
        position[0] -= bounds[0]
        position[0] /= bounds[1] - bounds[0]
        position[0] = int(position[0] * 99)
        position[1] -= bounds[2]
        position[1] /= bounds[3] - bounds[2]
        position[1] = int(position[1] * 99)
        position[2] -= bounds[4]
        position[2] /= bounds[5] - bounds[4]
        position[2] = int(position[2] * 99)

        grid_locator[position[0]][position[1]][position[2]].append(i)

    grid_locator = np.array(grid_locator)
    grid_locator = grid_locator.flatten()

    result = []

    for index_list in grid_locator:
        result += index_list

    #Make New Polydata
    sorted_points = vtk.vtkPoints()
    for idx in range(num_points):
        sorted_points.InsertNextPoint(polydata.GetPoint(result[idx]))

    #polydata.SetPoints(sorted_points)

    num_cells = polydata.GetPolys().GetNumberOfCells()
    sorted_polys = vtk.vtkCellArray()

    
    # sorted_polys.InsertNextCell(3);
    # sorted_polys.InsertCellPoint();

    for idx in range(num_cells):
        idlist = vtk.vtkIdList()
        polydata.GetCellPoints(idx, idlist)
        sorted_polys.InsertNextCell(3)
        sorted_polys.InsertCellPoint(result[idlist.GetId(0)])
        sorted_polys.InsertCellPoint(result[idlist.GetId(1)])
        sorted_polys.InsertCellPoint(result[idlist.GetId(2)])        
    #polydata.SetPolys(sorted_polys)

    
    return result



if __name__ == "__main__":


    #Import Input Data.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName('./processed/temp.stl')
    reader.Update()

    polydata = reader.GetOutput()

    sorted_index = sort_pointIndex(polydata)

    #Show Color
    polydataColor = vtk.vtkFloatArray()
    polydataColor.SetNumberOfComponents(1)
    num_points = polydata.GetNumberOfPoints()
    polydataColor.SetNumberOfTuples(num_points)
    for count, index in enumerate(sorted_index):
        polydataColor.SetTuple(index, [count / num_points]) 
    polydata.GetPointData().SetScalars(polydataColor)





    input_actor = utils.MakeActor(polydata)
    renderer.AddActor(input_actor)
    
    renderer.GetActiveCamera().Pitch(-30)
    renderer.ResetCamera()
    renderWindow.Render()

    iren.Start()