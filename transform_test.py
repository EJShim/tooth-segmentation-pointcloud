import vtk
import os
import pickle
import utils
import numpy as np
from datetime import datetime
import vtk.util.numpy_support as vtk_np
import time

# Initialize RenderWIndow
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1000, 1000)
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)





if __name__ == "__main__":


    #Import Input Data.stl')

    reader = vtk.vtkSTLReader()
    reader.SetFileName('./temp/temp.stl')
    reader.Update()

    polydata = reader.GetOutput()

    matrix = [0.8891658782958984, 0.06902347505092621, 0.4523492753505707, 0, 0.030766259878873825, 0.9773027300834656, -0.20960162580013275, 0, -0.4565495252609253, 0.20028769969940186, 0.8668608069419861, 0, 0, 0, 0, 1]
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transform.Inverse()
    
    #Apply Transform
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    polydata = transformFilter.GetOutput()




    num_points = polydata.GetNumberOfPoints()    

    #Sort Index of Polydata    
    utils.ArrangePolyData(polydata)
    

    #Show Color
    polydataColor = vtk.vtkFloatArray()
    polydataColor.SetNumberOfComponents(1)
    
    polydataColor.SetNumberOfTuples(num_points)
    for index in range(num_points):
        polydataColor.SetTuple(index, [index / num_points]) 
    polydata.GetPointData().SetScalars(polydataColor)


    input_actor = utils.MakeActor(polydata)
    renderer.AddActor(input_actor)
    
    renderer.GetActiveCamera().Pitch(-30)
    renderer.ResetCamera()
    renderWindow.Render()

    iren.Start()