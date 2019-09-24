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
renderWindow.SetFullScreen(True)
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