import vtk
import os
import random
import numpy as np
import utils
import math

################################################### Rendering ##########################################################
## Initialize RenderWIndow
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(1000, 500)
renderer = vtk.vtkRenderer()
renderWindow.AddRenderer(renderer)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)
interactorStyle = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(interactorStyle)



def MakeBoundingBoxActor(polydata):
    
    bbFilter = vtk.vtkOutlineFilter()
    bbFilter.SetInputData(polydata)
    bbFilter.Update()

    actor = utils.MakeActor(bbFilter.GetOutput())

    return actor

def VisualizePolydataColor(polydata):

    num_points = polydata.GetNumberOfPoints()
    polydataColor = vtk.vtkFloatArray()
    polydataColor.SetNumberOfComponents(1)    
    polydataColor.SetNumberOfTuples(num_points)
    for index in range(num_points):
        polydataColor.SetTuple(index, [index / num_points]) 
    polydata.GetPointData().SetScalars(polydataColor)

def normalizePolydata(polydata):

    result = polydata


    num_points = polydata.GetNumberOfPoints()
    boundingBox = polydata.GetBounds()
    
    xRange = boundingBox[1] - boundingBox[0]
    yRange = boundingBox[3] - boundingBox[2]
    zRange = boundingBox[5] - boundingBox[4]

    
    normalizeRange = 1#np.max([xRange, yRange, zRange])


    for idx in range(num_points):
        position = polydata.GetPoint(idx)
        position_normalized = [(position[0] - boundingBox[0]) / normalizeRange, (position[1] - boundingBox[2]) / normalizeRange, (position[2] - boundingBox[4]) / normalizeRange]
        polydata.GetPoints().SetPoint(idx, position_normalized)
    polydata.GetPoints().Modified()

    return result

    
if __name__ == "__main__":

    #Import Input Data
    input_poly = utils.ReadSTL('./temp/temp.stl')



    #Add Arrangemetnt Color
    utils.ArrangePolyData(input_poly)
    VisualizePolydataColor(input_poly)
  



    #Normalized polydat
    normalized_poly = vtk.vtkPolyData()
    normalized_poly.DeepCopy(input_poly)

    center = normalized_poly.GetCenter()
    
    #Transform
    transform = vtk.vtkTransform()

    transform.Translate(center[0], center[1], center[2])
    transform.RotateZ(random.randrange(0, 45))
    transform.RotateY(random.randrange(0, 45))
    transform.RotateX(random.randrange(0, 45))
    transform.Translate(-center[0], -center[1], -center[2])
    
    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(normalized_poly)
    transformFilter.SetTransform(transform)
    transformFilter.Update()
    normalized_poly = transformFilter.GetOutput()

    corner = [0, 0, 0]
    max = 
    vtk.vtkOBBTree.ComputeOBB(normalized_poly.GetPoints(), [0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0])



    #normalized_poly = normalizePolydata(normalized_poly)
    #Add Arrangemetnt Color
    utils.ArrangePolyData(normalized_poly)
    VisualizePolydataColor(normalized_poly)




    #Visualize Actor
    input_actor_original = utils.MakeActor(input_poly)
    renderer.AddActor(input_actor_original)
    renderer.AddActor(MakeBoundingBoxActor(input_poly))

    input_actor_normalized = utils.MakeActor(normalized_poly)
    renderer.AddActor(input_actor_normalized)
    renderer.AddActor(MakeBoundingBoxActor(normalized_poly))


    renderWindow.Render()
    iren.Start()
