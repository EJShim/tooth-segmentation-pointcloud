import vtk



def ReadSTL(filepath, vertexColor = [255, 255, 255]):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filepath)
    reader.Update()

    polydata = reader.GetOutput()

    polydataColor = vtk.vtkUnsignedCharArray()
    polydataColor.SetNumberOfComponents(3)
    for i in range(polydata.GetNumberOfPoints()):
        polydataColor.InsertNextTuple(vertexColor)
    polydata.GetPointData().SetScalars(polydataColor)

    return polydata

def MakeActor(polydata):
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor

def Visualize_segmentation(polydata, segmentationData):
    if not polydata.GetNumberOfPoints() == len(segmentationData):
        print("something wrong..")
        return
    
    
    for idx, gt in enumerate(segmentationData):
        if gt == 1:
            polydata.GetPointData().GetScalars().SetTuple(idx, [0, 255, 0])
    polydata.GetPointData().Modified()

