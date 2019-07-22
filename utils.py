import vtk

def PoitncloudToMesh(pointcloud):
    output_points = vtk.vtkPoints()
    output_vertices = vtk.vtkCellArray()
    output_colors = vtk.vtkUnsignedCharArray()
    output_colors.SetNumberOfComponents(3)
    output_colors.SetName("Colors")
    
    
    for point in pointcloud:
        idx = output_points.InsertNextPoint(point)
        output_vertices.InsertNextCell(1)
        output_vertices.InsertCellPoint(idx)
        output_colors.InsertNextTuple([255, 255, 255])

    output_polydata = vtk.vtkPolyData()
    output_polydata.SetPoints(output_points) # Point data를 입력
    output_polydata.SetVerts(output_vertices) # vertex 정보를 입력
    output_polydata.GetPointData().SetScalars(output_colors) # 위에서 지정한 색 입력
    output_polydata.Modified()


    return output_polydata


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
        else:
            polydata.GetPointData().GetScalars().SetTuple(idx, [255, 255, 255])

    polydata.GetPointData().Modified()

