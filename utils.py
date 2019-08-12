import vtk
import math
import numpy as np

color_preset = [
    [255, 255, 255],    
    [75, 0, 130],
    [0, 0, 255],
    [148, 0, 211],
    [0, 255, 0],
    [255, 255, 0],
    [255, 127, 0],
    [255, 0, 0],
    [255, 0, 0],
    [255, 127, 0],
    [255, 255, 0],
    [0, 255, 0],
    [148, 0, 211],
    [0, 0, 255],
    [75, 0, 130],
    [0, 0, 255],
]
def sort_by_position(position_data, gt_data):
    sort_key = []
    for position in position_data:
        sort_key.append(np.mean(position))
    sort_key = np.array(sort_key)

    sort_key = sort_key.argsort()


    position_data = position_data[sort_key]
    gt_data = gt_data[sort_key]

    return position_data, gt_data

def make_gt_data(gtData, numberOfPoints):
    result = np.zeros(shape=(numberOfPoints,))

    for gt in gtData:
        result[ gt['pointIndex'] ] = gt['label']


    return result

def normalize_input_data(data):
    xmin = np.amin(data[:,0])
    ymin = np.amin(data[:,1])
    zmin = np.amin(data[:,2])
    

    data[:,0] = data[:,0] - xmin
    data[:,1] = data[:,1] - ymin
    data[:,2] = data[:,2] - zmin

    xmax = np.amax(data[:,0])
    ymax = np.amax(data[:,1])
    zmax = np.amax(data[:,2])

    data[:,0] = data[:,0] / xmax
    data[:,1] = data[:,1] / ymax
    data[:,2] = data[:,2] / zmax

    xmax = np.amax(data[:,0])
    ymax = np.amax(data[:,1])
    zmax = np.amax(data[:,2])

    
    return data




def sort_subsample(original_data, target_data):
    sort_key = []
    

    # it is....complicated.....
    axes = [-1000, -1000, -1000]
    for idx in target_data:
        sort_key.append(np.linalg.norm(original_data[idx]-axes))
    sort_key = np.array(sort_key)
    sort_key = sort_key.argsort()


    output = target_data[sort_key]
    return output


def make_subsample_data(original_data, original_ground_truth, size = None, sample_size = 32768):

    if size == None:
        size = math.ceil (original_ground_truth.shape[0] / sample_size)
    output = []

    index_list = np.arange(original_data.shape[0])
    for i in range(size):        
        #subsample
        
        if index_list.shape[0] < sample_size :
            sample_list = np.array([i for i in np.arange(original_data.shape[0]) if i not in index_list]) 
            subsample_idx = np.random.choice( sample_list, sample_size - index_list.shape[0], replace=False )
            subsample_idx = np.concatenate( (subsample_idx, index_list) )            

            index_list = np.arange(original_data.shape[0])
        else:
            subsample_idx = np.random.choice( index_list, sample_size, replace=False )        
            index_list = np.array([i for i in index_list if i not in subsample_idx])


        #Sort order
        subsample_idx = np.sort(subsample_idx)
        

        subsample_input = []
        subsample_gt = []

        for idx in subsample_idx:
            subsample_input.append( original_data[idx] )
            subsample_gt.append( original_ground_truth[idx] )

        subsample_input = np.array(subsample_input)
        subsample_gt = np.array(subsample_gt)

        output.append({'idx':subsample_idx, 'input':subsample_input, 'gt':subsample_gt})

    return output


def UpsampleGroundTruth(subsampled_gt, input_data, subsample_idx):    

    upsampled_gt = np.zeros((input_data.GetNumberOfPoints(),))

    for idx, data in enumerate(subsampled_gt):
        upsampled_gt[ subsample_idx[idx] ] = data



    
    return upsampled_gt


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

    polydataColor = vtk.vtkFloatArray()
    polydataColor.SetNumberOfComponents(3)
    for i in range(polydata.GetNumberOfPoints()):
        polydataColor.InsertNextTuple(vertexColor)
    polydata.GetPointData().SetScalars(polydataColor)

    return polydata

def GetPointData(polyData):

    result = []
    for idx in range(polyData.GetNumberOfPoints()):
        position = np.array(polyData.GetPoint(idx))
        result.append(position)

    result = np.array(result)

    return result

def MakeActor(polydata):
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def sort_pointIndex(polydata):

    result = []


    bounds = polydata.GetBounds() 
    grid_locator = np.empty(shape=(100,100,100, 0)).tolist()


    for i in range(polydata.GetNumberOfPoints()):

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

    
    return result


def update_segmentation(polydata, outputdata, outputidx):

    for outputidx, dataidx in enumerate(outputidx):
        # color = [255, 0, 0]
        # if outputdata[outputidx] == 1: color = [0, 255, 0]
        polydata.GetPointData().GetScalars().SetTuple(dataidx, color_preset[int(outputdata[outputidx])])
    polydata.GetPointData().Modified()

