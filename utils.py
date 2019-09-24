import vtk
import math
import numpy as np
import pymongo
import gridfs
from bson import ObjectId
import os
import pickle



db = pymongo.MongoClient().maxilafacial
fileDB = gridfs.GridFS(db)



color_preset = [
    [255, 255, 255],
    [22, 176, 221],
    [151,73,148],
    [228,59,55],
    [253,147,29],
    [247,188,36],
    [104,199,79],
    [22, 176, 221],
    [151,73,148],
    [228,59,55],
    [253,147,29],
    [247,188,36],
    [104,199,79],
    [22, 176, 221],
    [151,73,148],
    [228,59,55],
    [253,147,29],
    [255, 255, 255],
    [255, 255, 255],
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

    polydataColor = vtk.vtkUnsignedCharArray()
    polydataColor.SetNumberOfComponents(3)
    polydataColor.SetNumberOfTuples(polydata.GetNumberOfPoints())
    for i in range(polydata.GetNumberOfPoints()):
        polydataColor.SetTuple3(i, vertexColor[0], vertexColor[1], vertexColor[2])
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

def ArrangeNormalizedPointData(pointdata, gt, resolution = 100):
    bounds = [np.min(pointdata[:,0]), np.max(pointdata[:,0]), np.min(pointdata[:,1]), np.max(pointdata[:,1]), np.min(pointdata[:,2]), np.max(pointdata[:,2])]
    bounds = [bounds[0]*resolution, bounds[1]*resolution, bounds[2]*resolution, bounds[3]*resolution, bounds[4]*resolution, bounds[5]*resolution]
    grid_locator = np.empty(shape=(resolution*resolution*resolution, 0)).tolist()
    num_points = pointdata.shape[0]

    for i in range(num_points):

        #Get Normalized Position -> Get 3D Grid Index
        position = pointdata[i]
        position = [position[0]*resolution, position[1]*resolution, position[2]*resolution]
        #position = [position[0], position[1], position[2]]
        position[0] -= bounds[0]
        position[0] /= bounds[1] - bounds[0] + 1
        position[0] = int(position[0] * resolution)
        position[1] -= bounds[2]
        position[1] /= bounds[3] - bounds[2] + 1
        position[1] = int(position[1] * resolution)
        position[2] -= bounds[4]
        position[2] /= bounds[5] - bounds[4] + 1
        position[2] = int(position[2] * resolution)

        grid_index = position[0] * resolution * resolution + position[1] * resolution + position[2]
        #Set Index Information
        

        grid_locator[grid_index].append(i)


    #Get Result Indices
    result = []    
    for index_list in grid_locator:
        result += index_list

    sorted_points = []
    sorted_gt = []
    for idx in range(num_points):
        sorted_points.append(pointdata[result[idx]])
        sorted_gt.append(gt[result[idx]])
    
    return np.array(sorted_points), np.array(sorted_gt)

def CenterPolyData(polydata):
    center = polydata.GetCenter()

    transform = vtk.vtkTransform()
    transform.Translate(-center[0], -center[1], -center[2])
    transform.Update()

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    polydata = transformFilter.GetOutput()
    
    ##Normalize
    num_points = polydata.GetNumberOfPoints()
    boundingBox = polydata.GetBounds()
    
    xRange = boundingBox[1] - boundingBox[0]
    yRange = boundingBox[3] - boundingBox[2]
    zRange = boundingBox[5] - boundingBox[4]

    
    normalizeRange = np.max([xRange, yRange, zRange])


    for idx in range(num_points):
        position = polydata.GetPoint(idx)
        position_normalized = [position[0] / normalizeRange, position[1] / normalizeRange, position[2] / normalizeRange]
        polydata.GetPoints().SetPoint(idx, position_normalized)
    polydata.GetPoints().Modified()

    return polydata


def ArrangePolyData(polydata, resolution = 100):
    bounds = polydata.GetBounds()
    bounds = [bounds[0]*resolution, bounds[1]*resolution, bounds[2]*resolution, bounds[3]*resolution, bounds[4]*resolution, bounds[5]*resolution]
    grid_locator = np.empty(shape=(resolution*resolution*resolution, 0)).tolist()
    num_points = polydata.GetNumberOfPoints()

    for i in range(num_points):

        #Get Normalized Position -> Get 3D Grid Index
        position = polydata.GetPoint(i) 
        position = [position[0]*resolution, position[1]*resolution, position[2]*resolution]        
        #position = [position[0], position[1], position[2]]
        position[0] -= bounds[0]
        position[0] /= bounds[1] - bounds[0] + 1
        position[0] = int(position[0] * resolution)
        position[1] -= bounds[2]
        position[1] /= bounds[3] - bounds[2] + 1
        position[1] = int(position[1] * resolution)
        position[2] -= bounds[4]
        position[2] /= bounds[5] - bounds[4] + 1
        position[2] = int(position[2] * resolution)

        grid_index = position[0] * resolution * resolution + position[1] * resolution + position[2]
        #Set Index Information
        grid_locator[grid_index].append(i)


    #Get Result Indices
    result = []    
    for index_list in grid_locator:
        result += index_list
    

    #Make New Polydata, Inverse results for triangle
    inverse_result = [0] * num_points
    sorted_points = vtk.vtkPoints()
    for idx in range(num_points):
        sorted_points.InsertNextPoint(polydata.GetPoint(result[idx]))
        inverse_result[result[idx]] = idx
    polydata.SetPoints(sorted_points)

    #Make Triangles
    num_cells = polydata.GetPolys().GetNumberOfCells()
    sorted_polys = vtk.vtkCellArray()
    idlist = vtk.vtkIdList()
    for idx in range(num_cells):        
        polydata.GetCellPoints(idx, idlist)
        sorted_polys.InsertNextCell(3)
        sorted_polys.InsertCellPoint(inverse_result[idlist.GetId(0)])
        sorted_polys.InsertCellPoint(inverse_result[idlist.GetId(1)])
        sorted_polys.InsertCellPoint(inverse_result[idlist.GetId(2)])
    polydata.SetPolys(sorted_polys)


def update_segmentation(polydata, outputdata, outputidx):

    for outputidx, dataidx in enumerate(outputidx):
        # color = [255, 0, 0]
        # if outputdata[outputidx] == 1: color = [0, 255, 0]
        polydata.GetPointData().GetScalars().SetTuple(dataidx, color_preset[int(outputdata[outputidx])])
    polydata.GetPointData().Modified()


def save_training_data(patientID, num_point = 1024):
    
    patient = db.patient.find_one({'_id':patientID})
    mandible_data = db.fs.files.find_one({"_id":{"$in":  list(map(ObjectId, patient['data']) ) }, "dataIndex" : 137  })

    mandible_file = fileDB.get( mandible_data['_id'] )

    #Write Temp file
    file_path = os.path.join('temp','temp.stl')
    open(file_path, 'wb').write(mandible_file.read())


    #Get Input data
    polydata = ReadSTL(file_path)
    polydata = CenterPolyData(polydata)
    point_data = GetPointData(polydata)


    #Get Ground Truth Data
    mandible_gt = patient["pointcloudGroundTruth"]["mandible"]
    gt_data = make_gt_data(mandible_gt, point_data.shape[0])

    #Save GT DAta
    with open('./temp/temp_gt', 'wb') as filehandler:
        # store the data as binary data stream
        pickle.dump(gt_data, filehandler)

def make_training_data(patientID, size = None, sample_size = 1024):

    patient = db.patient.find_one({'_id':patientID})
    mandible_data = db.fs.files.find_one({"_id":{"$in":  list(map(ObjectId, patient['data']) ) }, "dataIndex" : 137  })

    mandible_file = fileDB.get( mandible_data['_id'] )

    #Write Temp file
    file_path = os.path.join('temp','temp_training.stl')
    open(file_path, 'wb').write(mandible_file.read())


    #Get Input data
    polydata = ReadSTL(file_path)
    polydata = CenterPolyData(polydata)    
    point_data = GetPointData(polydata)


    #Get Ground Truth Data
    mandible_gt = patient["pointcloudGroundTruth"]["mandible"]

    gt_data = make_gt_data(mandible_gt, point_data.shape[0])

    train_set = make_subsample_data(point_data, gt_data, size=size, sample_size=sample_size)



    return train_set

