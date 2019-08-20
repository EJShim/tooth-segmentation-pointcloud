#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkIdList.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vector>

void PolyDataSort(vtkSmartPointer<vtkPolyData> polyData, int resolution = 100){
    //Initialize Vector with resolution 1000
    //int resolution = 100;
    int numPoints = polyData->GetNumberOfPoints();
    double* bounds = polyData->GetBounds();
    std::vector<std::vector<int>> grid_locator(resolution*resolution*resolution);

    for(int index = 0 ; index < numPoints ; index++ ){
        double* position = polyData->GetPoint(index);

        position[0] -= bounds[0];
        position[0] /= bounds[1] - bounds[0] + 1;
        int positionX = int(position[0] * resolution);

        position[1] -= bounds[2];
        position[1] /= bounds[3] - bounds[2] + 1;
        int positionY = int(position[1] * resolution);

        position[2] -= bounds[4];
        position[2] /= bounds[5] - bounds[4] + 1;
        int positionZ = int(position[2] * resolution);

        //Calculate Index of grid
        int locator_index = positionX * resolution * resolution + positionY * resolution + positionZ;

        grid_locator[locator_index].push_back(index);
    }

    //Merge into one reusult
    std::vector<int> result;
    for(int i = 0 ; i < grid_locator.size() ; i++){
        result.insert(std::end(result), std::begin(grid_locator[i]), std::end(grid_locator[i]));
    }

    //Make Inverse Result
    std::vector<int> inverseResult(result.size());
    vtkSmartPointer<vtkPoints> sortedPoints = vtkSmartPointer<vtkPoints>::New();
    for(int index=0 ; index<numPoints ; index++){
        sortedPoints->InsertNextPoint(polyData->GetPoint(result[index]));
        inverseResult[result[index]] = index;
    }
    polyData->SetPoints(sortedPoints);

    //Make Triangles
    int numCells = polyData->GetPolys()->GetNumberOfCells();
    vtkSmartPointer<vtkCellArray> sortedPolys = vtkSmartPointer<vtkCellArray>::New();
    vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();

    for(int index=0 ; index<numCells ; index++){
        polyData->GetCellPoints(index, idList);
        sortedPolys->InsertNextCell(3);
        sortedPolys->InsertCellPoint(inverseResult[idList->GetId(0)]);
        sortedPolys->InsertCellPoint(inverseResult[idList->GetId(1)]);
        sortedPolys->InsertCellPoint(inverseResult[idList->GetId(2)]);
    }

    polyData->SetPolys(sortedPolys);
}

int main ( int argc, char *argv[] )
{


    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(500, 500);
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    vtkSmartPointer<vtkInteractorStyleTrackballCamera> interactorStyle = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    renderWindowInteractor->SetInteractorStyle(interactorStyle);



    //Initialize Polydata
        
    vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
    reader->SetFileName("../processed/temp.stl");
    reader->Update();

    //get polydata]
    vtkSmartPointer<vtkPolyData> polyData = reader->GetOutput();

    //
    PolyDataSort(polyData);



    int numPoints = polyData->GetNumberOfPoints();
    //add color
    vtkSmartPointer<vtkFloatArray> polyColor = vtkSmartPointer<vtkFloatArray>::New();
    polyColor->SetNumberOfComponents(1);
    polyColor->SetNumberOfTuples(numPoints);

    for(int index = 0 ; index < numPoints ; index++){

        polyColor->SetTuple1(index, float(index) / float(numPoints));
    }
    polyData->GetPointData()->SetScalars(polyColor);

    // Visualize
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputData(polyData);

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    renderer->AddActor(actor);

    renderWindow->Render();
    renderWindowInteractor->Start();

    return EXIT_SUCCESS;
}