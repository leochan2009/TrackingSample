//
//  CupModeling.hpp
//  openni_tracking
//
//  Created by Longquan Chen on 8/4/16.
//
//

#ifndef CupModeling_hpp
#define CupModeling_hpp

#include <stdio.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkSmartPointer.h>
#endif /* CupModeling_hpp */

#include <pcl/point_types.h>
typedef pcl::PointXYZRGB PointT;

void SegmentCylindarObject(vtkSmartPointer<vtkPolyData> polyData, int frameNum);
void TrackCylindarObject(vtkSmartPointer<vtkPolyData> polyData);
void trackingInitialization(const std::string targetFileName);