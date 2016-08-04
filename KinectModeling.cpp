//
//  KinectModeling.cpp
//  openni_tracking
//
//  Created by Longquan Chen on 8/3/16.
//
//

#include <stdio.h>

//PCL Include
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <cstdlib>
#include <cstring>
//OpenIGTLink Include
#include "igtlOSUtil.h"
#include "igtlClientSocket.h"
#include "igtlMessageHeader.h"
#include "igtlVideoMessage.h"
#include "igtlTransformMessage.h"
#include "igtlMultiThreader.h"
#include "igtlConditionVariable.h"

// VTK includes
#include <vtkNew.h>
#include <vtkCallbackCommand.h>
#include <vtkImageData.h>
#include <vtkTransform.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkVector.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleSwitch.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>


// Local include
#include "CupModeling.hpp"
uint8_t *RGBFrame;
uint8_t *DepthFrame;
uint8_t *DepthIndex;
vtkSmartPointer<vtkUnsignedCharArray> colors;
vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter;
vtkSmartPointer<vtkPoints>  cloud;
vtkSmartPointer<vtkPolyData> polyData;
vtkSmartPointer<vtkPolyDataMapper> mapper;
vtkSmartPointer<vtkActor> actor;
vtkSmartPointer<vtkRenderer> renderer;
vtkSmartPointer<vtkRenderWindow> renderWindow;
vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;
bool interactionActive;
igtl::ConditionVariable::Pointer conditionVar;
igtl::SimpleMutexLock * localMutex;
int frameNum = 0;
void ConvertDepthToPoints(unsigned char* buf, unsigned char* bufIndex, unsigned char* bufColor, int depth_width_, int depth_height_) // now it is fixed to 512, 424
{
  ;//(depth_width_*depth_height_,vtkVector<float, 3>());
  
  bool isDepthOnly = false;
  cloud->Reset();
  colors->Reset();
  colors->SetNumberOfComponents(3);
  colors->SetName("Colors");
  //I inserted 525 as Julius Kammerl said as focal length
  register float constant = 0;
  
  if (isDepthOnly)
  {
    constant = 3.501e-3f;
  }
  else
  {
    constant = 1.83e-3f;
  }
  
  register int centerX = (depth_width_ >> 1);
  int centerY = (depth_height_ >> 1);
  
  //I also ignore invalid values completely
  float bad_point = std::numeric_limits<float>::quiet_NaN ();
  std::vector<uint8_t> pBuf(depth_width_*depth_height_, 0);
  for (int i = 0; i< depth_width_*depth_height_; i++)
  {
    pBuf[i] = *(bufIndex+i);
  }
  register int depth_idx = 0;
  int pointNum = 0;
  float meanX = 0;
  float meanY = 0;
  float meanZ = 0;
  
  for (int v = -centerY; v < centerY; ++v)
  {
    for (register int u = -centerX; u < centerX; ++u, ++depth_idx)
    {
      
      vtkVector<float, 3> pt;
      //This part is used for invalid measurements, I removed it
      int pixelValue = buf[depth_idx];
      if((bufIndex[depth_idx] == 0) || (bufColor[3*depth_idx] == 0 && bufColor[3*depth_idx+1] == 0 && bufColor[3*depth_idx+2]==0) )
      {
        // not valid
        pt[0] = pt[1] = pt[2] = bad_point;
        continue;
      }
      pt[2] = pixelValue + (bufIndex[depth_idx]-1)*256 + 500;
      pt[0] = static_cast<float> (-u) * pt[2] * constant;
      pt[1] = static_cast<float> (-v) * pt[2] * constant;
      meanX += pt[0];
      meanY += pt[1];
      meanZ += pt[2];
      cloud->InsertNextPoint(pt[0],pt[1],pt[2]);
      unsigned char color[3] = {bufColor[3*depth_idx],bufColor[3*depth_idx+1],bufColor[3*depth_idx+2]};
      colors->InsertNextTupleValue(color);
      pointNum ++;
    }
  }
  meanX /= pointNum;
  meanY /= pointNum;
  meanZ /= pointNum;
  if (pointNum>0)
  {
    polyData->SetPoints(cloud);
    polyData->GetPointData()->SetScalars(colors);
    vertexFilter->SetInputData(polyData);
    vertexFilter->Update();
    polyData->ShallowCopy(vertexFilter->GetOutput());
  }
}


int ReceiveVideoStream(igtl::Socket * socket, igtl::MessageHeader::Pointer& header)
{
  std::cerr << "Receiving TRANSFORM data type. " << header->GetDeviceName()<< std::endl;
  
  // Create a message buffer to receive transform data
  igtl::VideoMessage::Pointer videoMessage = igtl::VideoMessage::New();
  videoMessage->SetMessageHeader(header);
  videoMessage->AllocatePack(header->GetBodySizeToRead());
  // Receive transform data from the socket
  int read = socket->Receive(videoMessage->GetPackBodyPointer(), videoMessage->GetPackBodySize());
  
  // Deserialize the transform data
  // If you want to skip CRC check, call Unpack() without argument.
  int c = videoMessage->Unpack();
  int32_t iWidth = 512, iHeight = 424;
  if (read == videoMessage->GetPackBodySize() && igtl::MessageHeader::UNPACK_BODY && videoMessage->GetWidth() == 512 && videoMessage->GetHeight() == 424) // if CRC check is OK
  {
    int streamLength = videoMessage->GetPackBodySize()- IGTL_VIDEO_HEADER_SIZE;
    if(strcmp(header->GetDeviceName(), "DepthFrame") == 0)
    {
      DepthFrame = NULL;
      DepthFrame = new uint8_t[iHeight*iWidth];
      memcpy(DepthFrame, videoMessage->GetPackFragmentPointer(2), iWidth*iHeight);
    }
    else if(strcmp(header->GetDeviceName(), "DepthIndex") == 0)
    {
      DepthIndex = NULL;
      DepthIndex = new uint8_t[iHeight*iWidth];
      memcpy(DepthIndex, videoMessage->GetPackFragmentPointer(2), iWidth*iHeight);
    }
    else if(strcmp(header->GetDeviceName(), "ColorFrame") == 0)
    {
      RGBFrame = NULL;
      RGBFrame = new uint8_t[iHeight*iWidth*3];
      memcpy(RGBFrame, videoMessage->GetPackFragmentPointer(2), iWidth*iHeight*3);
    }
    return 1;
  }
  return 0;
}


void ConnectionThread()
{
  char*  hostname = "10.22.178.28";
  int    port     = 18944;
  
  //------------------------------------------------------------
  // Establish Connection
  
  igtl::ClientSocket::Pointer socket;
  socket = igtl::ClientSocket::New();
  int r = socket->ConnectToServer(hostname, port);
  
  if (r != 0)
  {
    std::cerr << "Cannot connect to the server." << std::endl;
    exit(0);
  }
  
  //------------------------------------------------------------
  // Create a message buffer to receive header
  igtl::MessageHeader::Pointer headerMsg;
  headerMsg = igtl::MessageHeader::New();
  
  //------------------------------------------------------------
  // Allocate a time stamp
  igtl::TimeStamp::Pointer ts;
  ts = igtl::TimeStamp::New();
  igtl::StartVideoDataMessage::Pointer startVideoMsg;
  startVideoMsg = igtl::StartVideoDataMessage::New();
  startVideoMsg->Pack();
  socket->Send(startVideoMsg->GetPackPointer(), startVideoMsg->GetPackSize());
  
  while (1)
  {
    // Initialize receive buffer
    headerMsg->InitPack();
    
    // Receive generic header from the socket
    int r = socket->Receive(headerMsg->GetPackPointer(), headerMsg->GetPackSize());
    if (r == 0)
    {
      socket->CloseSocket();
      exit(0);
    }
    if (r != headerMsg->GetPackSize())
    {
      continue;
    }
    
    // Deserialize the header
    headerMsg->Unpack();
    
    // Get time stamp
    igtlUint32 sec;
    igtlUint32 nanosec;
    
    headerMsg->GetTimeStamp(ts);
    ts->GetTimeStamp(&sec, &nanosec);
    
    
    // Check data type and receive data body
    if (strcmp(headerMsg->GetDeviceType(), "ColoredDepth") == 0)
    {
      ReceiveVideoStream(socket, headerMsg);
      if (DepthFrame && RGBFrame && DepthIndex)
      {
        localMutex->Lock();
        while(interactionActive)
        {
          conditionVar->Wait(localMutex);
        }
        localMutex->Unlock();
        ConvertDepthToPoints((unsigned char*)DepthFrame,DepthIndex, RGBFrame, 512, 424);
        frameNum++;
        //SegmentCylindarObject(polyData, frameNum);
        TrackCylindarObject(polyData);
        delete [] DepthFrame;
        DepthFrame = NULL;
        delete [] RGBFrame;
        RGBFrame = NULL;
        delete [] DepthIndex;
        DepthIndex = NULL;
        mapper->SetInputData(polyData);
        renderer->GetRenderWindow()->Render();
      }
    }
    else
    {
      std::cerr << "Receiving : " << headerMsg->GetDeviceType() << std::endl;
      socket->Skip(headerMsg->GetBodySizeToRead(), 0);
    }
  }
  
  //------------------------------------------------------------
  // Close connection (The example code never reaches this section ...)
  
  socket->CloseSocket();
}
typedef pcl::PointXYZ PointT;
class customMouseInteractorStyle : public vtkInteractorStyleTrackballCamera
{
public:
  static customMouseInteractorStyle* New();
  vtkTypeMacro(customMouseInteractorStyle, vtkInteractorStyleTrackballCamera);
  virtual void OnLeftButtonDown()
  {
    std::cout << "Pressed left mouse button." << std::endl;
    // Forward events
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
    interactionActive = true;
  }
  
  virtual void OnLeftButtonUp()
  {
    std::cout << "Release left mouse button." << std::endl;
    // Forward events
    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
    interactionActive = false;
    conditionVar->Signal();
  }
  
  virtual void OnMiddleButtonDown()
  {
    std::cout << "Pressed middle mouse button." << std::endl;
    // Forward events
    vtkInteractorStyleTrackballCamera::OnMiddleButtonDown();
  }
  
  virtual void OnRightButtonDown()
  {
    std::cout << "Pressed right mouse button." << std::endl;
    // Forward events
    vtkInteractorStyleTrackballCamera::OnRightButtonDown();
  }
};
vtkStandardNewMacro(customMouseInteractorStyle);

int main(int argc, char* argv[])
{
  trackingInitialization();
  conditionVar = igtl::ConditionVariable::New();
  localMutex = igtl::SimpleMutexLock::New();
  interactionActive = false;
  // Create a mapper
  mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(polyData);
  mapper->SetColorModeToDefault();
  
  // Create an actor
  actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  
  // A renderer and render window
  vtkSmartPointer<vtkCamera> camera =
  vtkSmartPointer<vtkCamera>::New();
  camera->SetPosition(0, 0, -1000);
  camera->SetFocalPoint(157, 312, 2800);
  
  // Create a renderer, render window, and interactor
  renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->SetActiveCamera(camera);
  renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  
  // An interactor
  renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
  
  // Add the actors to the scene
  renderer->AddActor(actor);
  renderer->SetBackground(.3, .4, .5);
  
  // Render an image (lights and cameras are created automatically)
  
  renderWindow->Render();
  igtl::MultiThreader::Pointer threadViewer = igtl::MultiThreader::New();
  // Begin mouse interaction
  
  RGBFrame = NULL;
  DepthFrame = NULL;
  DepthIndex = NULL;
  colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
  vertexFilter = vtkSmartPointer<vtkVertexGlyphFilter>::New();
  cloud = vtkSmartPointer<vtkPoints>::New();
  polyData = vtkSmartPointer<vtkPolyData>::New();
  
  pcl::PCDReader reader;
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  reader.read ("/Users/longquanchen/Desktop/Github/TrackingSample/build/Debug/table_scene_mug_stereo_textured_cylinder.pcd", *cloud);
  std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;
  pcl::io::pointCloudTovtkPolyData<PointT>(*cloud, polyData.GetPointer());
  mapper->SetInputData(polyData);
  // Render an image (lights and cameras are created automatically)
  renderWindow->SetSize(1000, 600);
  renderWindow->Render();
  vtkInteractorObserver *currentStyle =
  renderWindowInteractor->GetInteractorStyle();
  std::cout << "currentStyle class name: " << currentStyle->GetClassName() << std::endl;
  
  vtkInteractorStyleSwitch *iss = vtkInteractorStyleSwitch::SafeDownCast(currentStyle);
  vtkInteractorObserver *actualStyle = iss->GetCurrentStyle();
  std::cout << "actualStyle class name: " <<
  actualStyle->GetClassName() << std::endl;
  vtkSmartPointer<customMouseInteractorStyle> style = vtkSmartPointer<customMouseInteractorStyle>::New();
  renderWindowInteractor->SetInteractorStyle( style );
  std::cout << "currentStyle class name: " << renderWindowInteractor->GetInteractorStyle()->GetClassName() << std::endl;
  threadViewer->SpawnThread((igtl::ThreadFunctionType) &ConnectionThread, NULL);
  renderWindowInteractor->Start();

  while(1)
  {
    igtl::Sleep(1000);
  }
}

