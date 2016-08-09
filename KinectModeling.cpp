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
#include "igtlTimeStamp.h"

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


// H264 encoder include
extern "C"
{
  #include "codec/api/svc/codec_api.h"
  #include "codec/api/svc/codec_app_def.h"
  #include "test/utils/BufferedData.h"
  #include "test/utils/FileInputStream.h"
}

// Local include
#include "CupModeling.hpp"
uint8_t* pData[3];
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

ISVCDecoder* decoderDepthFrame_;
ISVCDecoder* decoderDepthIndex_;
ISVCDecoder* decoderColor_;
#define NO_DELAY_DECODING
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

int YUV420ToRGBConversion(uint8_t *RGBFrame, uint8_t * YUV420Frame, int iHeight, int iWidth)
{
  int componentLength = iHeight*iWidth;
  const uint8_t *srcY = YUV420Frame;
  const uint8_t *srcU = YUV420Frame + componentLength;
  const uint8_t *srcV = srcY + componentLength + componentLength/4;
  uint8_t * YUV444 = new uint8_t[componentLength * 3];
  uint8_t *dstY = YUV444;
  uint8_t *dstU = dstY + componentLength;
  uint8_t *dstV = dstU + componentLength;
  
  memcpy(dstY, srcY, componentLength);
  const int halfHeight = iHeight/2;
  const int halfWidth = iWidth/2;
  
#pragma omp parallel for default(none) shared(dstV,dstU,srcV,srcU,iWidth)
  for (int y = 0; y < halfHeight; y++) {
    for (int x = 0; x < halfWidth; x++) {
      dstU[2 * x + 2 * y*iWidth] = dstU[2 * x + 1 + 2 * y*iWidth] = srcU[x + y*iWidth/2];
      dstV[2 * x + 2 * y*iWidth] = dstV[2 * x + 1 + 2 * y*iWidth] = srcV[x + y*iWidth/2];
    }
    memcpy(&dstU[(2 * y + 1)*iWidth], &dstU[(2 * y)*iWidth], iWidth);
    memcpy(&dstV[(2 * y + 1)*iWidth], &dstV[(2 * y)*iWidth], iWidth);
  }
  
  const int yOffset = 16;
  const int cZero = 128;
  int yMult, rvMult, guMult, gvMult, buMult;
  yMult =   76309;
  rvMult = 117489;
  guMult = -13975;
  gvMult = -34925;
  buMult = 138438;
  
  static unsigned char clp_buf[384+256+384];
  static unsigned char *clip_buf = clp_buf+384;
  
  // initialize clipping table
  memset(clp_buf, 0, 384);
  
  for (int i = 0; i < 256; i++) {
    clp_buf[384+i] = i;
  }
  memset(clp_buf+384+256, 255, 384);
  
  
#pragma omp parallel for default(none) shared(dstY,dstU,dstV,RGBFrame,yMult,rvMult,guMult,gvMult,buMult,clip_buf,componentLength)// num_threads(2)
  for (int i = 0; i < componentLength; ++i) {
    const int Y_tmp = ((int)dstY[i] - yOffset) * yMult;
    const int U_tmp = (int)dstU[i] - cZero;
    const int V_tmp = (int)dstV[i] - cZero;
    
    const int R_tmp = (Y_tmp                  + V_tmp * rvMult ) >> 16;//32 to 16 bit conversion by left shifting
    const int G_tmp = (Y_tmp + U_tmp * guMult + V_tmp * gvMult ) >> 16;
    const int B_tmp = (Y_tmp + U_tmp * buMult                  ) >> 16;
    
    RGBFrame[3*i]   = clip_buf[R_tmp];
    RGBFrame[3*i+1] = clip_buf[G_tmp];
    RGBFrame[3*i+2] = clip_buf[B_tmp];
  }
  
  delete [] YUV444;
  YUV444 = NULL;
  dstY = NULL;
  dstU = NULL;
  dstV = NULL;
  return 1;
}

int64_t getTime()
{
  struct timeval tv_date;
  gettimeofday(&tv_date, NULL);
  return ((int64_t) tv_date.tv_sec * 1000000 + (int64_t) tv_date.tv_usec);
  
}

void ComposeByteSteam(uint8_t** inputData, SBufferInfo bufInfo, uint8_t *outputByteStream,  int iWidth, int iHeight)
{
  int iStride [2] = {bufInfo.UsrData.sSystemBuffer.iStride[0],bufInfo.UsrData.sSystemBuffer.iStride[1]};
#pragma omp parallel for default(none) shared(outputByteStream,inputData, iStride, iHeight, iWidth)
  for (int i = 0; i < iHeight; i++)
  {
    uint8_t* pPtr = inputData[0]+i*iStride[0];
    for (int j = 0; j < iWidth; j++)
    {
      outputByteStream[i*iWidth + j] = pPtr[j];
    }
  }
#pragma omp parallel for default(none) shared(outputByteStream,inputData, iStride, iHeight, iWidth)
  for (int i = 0; i < iHeight/2; i++)
  {
    uint8_t* pPtr = inputData[1]+i*iStride[1];
    for (int j = 0; j < iWidth/2; j++)
    {
      outputByteStream[i*iWidth/2 + j + iHeight*iWidth] = pPtr[j];
    }
  }
#pragma omp parallel for default(none) shared(outputByteStream, inputData, iStride, iHeight, iWidth)
  for (int i = 0; i < iHeight/2; i++)
  {
    uint8_t* pPtr = inputData[2]+i*iStride[1];
    for (int j = 0; j < iWidth/2; j++)
    {
      outputByteStream[i*iWidth/2 + j + iHeight*iWidth*5/4] = pPtr[j];
    }
  }
  
}

void H264Decode (ISVCDecoder* pDecoder, unsigned char* kpH264BitStream, int32_t& iWidth, int32_t& iHeight, int32_t& iStreamSize, uint8_t* outputByteStream) {
  
  unsigned long long uiTimeStamp = 0;
  int64_t iStart = 0, iEnd = 0, iTotal = 0;
  int32_t iSliceSize;
  int32_t iSliceIndex = 0;
  uint8_t* pBuf = NULL;
  uint8_t uiStartCode[4] = {0, 0, 0, 1};
  
  iStart = getTime();
  pData[0] = NULL;
  pData[1] = NULL;
  pData[2] = NULL;
  
  SBufferInfo sDstBufInfo;
  
  int32_t iBufPos = 0;
  int32_t i = 0;
  int32_t iFrameCount = 0;
  int32_t iEndOfStreamFlag = 0;
  //for coverage test purpose
  int32_t iErrorConMethod = (int32_t) ERROR_CON_SLICE_MV_COPY_CROSS_IDR_FREEZE_RES_CHANGE;
  pDecoder->SetOption (DECODER_OPTION_ERROR_CON_IDC, &iErrorConMethod);
  //~end for
  double dElapsed = 0;
  
  if (pDecoder == NULL) return;
  
  if (iStreamSize <= 0) {
    fprintf (stderr, "Current Bit Stream File is too small, read error!!!!\n");
    goto label_exit;
  }
  pBuf = new uint8_t[iStreamSize + 4];
  if (pBuf == NULL) {
    fprintf (stderr, "new buffer failed!\n");
    goto label_exit;
  }
  memcpy (pBuf, kpH264BitStream, iStreamSize);
  memcpy (pBuf + iStreamSize, &uiStartCode[0], 4); //confirmed_safe_unsafe_usage
  
  while (true) {
    if (iBufPos >= iStreamSize) {
      iEndOfStreamFlag = true;
      if (iEndOfStreamFlag)
        pDecoder->SetOption (DECODER_OPTION_END_OF_STREAM, (void*)&iEndOfStreamFlag);
      break;
    }
    for (i = 0; i < iStreamSize; i++) {
      if ((pBuf[iBufPos + i] == 0 && pBuf[iBufPos + i + 1] == 0 && pBuf[iBufPos + i + 2] == 0 && pBuf[iBufPos + i + 3] == 1
           && i > 0) || (pBuf[iBufPos + i] == 0 && pBuf[iBufPos + i + 1] == 0 && pBuf[iBufPos + i + 2] == 1 && i > 0)) {
        break;
      }
    }
    iSliceSize = i;
    if (iSliceSize < 4) { //too small size, no effective data, ignore
      iBufPos += iSliceSize;
      continue;
    }
    
    iStart = getTime();
    delete [] pData[0];
    pData[0] = NULL;
    delete [] pData[1];
    pData[1] = NULL;
    delete [] pData[2];
    pData[2] = NULL;
    uiTimeStamp ++;
    memset (&sDstBufInfo, 0, sizeof (SBufferInfo));
    sDstBufInfo.uiInBsTimeStamp = uiTimeStamp;
#ifndef NO_DELAY_DECODING
    pDecoder->DecodeFrameNoDelay (pBuf + iBufPos, iSliceSize, pData, &sDstBufInfo);
#else
    pDecoder->DecodeFrame2 (pBuf + iBufPos, iSliceSize, pData, &sDstBufInfo);
#endif
    
#ifdef NO_DELAY_DECODING
    
    pData[0] = NULL;
    pData[1] = NULL;
    pData[2] = NULL;
    memset (&sDstBufInfo, 0, sizeof (SBufferInfo));
    sDstBufInfo.uiInBsTimeStamp = uiTimeStamp;
    pDecoder->DecodeFrame2 (NULL, 0, pData, &sDstBufInfo);
    //fprintf (stderr, "iStreamSize:\t%d \t slice size:\t%d\n", iStreamSize, iSliceSize);
    iEnd  = getTime();
    iTotal = iEnd - iStart;
    if (sDstBufInfo.iBufferStatus == 1) {
      int64_t iStart2 = getTime();
      ComposeByteSteam(pData, sDstBufInfo, outputByteStream, iWidth,iHeight);
      
      //fprintf (stderr, "compose time:\t%f\n", (getTime()-iStart2)/1e6);
      ++ iFrameCount;
    }
#endif
    if (iFrameCount)
    {
      dElapsed = iTotal / 1e6;
      //fprintf (stderr, "-------------------------------------------------------\n");
      //fprintf (stderr, "iWidth:\t\t%d\nheight:\t\t%d\nFrames:\t\t%d\ndecode time:\t%f sec\nFPS:\t\t%f fps\n",
      //         iWidth, iHeight, iFrameCount, dElapsed, (iFrameCount * 1.0) / dElapsed);
      //fprintf (stderr, "-------------------------------------------------------\n");
    }
    iBufPos += iSliceSize;
    ++ iSliceIndex;
  }
  // coverity scan uninitial
label_exit:
  if (pBuf) {
    delete[] pBuf;
    pBuf = NULL;
  }
}

int SetupDecoder()
{
  
  SDecodingParam decParam;
  memset (&decParam, 0, sizeof (SDecodingParam));
  decParam.uiTargetDqLayer = UCHAR_MAX;
  decParam.eEcActiveIdc = ERROR_CON_SLICE_COPY;
  decParam.sVideoProperty.eVideoBsType = VIDEO_BITSTREAM_DEFAULT;
  WelsCreateDecoder (&decoderDepthIndex_);
  decoderDepthIndex_->Initialize (&decParam);
  WelsCreateDecoder (&decoderDepthFrame_);
  decoderDepthFrame_->Initialize (&decParam);
  WelsCreateDecoder (&decoderColor_);
  decoderColor_->Initialize (&decParam);
  
  return 1;
}

int ReceiveVideoStream(igtl::Socket * socket, igtl::MessageHeader::Pointer& header)
{
  //std::cerr << "Receiving TRANSFORM data type. " << header->GetDeviceName()<< std::endl;
  
  // Create a message buffer to receive transform data
  igtl::VideoMessage::Pointer videoMessage = igtl::VideoMessage::New();
  videoMessage->SetMessageHeader(header);
  videoMessage->AllocatePack(header->GetBodySizeToRead());
  // Receive transform data from the socket
  //igtl::TimeStamp::Pointer timer = igtl::TimeStamp::New();
  //long timePre = timer->GetTimeStampInNanoseconds();
  int read = socket->Receive(videoMessage->GetPackBodyPointer(), videoMessage->GetPackBodySize());
  
  // Deserialize the transform data
  // If you want to skip CRC check, call Unpack() without argument.
  int c = videoMessage->Unpack();
  int32_t iWidth = 512, iHeight = 424;
  int DemuxMethod = 2;
  if (read == videoMessage->GetPackBodySize() && igtl::MessageHeader::UNPACK_BODY && videoMessage->GetWidth() == 512 && videoMessage->GetHeight() == 424) // if CRC check is OK
  {
    if (DemuxMethod == 3)
    {
      int streamLength = videoMessage->GetPackBodySize()- IGTL_VIDEO_HEADER_SIZE;
      if(strcmp(header->GetDeviceName(), "DepthFrame") == 0)
      {
        DepthFrame = NULL;
        DepthFrame = new uint8_t[iHeight*iWidth*3/2];
        //AVDecode(this->AVDecoderDepthIndex, videoMsg->GetPackFragmentPointer(2), iWidth, iHeight, streamLength, DepthFrame);
        H264Decode(decoderDepthFrame_, videoMessage->GetPackFragmentPointer(2), iWidth, iHeight, streamLength, DepthFrame);
      }
      else if(strcmp(header->GetDeviceName(), "DepthIndex") == 0)
      {
        DepthIndex = NULL;
        DepthIndex = new uint8_t[iHeight*iWidth*3/2];
        //AVDecode(this->AVDecoderDepthFrame, videoMsg->GetPackFragmentPointer(2), iWidth, iHeight, streamLength, DepthIndex);
        H264Decode(decoderDepthIndex_, videoMessage->GetPackFragmentPointer(2), iWidth, iHeight, streamLength, DepthIndex);
      }
      else if(strcmp(header->GetDeviceName(), "ColorFrame") == 0)
      {
        RGBFrame = NULL;
        RGBFrame = new uint8_t[iHeight*iWidth*3];
        uint8_t* YUV420Frame = new uint8_t[iHeight*iWidth*3/2];
        //AVDecode(this->AVDecoderDepthColor, videoMsg->GetPackFragmentPointer(2), iWidth, iHeight, streamLength, YUV420Frame);
        H264Decode(decoderColor_, videoMessage->GetPackFragmentPointer(2), iWidth, iHeight, streamLength, YUV420Frame);
        bool bConverion = YUV420ToRGBConversion(RGBFrame, YUV420Frame, iHeight, iWidth);
      }
    }
    else if(DemuxMethod == 2)
    {
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
    }
    return 1;
  }
  return 0;
}


void ConnectionThread()
{
  char*  hostname = "10.22.178.137";
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
        if (frameNum >6)
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


#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>

int main(int argc, char* argv[])
{
  const std::string targetFileName = "/Users/longquanchen/Desktop/Github/TrackingSample/table_scene_mug_stereo_textured_Starbuck_Filtered.pcd";
  trackingInitialization(targetFileName);
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
  reader.read (targetFileName, *cloud);
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
  SetupDecoder();
  threadViewer->SpawnThread((igtl::ThreadFunctionType) &ConnectionThread, NULL);
  renderWindowInteractor->Start();

  while(1)
  {
    igtl::Sleep(1000);
  }
}

