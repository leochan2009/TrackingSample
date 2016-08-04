//
//  CupModeling.cpp
//  openni_tracking
//
//  Created by Longquan Chen on 8/4/16.
//
//

#include "CupModeling.hpp"
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/common/centroid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/search/pcl_search.h>
#include <pcl/common/transforms.h>
#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>
#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

typedef pcl::PointXYZRGB PointT;
pcl::PointCloud<PointT>::Ptr cloud_cylinder_total (new pcl::PointCloud<PointT> ());
vtkSmartPointer<vtkVertexGlyphFilter> vertexFilter2(vtkSmartPointer<vtkVertexGlyphFilter>::New());
vtkSmartPointer<vtkUnsignedCharArray> colorsProcessed(vtkSmartPointer<vtkUnsignedCharArray>::New());
vtkSmartPointer<vtkPoints>  cloudProcessed(vtkSmartPointer<vtkPoints>::New());

typedef pcl::PointCloud<pcl::PointXYZRGB> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;
typedef pcl::PointXYZRGB RefPointType;
typedef pcl::tracking::ParticleXYZRPY ParticleT;
typedef pcl::tracking::ParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;
typedef pcl::PointXYZRGB RefPointType;
pcl::PointCloud<PointT>::Ptr target_cloud (new pcl::PointCloud<PointT>);
CloudPtr cloud_pass_;
CloudPtr cloud_pass_downsampled_;
boost::mutex mtx_;
boost::shared_ptr<ParticleFilter> tracker_;

void SegmentCylindarObject(vtkSmartPointer<vtkPolyData> polyData, int frameNum)
{
  
  pcl::PassThrough<PointT> pass;
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::PCDWriter writer;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  
  // Datasets
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);
  pcl::io::vtkPolyDataToPointCloud<PointT>(polyData.GetPointer(), *cloud);
  // Build a passthrough filter to remove spurious NaNs
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (1200, 1500);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;
  
  // Estimate point normals
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_filtered);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);
  
  // Create the segmentation object for the planar model and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight (0.1);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (50);
  seg.setInputCloud (cloud_filtered);
  seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);
  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
  
  // Extract the planar inliers from the input cloud
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_plane);

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_filtered2);
  extract_normals.setNegative (true);
  extract_normals.setInputCloud (cloud_normals);
  extract_normals.setIndices (inliers_plane);
  extract_normals.filter (*cloud_normals2);
  
  // Create the segmentation object for cylinder segmentation and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (5);
  seg.setRadiusLimits (0, 100);
  seg.setInputCloud (cloud_filtered2);
  seg.setInputNormals (cloud_normals2);
  
  // Obtain the cylinder inliers and coefficients
  seg.segment (*inliers_cylinder, *coefficients_cylinder);
  std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;
  
  // Write the cylinder inliers to disk
  extract.setInputCloud (cloud_filtered2);
  extract.setIndices (inliers_cylinder);
  extract.setNegative (false);
  pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_cylinder);
  cloud_cylinder_total->insert(cloud_cylinder_total->begin(),cloud_cylinder->begin(),cloud_cylinder->end());
  vtkSmartPointer<vtkPolyData> localPolyData = vtkSmartPointer<vtkPolyData>::New();
  pcl::io::pointCloudTovtkPolyData<PointT>(*cloud_cylinder, localPolyData.GetPointer());
  vtkSmartPointer<vtkPoints> ptsLocal = localPolyData->GetPoints();
  vtkSmartPointer<vtkPoints> pts = polyData->GetPoints();
  vtkSmartPointer<vtkUnsignedCharArray> colorDataLocal = (vtkUnsignedCharArray*)localPolyData->GetPointData()->GetScalars();
  vtkSmartPointer<vtkUnsignedCharArray> colorData = (vtkUnsignedCharArray*)polyData->GetPointData()->GetScalars();
  cloudProcessed->Reset();
  colorsProcessed->Reset();
  colorsProcessed->SetNumberOfComponents(3);
  colorsProcessed->SetName("Colors");
  for(int index = 0; index<pts->GetNumberOfPoints();index++)
  {
    cloudProcessed->InsertNextPoint(pts->GetPoint(index)[0],pts->GetPoint(index)[1],pts->GetPoint(index)[2]);
    unsigned char color[3] = {colorData->GetTuple(index)[0], colorData->GetTuple(index)[1], colorData->GetTuple(index)[2]};
    colorsProcessed->InsertNextTupleValue(color);
  }
  for(int index = 0; index<ptsLocal->GetNumberOfPoints();index++)
  {
    cloudProcessed->InsertNextPoint(ptsLocal->GetPoint(index)[0],ptsLocal->GetPoint(index)[1],ptsLocal->GetPoint(index)[2]);
    unsigned char color[3] = {0,0,255};
    colorsProcessed->InsertNextTupleValue(color);
  }
  polyData->SetPoints(cloudProcessed);
  polyData->GetPointData()->SetScalars(colorsProcessed);
  vertexFilter2->SetInputData(polyData);
  vertexFilter2->Update();
  polyData->ShallowCopy(vertexFilter2->GetOutput());
  if (cloud_cylinder->points.empty ())
    std::cerr << "Can't find the cylindrical component." << std::endl;
  else
  {
    if(frameNum<20)
    {
      std::cerr << "Saving the data to the vector" << std::endl;
    }
    else if(frameNum >=20 && frameNum<50)
    {
      std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
      //writer.write ("table_scene_mug_stereo_textured_cylinder.pcd", *cloud_cylinder_total, false);
    }
  }
}

void gridSampleApprox (const CloudConstPtr &cloud, Cloud &result, double leaf_size)
{
  pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> grid;
  grid.setLeafSize (static_cast<float> (leaf_size), static_cast<float> (leaf_size), static_cast<float> (leaf_size));
  grid.setInputCloud (cloud);
  grid.filter (result);
}

void trackingInitialization()
{
  pcl::PCDReader reader;
  reader.read ("/Users/longquanchen/Desktop/Github/TrackingSample/build/Debug/table_scene_mug_stereo_textured_cylinder.pcd", *target_cloud);
  std::cerr << "PointCloud has: " << target_cloud->points.size () << " data points." << std::endl;
  //Set parameters
  float downsampling_grid_size_ =  5;
  
  std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
  //default_step_covariance[0] *= 1000.0;
  //default_step_covariance[1] *= 1000.0;
  //default_step_covariance[2] *= 1000.0;
  default_step_covariance[3] *= 40.0;
  default_step_covariance[4] *= 40.0;
  default_step_covariance[5] *= 40.0;
  
  std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.01);
  std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);
  
  boost::shared_ptr<pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
  (new pcl::tracking::KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> (8));
  
  ParticleT bin_size;
  bin_size.x = 1.0f;
  bin_size.y = 1.0f;
  bin_size.z = 1.0f;
  bin_size.roll = 0.1f;
  bin_size.pitch = 0.1f;
  bin_size.yaw = 0.1f;
  
  
  //Set all parameters for  KLDAdaptiveParticleFilterOMPTracker
  tracker->setMaximumParticleNum (1000);
  tracker->setDelta (0.99);
  tracker->setEpsilon (0.2);
  tracker->setBinSize (bin_size);
  
  //Set all parameters for  ParticleFilter
  tracker_ = tracker;
  tracker_->setTrans (Eigen::Affine3f::Identity ());
  tracker_->setStepNoiseCovariance (default_step_covariance);
  tracker_->setInitialNoiseCovariance (initial_noise_covariance);
  tracker_->setInitialNoiseMean (default_initial_mean);
  tracker_->setIterationNum (1);
  tracker_->setParticleNum (600);
  tracker_->setResampleLikelihoodThr(0.00);
  tracker_->setUseNormal (false);
  
  
  //Setup coherence object for tracking
  pcl::tracking::ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr coherence = pcl::tracking::ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr
  (new pcl::tracking::ApproxNearestPairPointCloudCoherence<RefPointType> ());
  
  boost::shared_ptr<pcl::tracking::DistanceCoherence<RefPointType> > distance_coherence
  = boost::shared_ptr<pcl::tracking::DistanceCoherence<RefPointType> > (new pcl::tracking::DistanceCoherence<RefPointType> ());
  coherence->addPointCoherence (distance_coherence);
  
  boost::shared_ptr<pcl::search::Octree<RefPointType> > search (new pcl::search::Octree<RefPointType> (0.01));
  coherence->setSearchMethod (search);
  coherence->setMaximumDistance (0.01);
  
  tracker_->setCloudCoherence (coherence);
  
  //prepare the model of tracker's target
  Eigen::Vector4f c;
  Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
  CloudPtr transed_ref (new Cloud);
  CloudPtr transed_ref_downsampled (new Cloud);
  
  pcl::compute3DCentroid<RefPointType> (*target_cloud, c);
  trans.translation ().matrix () = Eigen::Vector3f (c[0], c[1], c[2]);
  pcl::transformPointCloud<RefPointType> (*target_cloud, *transed_ref, trans.inverse());
  gridSampleApprox (transed_ref, *transed_ref_downsampled, downsampling_grid_size_);
  
  //set reference model and trans
  tracker_->setReferenceCloud (transed_ref_downsampled);
  tracker_->setTrans (trans);
  
}

//Filter along a specified dimension
void filterPassThrough (const CloudConstPtr &cloud, Cloud &result)
{
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (1200.0, 1600.0);
  pass.setKeepOrganized (false);
  pass.setInputCloud (cloud);
  pass.filter (result);
}

//Draw model reference point cloud
void
drawResult ()
{
  pcl::tracking::ParticleXYZRPY result = tracker_->getResult ();
  Eigen::Affine3f transformation = tracker_->toEigenMatrix (result);
  
  //move close to camera a little for better visualization
  //transformation.translation () += Eigen::Vector3f (0.0f, 0.0f, -0.005f);
  CloudPtr result_cloud (new Cloud ());
  pcl::transformPointCloud<RefPointType> (*(tracker_->getReferenceCloud ()), *result_cloud, transformation);
  for(int index = 0; index<result_cloud->size();index++)
  {
    cloudProcessed->InsertNextPoint(result_cloud->at(index).data[0],result_cloud->at(index).data[1],result_cloud->at(index).data[2]);
    unsigned char color[3] = {0,0,255};
    colorsProcessed->InsertNextTupleValue(color);
  }
  
  ParticleFilter::PointCloudStatePtr particles = tracker_->getParticles ();
  //Set pointCloud with particle's points
  for (size_t i = 0; i < particles->points.size (); i++)
  {
    cloudProcessed->InsertNextPoint(particles->points[i].x,particles->points[i].y,particles->points[i].z);
    unsigned char color[3] = {255,0,0};
    colorsProcessed->InsertNextTupleValue(color);
  }
  
}

void TrackCylindarObject (vtkSmartPointer<vtkPolyData> polydata)
{
  boost::mutex::scoped_lock lock (mtx_);
  cloud_pass_.reset (new Cloud);
  cloud_pass_downsampled_.reset (new Cloud);
  CloudPtr cloud;
  cloud.reset(new Cloud);
  pcl::io::vtkPolyDataToPointCloud(polydata.GetPointer(), *cloud);
  filterPassThrough (cloud, *cloud_pass_);
  
  //Track the object
  tracker_->setInputCloud (cloud_pass_);
  tracker_->compute ();
  cloudProcessed->Reset();
  colorsProcessed->Reset();
  colorsProcessed->SetNumberOfComponents(3);
  colorsProcessed->SetName("Colors");
  for(int index = 0; index<cloud_pass_->size();index++)
  {
    cloudProcessed->InsertNextPoint(cloud_pass_->at(index).data[0],cloud_pass_->at(index).data[1],cloud_pass_->at(index).data[2]);
    unsigned char color[3] = {cloud_pass_->at(index).r,cloud_pass_->at(index).g,cloud_pass_->at(index).b};
    colorsProcessed->InsertNextTupleValue(color);
  }
  drawResult();
  polydata->SetPoints(cloudProcessed);
  polydata->GetPointData()->SetScalars(colorsProcessed);
  vertexFilter2->SetInputData(polydata);
  vertexFilter2->Update();
  polydata->ShallowCopy(vertexFilter2->GetOutput());
}