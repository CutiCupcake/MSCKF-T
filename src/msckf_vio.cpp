/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>

#include <fstream>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>  // 引入PCL的变换库
#include <pcl/filters/filter.h>
#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>
#include <msckf_vio/ikd_Tree.h>
#include <omp.h>
#define NUM_MATCH_POINTS    (5)
using namespace std;
using namespace Eigen;
using namespace sensor_msgs;
using namespace pcl;
namespace msckf_vio{
typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZ;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;

// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0.001;
double IMUState::acc_noise = 0.01;
double IMUState::gyro_bias_noise = 0.001;
double IMUState::acc_bias_noise = 0.01;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();
pcl::PointCloud<pcl::PointXYZ>::Ptr mapping_ptr(new pcl::PointCloud<pcl::PointXYZ>());

// Static member variables in CAMState class.
//Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();
int feats_down_size = 0;
KD_TREE<PointType> ikdtree;
vector<PointVector>  Nearest_Points; 
bool   point_selected_surf[100000] = {0};
int add_point_size= 0;

int times = 0;
int pc_width = 224;
int pc_height = 172;
int frame =0;

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
double TofState::observation_noise = 0.02;
Feature::OptimizationConfig Feature::optimization_config;

map<int, double> MsckfVio::chi_squared_test_table;

MsckfVio::MsckfVio(ros::NodeHandle& pnh):
  is_gravity_set(false),
  is_first_img(true),
  first_tof(true),
  nh(pnh) {
  return;
}

bool MsckfVio::loadParameters() {
  // Frame id
  nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
  nh.param<string>("child_frame_id", child_frame_id, "robot");
  nh.param<bool>("publish_tf", publish_tf, true);
  nh.param<double>("frame_rate", frame_rate, 40.0);//40
  nh.param<double>("position_std_threshold", position_std_threshold, 8.0);

  nh.param<double>("rotation_threshold", rotation_threshold, 0.2618);
  nh.param<double>("translation_threshold", translation_threshold, 0.4);
  nh.param<double>("tracking_rate_threshold", tracking_rate_threshold, 0.5);

  // Feature optimization parameters
  nh.param<double>("feature/config/translation_threshold",
      Feature::optimization_config.translation_threshold, 0.2);

  // Noise related parameters
  nh.param<double>("noise/gyro", IMUState::gyro_noise, 0.001);
  nh.param<double>("noise/acc", IMUState::acc_noise, 0.01);
  nh.param<double>("noise/gyro_bias", IMUState::gyro_bias_noise, 0.001);
  nh.param<double>("noise/acc_bias", IMUState::acc_bias_noise, 0.01);
  nh.param<double>("noise/feature", Feature::observation_noise, 0.01);
  nh.param<double>("noise/tof", TofState::observation_noise, 0.02);
  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;
  TofState::observation_noise *= TofState::observation_noise;
  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?
  nh.param<double>("initial_state/velocity/x",
      state_server.imu_state.velocity(0), 0.0);
  nh.param<double>("initial_state/velocity/y",
      state_server.imu_state.velocity(1), 0.0);
  nh.param<double>("initial_state/velocity/z",
      state_server.imu_state.velocity(2), 0.0);

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);
  double extrinsic_LI_rotation_cov, extrinsic_LI_translation_cov;//modified 
  nh.param<double>("initial_covariance/extrinsic_LI_rotation_cov",
      extrinsic_LI_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_LI_translation_cov",
      extrinsic_LI_translation_cov, 1e-4);
  nh.param<double>("filter_size_map_min",
      filter_size_map_min, 0.001);  
  state_server.state_cov = MatrixXd::Zero(33, 33);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;
  for (int i = 21; i < 24; ++i)
    state_server.state_cov(i, i) = extrinsic_LI_rotation_cov;
  for (int i = 24; i < 27; ++i)
    state_server.state_cov(i, i) = extrinsic_LI_translation_cov;


  // Transformation offsets between the frames involved.
  Isometry3d T_imu_cam0 = utils::getTransformEigen(nh, "cam0/T_cam_imu");
  Isometry3d T_cam0_imu = T_imu_cam0.inverse();
    //modified T是从imu到lidar的变换。
  Isometry3d T_imu_lidar = utils::getTransformEigen(nh, "T_lidar_imu");
  Isometry3d T_lidar_imu = T_imu_lidar.inverse();

  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
    //modified  赋予到IMU的state里
  state_server.imu_state.R_imu_lidar = T_lidar_imu.linear().transpose();
  state_server.imu_state.t_lidar_imu = T_lidar_imu.translation();
  state_server.tof_state.position = state_server.imu_state.position;
  state_server.tof_state.orientation = state_server.imu_state.orientation; 

  //CAMState::T_cam0_cam1 =
    //utils::getTransformEigen(nh, "cam1/T_cn_cnm1");
  IMUState::T_imu_body =
    utils::getTransformEigen(nh, "T_imu_body").inverse();

  // Maximum number of camera states to be stored
  nh.param<int>("max_cam_state_size", max_cam_state_size, 5);
  ROS_INFO("===========================================");
  ROS_INFO("fixed frame id: %s", fixed_frame_id.c_str());
  ROS_INFO("child frame id: %s", child_frame_id.c_str());
  ROS_INFO("publish tf: %d", publish_tf);
  ROS_INFO("frame rate: %f", frame_rate);
  ROS_INFO("position std threshold: %f", position_std_threshold);
  ROS_INFO("Keyframe rotation threshold: %f", rotation_threshold);
  ROS_INFO("Keyframe translation threshold: %f", translation_threshold);
  ROS_INFO("Keyframe tracking rate threshold: %f", tracking_rate_threshold);
  ROS_INFO("gyro noise: %.10f", IMUState::gyro_noise);
  ROS_INFO("gyro bias noise: %.10f", IMUState::gyro_bias_noise);
  ROS_INFO("acc noise: %.10f", IMUState::acc_noise);
  ROS_INFO("acc bias noise: %.10f", IMUState::acc_bias_noise);
  ROS_INFO("observation noise: %.10f", Feature::observation_noise);
  ROS_INFO("initial velocity: %f, %f, %f",
      state_server.imu_state.velocity(0),
      state_server.imu_state.velocity(1),
      state_server.imu_state.velocity(2));
  ROS_INFO("initial gyro bias cov: %f", gyro_bias_cov);
  ROS_INFO("initial acc bias cov: %f", acc_bias_cov);
  ROS_INFO("initial velocity cov: %f", velocity_cov);
  ROS_INFO("initial extrinsic rotation cov: %f",
      extrinsic_rotation_cov);
  ROS_INFO("initial extrinsic translation cov: %f",
      extrinsic_translation_cov);

  cout << T_imu_cam0.linear() << endl;
  cout << T_imu_cam0.translation().transpose() << endl;

  ROS_INFO("max camera state #: %d", max_cam_state_size);
  ROS_INFO("===========================================");
  return true;
}

bool MsckfVio::createRosIO() {
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);
  feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
      "feature_point_cloud", 10);

  reset_srv = nh.advertiseService("reset",
      &MsckfVio::resetCallback, this);
  map_pub = nh.advertise<sensor_msgs::PointCloud2>("mapping_point",10);//优化后的点
  pubEdge = nh.advertise<sensor_msgs::PointCloud2>("edge_points",1);


  imu_sub = nh.subscribe("imu", 200,
      &MsckfVio::imuCallback, this);
  // feature_sub = nh.subscribe("features", 50,
  //     &MsckfVio::featureCallback, this);
  tof_sub = nh.subscribe("/royale_cam0/point_cloud_0", 40,&MsckfVio::tofCallback, this);

  mocap_odom_sub = nh.subscribe("mocap_odom", 10,
      &MsckfVio::mocapOdomCallback, this);
  mocap_odom_pub = nh.advertise<nav_msgs::Odometry>("gt_odom", 1);

  return true;
}

bool MsckfVio::initialize() {
  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");

  // Initialize state server
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  }

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}

void MsckfVio::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {

  // IMU msgs are pushed backed into a buffer instead of
  // being processed immediately. The IMU msgs are processed
  // when the next image is available, in which way, we can
  // easily handle the transfer delay.
  imu_msg_buffer.push_back(*msg);

  if (!is_gravity_set) {
    if (imu_msg_buffer.size() < 200) return;
    //if (imu_msg_buffer.size() < 10) return;
    initializeGravityAndBias();
    is_gravity_set = true;
  }

  return;
}

void MsckfVio::initializeGravityAndBias() {//

  // Initialize gravity and gyro bias.
  Vector3d sum_angular_vel = Vector3d::Zero();
  Vector3d sum_linear_acc = Vector3d::Zero();

  for (const auto& imu_msg : imu_msg_buffer) {
    Vector3d angular_vel = Vector3d::Zero();
    Vector3d linear_acc = Vector3d::Zero();

    tf::vectorMsgToEigen(imu_msg.angular_velocity, angular_vel);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, linear_acc);

    sum_angular_vel += angular_vel;
    sum_linear_acc += linear_acc;
  }

  state_server.imu_state.gyro_bias =
    sum_angular_vel / imu_msg_buffer.size();
  //IMUState::gravity =
  //  -sum_linear_acc / imu_msg_buffer.size();
  // This is the gravity in the IMU frame.
  Vector3d gravity_imu =
    sum_linear_acc / imu_msg_buffer.size();

  // Initialize the initial orientation, so that the estimation
  // is consistent with the inertial frame.
  double gravity_norm = gravity_imu.norm();
  IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);

  Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
    gravity_imu, -IMUState::gravity);
  state_server.imu_state.orientation =
    rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

  return;
}

bool MsckfVio::resetCallback(
    std_srvs::Trigger::Request& req,
    std_srvs::Trigger::Response& res) {

  ROS_WARN("Start resetting msckf vio...");
  // Temporarily shutdown the subscribers to prevent the
  // state from updating.
  feature_sub.shutdown();
  imu_sub.shutdown();

  // Reset the IMU state.
  IMUState& imu_state = state_server.imu_state;
  imu_state.time = 0.0;
  imu_state.orientation = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position = Vector3d::Zero();
  imu_state.velocity = Vector3d::Zero();
  imu_state.gyro_bias = Vector3d::Zero();
  imu_state.acc_bias = Vector3d::Zero();
  imu_state.orientation_null = Vector4d(0.0, 0.0, 0.0, 1.0);
  imu_state.position_null = Vector3d::Zero();
  imu_state.velocity_null = Vector3d::Zero();

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Clear all exsiting features in the map.
  map_server.clear();

  // Clear the IMU msg buffer.
  imu_msg_buffer.clear();

  // Reset the starting flags.
  is_gravity_set = false;
  is_first_img = true;

  // Restart the subscribers.
  imu_sub = nh.subscribe("imu", 100,
      &MsckfVio::imuCallback, this);
  feature_sub = nh.subscribe("features", 40,
      &MsckfVio::featureCallback, this);
//modified
  //tof_sub = nh.subscribe("tof_point", 40,&MsckfVio::tofCallback, this);
      
  // TODO: When can the reset fail?
  res.success = true;
  ROS_WARN("Resetting msckf vio completed...");
  return true;
}

void MsckfVio::tofCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
  //0.判断FOV内是否有存在KDTree的点云，没有则进入LIO模式（todos）
  if (first_tof){
    initialize_tof(&msg);
  }
  //if(ikdtree.size() < 300) return;
  if (!is_gravity_set) return;
  //1.点线特征提取
  pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>());
  FeatureExtract(msg,output);
  //ROS_INFO("DONE FeatureExtract");
  batchImuProcessing(msg->header.stamp.toSec());//1.得到当前图像帧时间的imu积分后的状态
  //ROS_INFO("DONE batchImuProcessing");
   //4.预测部分，得到位姿和协方差矩阵以及后续的reset处理
  stateAugmentationLidar(msg->header.stamp.toSec());//2.对状态进行增广，包括计算新的相机状态cam、计算新的J、计算增广的协方差矩阵
  //ROS_INFO("DONE stateAugmentationLidar");
   
    //5.更新部分。更新了状态。
  LidarUpdate(output);
  ROS_INFO("DONE LidarUpdate");
  
  

    //6.发布位姿
  publish(msg->header.stamp);
  ROS_INFO("DONE publish");
  return;
}

void MsckfVio::initialize_tof(const sensor_msgs::PointCloud2::ConstPtr &msg){


  mapping_ptr->clear();//mapping_ptr
  mapping_ptr->header.frame_id = fixed_frame_id;
  mapping_ptr->height = 1;
  //ROS_INFO("map_mapped:%d",map_mapped.size());
  for(const auto& item : map_mapped){
    const auto feature = item.second;
    mapping_ptr->points.push_back(pcl::PointXYZ(
      feature(0), feature(1), feature(2)));
  }
  mapping_ptr->width = mapping_ptr->points.size();
  feats_down_size = mapping_ptr->points.size();
  //ROS_INFO("before:feats_down_size:%d",feats_down_size);

  /*** initialize the map kdtree ***/
  if(ikdtree.Root_Node == nullptr)//初始化地图kd树，直到第一次有5个特征点才开始建树
  {
      if(feats_down_size > 0)
      {
          //ikdtree.set_downsample_param(filter_size_map_min);//设置add_points时起作用的下采样参数,设为 0.1 米意味着每个下采样区域是一个边长为 0.1 米的立方体。在该区域内，KDT只保留一个代表点
          //点云够稀少了，不需要下采样。
          ikdtree.Build(mapping_ptr->points);//根据点云簇所有点的XYZ的范围来对树进行构建。
      }
      
  }//建树过程
  else{
    map_incremental_cam();
  }
  int tree_size = ikdtree.size();
  ROS_INFO("KD Tree Size: %d", tree_size);
  // BoxPointType tree_range = ikdtree.tree_range();
  //   ROS_INFO("Tree Range: Min(%.2f, %.2f, %.2f), Max(%.2f, %.2f, %.2f)",
  //            tree_range.vertex_min[0], tree_range.vertex_min[1], tree_range.vertex_min[2],
  //            tree_range.vertex_max[0], tree_range.vertex_max[1], tree_range.vertex_max[2]);
  map_pub.publish(mapping_ptr);
  //modified

}



void MsckfVio:: LidarUpdate(pcl::PointCloud<pcl::PointXYZ>::ConstPtr output){//是否需要锁定output？
//process:
  //获得lidar点云数据
  //将lidar转到世界下：state_server.tof_states.position，state_server.tof_states.orientation
  //获得lidar的世界点云。
  //对每个点云。寻找与其最近的五个点：Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);//在kd树中搜索，这个world点的最近的5个点，并且更新最近点向量 和其分别的距离。
  //根据五个点计算质点和其特征向量，如果满足线段条件point_selected_surf[i]
  //

  //1.计算z
  //这俩应该是被imu计算出来的Lidar位姿。
  size_t num_points = output->size();
  Vector3d t_l_w = state_server.tof_state.position;//t_w_l
  Matrix3d R_l_w = quaternionToRotation(state_server.tof_state.orientation).transpose();//R_l_w
  int size_cov =  state_server.cam_states.size();
  VectorXd z;//残差函数列表


  ROS_INFO("size_cov%d",size_cov);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(0, 33+6*size_cov);
  ROS_INFO("after:H.rows=%d,H.cols() =%d",H.rows(),H.cols());
  Nearest_Points.resize(num_points);


  // Eigen::MatrixXd exam = Eigen::MatrixXd::Zero(0, 33+6*size_cov);

  // string file_path = "/home/g/debug_ws/src/MSCKF_VIO_MONO/msckf_vio/output.txt"; // 替换为你希望保存的路径
  // ofstream file(file_path);

  // file << "********************exam:*************************" << endl;
  // file << exam << endl;
  // exam.conservativeResize(exam.rows()+3,exam.cols());
  // exam.block(0, 0, 3, exam.cols()).setZero();
  // exam.block(exam.rows() - 3, 27, 3, 3) = Matrix3d::Identity();
  // file << "********************after exam:*************************" << endl;
  // file << exam << endl;


  //vector<double> z;//残差函数列表
  //vector<Vector3d> q;
  //vector<Vector3d> ne; // 方向向量列表
  vector<bool>use_point;
  use_point.resize(num_points);
  int points_size = output->points.size();
  ROS_INFO("points_size!%d",points_size);

  for (int i = 0; i < points_size; i++){
    pcl::PointXYZ point_body = output->points[i];//对每一个雷达点，用state里的lidar状态，转到世界坐标系
    pcl::PointXYZ point_world;
    if (std::isnan(point_body.x) || std::isnan(point_body.y) || std::isnan(point_body.z))
    {
        //ROS_INFO("REMOVE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        continue;
    }

    //ROS_INFO("11111");
    //是否引用&point_body去改变output？ //NO
    Vector3d p_body(point_body.x, point_body.y, point_body.z);
  
    Vector3d p_global(R_l_w*p_body + t_l_w);//1.得到点从tof坐标系--->经过imu的位姿与外参--->点的世界坐标系表示
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);

    vector<float> pointSearchSqDis(5);//找到最近的两个点的距离，todo：或者五个点，通过PCA获得线向量
    //vector<PointVector>  Nearest_Points_li[10]; //此每个元素代表每个点的最近的五个点
    auto &points_near = Nearest_Points[i];//points_near是一个pointVector
    
      /** Find the closest surfaces in the map **/
   //问题所在：
   //ROS_INFO("Dealing with error");
   ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
   //ROS_INFO("we make it!!!");
    //在kd树中搜索，这个world点的最近的5个点，并且更新最近5个点向量points_near 和其分别的距离pointSearchSqDis
    point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 1 ? false : true;//是否满足五个点？是否距离大于5？，都满足才确认为可取点。  
    //最远距离小于1，且满足了5个点。满足了才继续
  
    //计算地图的点云线特征：
    //1.最近的五个点的均值cx，cy，cz。
    //2.通过计算五个点的协方差，得到这个均值点的线特征
    //3.此点的地图点既能找到对应的线特征（最大的特征值是否大于次大特征值的3倍），又满足小残差（s > 0.1），此点放入雅可比
    if (!point_selected_surf[i]) continue;

    cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
    cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
    //计算五个点的平均坐标
    float cx = 0,cy = 0 ,cz = 0;
    for(int j = 0 ;j<5 ; j++){
      cx += points_near[j].x;//读取points_near[j]，等于读取某个pointXYZ点
      cy += points_near[j].y;
      cz += points_near[j].z;
    }
    
    cx /= 5; cy /= 5;  cz /= 5;//计算出平均坐标
    Vector3d q_map(cx,cy,cz);
    //ROS_INFO("44444");
    //这里通过累加每个点与中心点的坐标差的乘积，计算出协方差矩阵的各个元素，并将结果除以5得到平均值。
    float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
    for (int j = 0; j < 5; j++) {
        float ax = points_near[j].x - cx;
        float ay = points_near[j].y - cy;
        float az = points_near[j].z - cz;

        a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
        a22 += ay * ay; a23 += ay * az;
        a33 += az * az;
    }
    a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;//计算出中心点的协方差矩阵

    matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
    matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
    matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;
    //最大特征值对应的特征向量即为线特征的方向向量。
    cv::eigen(matA1, matD1, matV1);//这行代码计算 matA1 的特征值（存储在 matD1）和特征向量（存储在 matV1）

    //既满足线特征（最大的特征值是否大于次大特征值的3倍），又满足小残差（s > 0.1），此点放入雅可比
    if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {//这行代码检查最大的特征值是否大于次大特征值的3倍。这个条件用于判断点的分布是否是线性的。
      float x0 = point_world.x;
      float y0 = point_world.y;
      float z0 = point_world.z;
      float x1 = cx + 0.1 * matV1.at<float>(0, 0);
      float y1 = cy + 0.1 * matV1.at<float>(0, 1);
      float z1 = cz + 0.1 * matV1.at<float>(0, 2);
      float x2 = cx - 0.1 * matV1.at<float>(0, 0);
      float y2 = cy - 0.1 * matV1.at<float>(0, 1);
      float z2 = cz - 0.1 * matV1.at<float>(0, 2);
      
      //获得单位向量n，以及其斜对称矩阵：
      
      float dx = x2 - x1;
      float dy = y2 - y1;
      float dz = z2 - z1;
      float length = sqrt(dx * dx + dy * dy + dz * dz);

      float ux = dx / length;
      float uy = dy / length;
      float uz = dz / length;
      Vector3d n(ux,uy,uz);//n为单位向量
      Eigen::Matrix3d n_skew;
      Eigen::Matrix3d p_skew;
      n_skew <<  0, -uz,  uy, 
                uz,   0, -ux, 
              -uy,  ux,   0; // n_skew为其斜对称矩阵
      p_skew << 0,      -p_body[2], p_body[1],
                p_body[2],   0,     -p_body[0],
              -p_body[1],  p_body[0],     0;
      //获得距离残差向量v3d z
      Eigen::Vector3d diffVector(point_world.x - cx, point_world.y - cy, point_world.z - cz); // 当前点与最近点均值cx的差
      Eigen::Vector3d residualVector = n_skew * diffVector; // 残差向量

      // 输出残差向量
      float residual_x = residualVector(0);
      float residual_y = residualVector(1);
      float residual_z = residualVector(2);

      float dis_edge = sqrt(residual_x*residual_x +residual_y*residual_y+residual_z*residual_z);
      float s = 1 - 0.9 * fabs(dis_edge);//残差太大扔掉
      //ROS_INFO("total_res=%f ,res_x:%f,res_y:%f,res_z:%f",dis_edge,residual_x,residual_y,residual_z);

      if (s > 0.1) {
        //ROS_INFO("begin a H!");
        Eigen::Vector3d newElements;
        newElements<< residual_x,residual_y,residual_z;
          //ROS_INFO("b444!");
        use_point[i] = true;
        // ROS_INFO("b333!");
        int current_size = z.size();
        z.conservativeResize(current_size + 3);
        z.tail(3) = newElements; //这样z就是观测数据了。每次都往里每加3个残差
        //ROS_INFO("begin a z!");
        Eigen::Matrix3d newBlock;
        
        
        newBlock << n_skew *R_l_w* p_skew;
        //ROS_INFO("before:H.rows=%d,H.cols() =%d",H.rows(),H.cols());
        int oldrows = H.rows();
                  
        //string file_path = "/home/g/debug_ws/src/MSCKF_VIO_MONO/msckf_vio/output.txt"; // 替换为你希望保存的路径
        //ofstream file(file_path);

        //file << "********************first H:*************************" << endl;
        //file << H << endl;

        // file.close();
        H.conservativeResize(oldrows + 3, H.cols());
        //file << "********************Sec H:*************************" << endl;
        //file << H << endl;
        H.block(oldrows, 0, 3, H.cols()).setZero();//垃圾值置为0
        //file << "********************3 H:*************************" << endl;
        //file << H << endl;
        //ROS_INFO("after:H.rows=%d,H.cols() =%d",H.rows(),H.cols());
        H.block(H.rows() - 3, 27, 3, 3) = newBlock;
        H.block(H.rows() - 3, 30, 3, 3) = n_skew;
        // file << "********************last H:*************************" << endl;
        // file << H << endl;

         //file.close();
      }  
    //ROS_INFO("go to next!");
    }
  }




  ROS_INFO("H.rows/cols=%d/%d,z.size:%d,kdtree:%d",H.rows(),H.cols(),z.size(), ikdtree.size() );
  
  // //打开文件
  // string file_path = "/home/g/debug_ws/src/MSCKF_VIO_MONO/msckf_vio/output.txt"; // 替换为你希望保存的路径
  // ofstream file(file_path);

  // file << "********************Con H matrix:*************************" << endl;
  // file << H << endl;

  // file << "*********************************Con z matrix:***********************" << endl;
  // file << z << endl;


  // // file << "Con S matrix:" << endl;
  // // file << S << endl;

  // // // 输出 imu_state
  // // file << "imu_state.orientation: " << state_server.imu_state.orientation.transpose() << endl;
  // // file << "imu_state.gyro_bias: " << state_server.imu_state.gyro_bias.transpose() << endl;
  // // file << "imu_state.velocity: " << state_server.imu_state.velocity.transpose() << endl;
  // // file << "imu_state.acc_bias: " << state_server.imu_state.acc_bias.transpose() << endl;
  // // file << "imu_state.position: " << state_server.imu_state.position.transpose() << endl;

  // // 关闭文件
  // file.close();
  //获得了H和z之后，计算K与P
  LidarMeasurementUpdate(H,z);


  return ;
}






void MsckfVio::LidarMeasurementUpdate(const MatrixXd& H, const VectorXd& r){
  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(33+state_server.cam_states.size()*6);
    r_thin = r_temp.head(33+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;//协方差矩阵原封不动
   MatrixXd TT = H_thin*P*H_thin.transpose();


  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(H_thin.rows(), H_thin.rows());//HPH^T
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);//// 使用 LDLT 分解法求解这个方程，计算 K 的转置
  MatrixXd K = K_transpose.transpose();//再求得K。，

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

 
  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<27>();
  //VectorXd delta_x_imu_tmp = delta_x.head<21>();


  //如果太大了需要返回吗？todos：

  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
    ROS_WARN("Update change is too large.");
   //delta_x_imu_tmp *= 0.1;
    //const VectorXd& delta_x_imu = &delta_x_imu_tmp;
    //return;
  }
  
  // // 打开文件
  string file_path = "/home/g/debug_ws/src/MSCKF_VIO_MONO/msckf_vio/output.txt"; // 替换为你希望保存的路径
  ofstream file(file_path,ios::app);

   file << "********************Con delta_x matrix:*************************" << times << endl;
  file << delta_x.head<15>()  << endl;
  times ++;
  //  file << "*********************************Con H_thin matrix:***********************" << endl;
  //  file << H_thin << endl;

  // file << "*******************************8Con TT matrix:***************************" << endl;
  // file << TT << endl;
  // // file << "Con S matrix:" << endl;
  // // file << S << endl;

  // // // 输出 imu_state
  // // file << "imu_state.orientation: " << state_server.imu_state.orientation.transpose() << endl;
  // // file << "imu_state.gyro_bias: " << state_server.imu_state.gyro_bias.transpose() << endl;
  // // file << "imu_state.velocity: " << state_server.imu_state.velocity.transpose() << endl;
  // // file << "imu_state.acc_bias: " << state_server.imu_state.acc_bias.transpose() << endl;
  // // file << "imu_state.position: " << state_server.imu_state.position.transpose() << endl;

  // // 关闭文件
  file.close();


  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  state_server.imu_state.position += delta_x_imu.segment<3>(12);

  // const Vector4d dq_extrinsic =smallAngleQuaternion(delta_x_imu.segment<3>(15));
  // state_server.imu_state.R_imu_cam0 = quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  // state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  const Vector4d dq_extrinsic =smallAngleQuaternion(delta_x_imu.segment<3>(21));
  state_server.imu_state.R_imu_lidar = quaternionToRotation(dq_extrinsic) * state_server.imu_state.R_imu_lidar;
  state_server.imu_state.t_lidar_imu += delta_x_imu.segment<3>(24);


  //更新所有时刻的雷达的位姿。
  const VectorXd& delta_x_tof = delta_x.segment<6>(27);
  const Vector4d dq_tof = smallAngleQuaternion(delta_x_tof.head<3>());
  state_server.tof_state.orientation = quaternionMultiplication(
      dq_tof, state_server.tof_state.orientation);
  state_server.tof_state.position +=delta_x_tof.tail<3>(); 


  // Update the camera states.//更新所有时刻的相机位姿。
  // auto cam_state_iter = state_server.tof_states.begin();
  // for (int i = 0; i < state_server.tof_states.size();
  //     ++i, ++cam_state_iter) {
  //   const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
  //   const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
  //   cam_state_iter->second.orientation = quaternionMultiplication(
  //       dq_cam, cam_state_iter->second.orientation);
  //   cam_state_iter->second.position += delta_x_cam.tail<3>();
  // }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;

}

void MsckfVio::stateAugmentationLidar(const double& time)
{
  const Matrix3d& R_i_l = state_server.imu_state.R_imu_lidar;//读取旋转矩阵外参 i->c的变换
  const Vector3d& t_l_i = state_server.imu_state.t_lidar_imu;//读取平移外参 c-i的平移
  Matrix3d R_w_i = quaternionToRotation(state_server.imu_state.orientation);//得到world->imu的旋转矩阵

  Matrix3d R_w_l = R_i_l * R_w_i;//RLI * RIW = RLW ——W到L的旋转矩阵
  Vector3d t_l_w = state_server.imu_state.position + R_w_i.transpose()*t_l_i;
//2.通过imu和外参矩阵，间接地更新tof的位姿
  state_server.tof_state.time = time;
  state_server.tof_state.orientation = rotationToQuaternion(R_w_l);//设置world到lidar的旋转
  state_server.tof_state.position = t_l_w;//lidar到world的平移

  state_server.tof_state.orientation_null = state_server.tof_state.orientation;
  state_server.tof_state.position_null = state_server.tof_state.position;


  //3.计算B：
  Matrix<double, 6, 27> B = Matrix<double, 6, 27>::Zero();

  B.block<3, 3>(0, 0) = R_i_l;//左上角的3*3为i到c坐标的旋转（外参状态？）
  B.block<3, 3>(0, 21) = Matrix3d::Identity();//起始为第0行的第15列，identity是3*3单位矩阵
  B.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_l_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  B.block<3, 3>(3, 12) = Matrix3d::Identity();
  // B.block<3, 3>(3, 24) = Matrix3d::Identity();
  B.block<3, 3>(3, 24) = R_w_i.transpose();//原为上面，但我求的为R

  //4.提取原来协方差矩阵里的p11部分和p1c、pcc部分。
  // Rename some matrix blocks for convenience.

  size_t old_rows = state_server.state_cov.rows();//old_rows = 27+6+6N
  size_t old_cols = state_server.state_cov.cols();
  const Matrix<double, 27, 27>& Pii = state_server.state_cov.block<27, 27>(0, 0);
  const MatrixXd& Pic =state_server.state_cov.block(0, 33, 27, old_cols-33);
  const MatrixXd& Pci = Pic.transpose();
  const MatrixXd& Pcc =state_server.state_cov.block(33, 33, old_cols-33, old_cols-33);

  //5.更新协方差：
  // Fill in the augmented state covariance.
  state_server.state_cov.block(0, 27, 27, 6) = Pii*B.transpose();
  state_server.state_cov.block(27, 0, 6, 27) = B*Pii;
  state_server.state_cov.block(27, 27, 6, 6) = B*Pii*B.transpose();
  state_server.state_cov.block(27, 33, 6, old_cols-33) = B*Pic;
  state_server.state_cov.block(33, 27, old_cols-33, 6) = Pci*B.transpose();

  //上述协方差的变化为：
  // PII  PIC  PIC              PII    Pii*B^T    PIC

  // PLI  PLL  PLC      --->    B*Pii  B*PII*B^T  B*PIC

  // PCI  PCL  PCC              PCI    PCI*B^T    PCC
  //6.确保协方差矩阵对称
  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;

}



















void MsckfVio::FeatureExtract(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,pcl::PointCloud<pcl::PointXYZ>::Ptr output)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg,*cloud);//函数的参数要么复制副本要么直接用本体，效率来说用本体（指针解引用）更好。
    pcl::PointCloud<pcl::PointXYZ>::Ptr corner_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr PureEdge_points(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr edgeless_points(new pcl::PointCloud<pcl::PointXYZ>);    
    float cloudCurvature [250][250];
    int cloudSortInd [250][250];
    float cloudNeighborPicked [250][250];
    float cloudLabel [250][250];

    //二。curvature calculate
    vector<int> curvatureList ;
    for (int u=3; u<(pc_height-3);u++)//u代表行
    {
        for (int v=3;v<(pc_width-3);v++)//v代表列
        {   
            pcl::PointXYZ pt = cloud->at(u*pc_width+v);
            if(pt.z==pt.z)
            {
                float diffX = cloud->at((v - 3)+u*pc_width).x + cloud->at((v - 2)+u*pc_width).x + cloud->at((v - 1)+u*pc_width).x  - 6 * cloud->at(v+u*pc_width).x + cloud->at(v+1+u*pc_width).x + cloud->at(v+2+u*pc_width).x + cloud->at(v+3+u*pc_width).x ;
                float diffY = cloud->at((v - 3)+u*pc_width).y + cloud->at((v - 2)+u*pc_width).y + cloud->at((v - 1)+u*pc_width).y  - 6 * cloud->at(v+u*pc_width).y + cloud->at(v+1+u*pc_width).y + cloud->at(v+2+u*pc_width).y + cloud->at(v+3+u*pc_width).y ;
                float diffZ = cloud->at((v - 3)+u*pc_width).z + cloud->at((v - 2)+u*pc_width).z + cloud->at((v - 1)+u*pc_width).z  - 6 * cloud->at(v+u*pc_width).z + cloud->at(v+1+u*pc_width).z + cloud->at(v+2+u*pc_width).z + cloud->at(v+3+u*pc_width).z ;
                
                cloudCurvature[u][v] = diffX * diffX + diffY * diffY + diffZ * diffZ;//计算每个点的曲率，通过前后3个点总共6个点。
                cloudSortInd[u][v] = v;//记住其在每行的索引
                cloudNeighborPicked[u][v] = 0;//初始化其邻居
                cloudLabel[u][v] = 0; //初始化其标签           
            }
            
            
        }
        
        std::sort(&cloudSortInd[u][3], &cloudSortInd[u][pc_width - 3],
            [u, &cloudCurvature](int i, int j) {
                return cloudCurvature[u][i] > cloudCurvature[u][j];
            });
        //将每行的曲率进行排序，大的曲率点索引在前，注意只是将cloud里的索引数字根据其对应的曲率进行排序，大的排在前面


    }
    frame ++;
    //ROS_INFO("Frame:%i th",frame);    
    // for (int u = 5; u < pc_height-5; u++) {
    //     for (int v = 5; v < pc_width-5; v++) {
    //         std::cout << cloudSortInd[u][v] << " ";
    //     }
    //     std::cout << std::endl; // 换行打印下一行
    // }    


   
    // for (int u = 5; u < pc_height-5; u++) {
    //     for (int v = 5; v < pc_width-5; v++) {
    //         std::cout << cloudCurvature[u][v] << " ";
    //     }
    //     std::cout << std::endl; // 换行打印下一行
    // }

    for (int u=3; u<(pc_height-3);u++)//u代表行
    {
        int largestPickedNum = 0;//每行都要初始化，记录角点的挑选数

        for (int v=3;v<(pc_width-3);v++)//v代表列
        {   

            int ind = cloudSortInd[u][v];//从头开始取，意味着从大曲率开始取,得到其在第u行的索引
            if(cloudCurvature[u][ind]> 0.002 &&cloudNeighborPicked[u][ind] == 0)
            {//如果周围没被挑选过，且其曲率大于0.1
             largestPickedNum++;
             if(largestPickedNum<=2)//如果当前行的角点小于2
             {
                cloudLabel[u][ind] = 2;//给其贴上标签，2为角
                corner_points->push_back(cloud->at(ind+u*pc_width));//存入角点点云
                edge_points->push_back(cloud->at(ind+u*pc_width));//存入疑似边缘点云
             }
             else if (largestPickedNum<=20)
             {
                cloudLabel[u][ind] = 1;//标签为边。
                edge_points->push_back(cloud->at(ind+u*pc_width));
                PureEdge_points->push_back(cloud->at(ind+u*pc_width));//纯粹边点
                //edgeless_points->push_back(cloud->at(ind+u*pc_width));

             }
             else
             {
                break;//超过20进入下一个u
             }
             cloudNeighborPicked[u][ind] = 1;
             //避免相邻点重复处理，对每个点前后进行标记，也就是角点处理完后，其周围点都要被标记一次。
            for (int l = 1; l <= 3; l++)
            {
                float diffX = cloud->at(ind+l +u*pc_width).x - cloud->at(ind+l-1  +u*pc_width).x;
                float diffY = cloud->at(ind+l +u*pc_width).y - cloud->at(ind+l-1  +u*pc_width).y;
                float diffZ = cloud->at(ind+l +u*pc_width).z - cloud->at(ind+l-1  +u*pc_width).z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                {
                    break;
                }

                cloudNeighborPicked[u][ind+l] = 1;
            }
            for (int l = -1; l >= -3; l--)
            {
                float diffX = cloud->at(ind+l +u*pc_width).x - cloud->at(ind+ l + 1  +u*pc_width).x;
                float diffY = cloud->at(ind+l +u*pc_width).y - cloud->at(ind+ l + 1  +u*pc_width).y;
                float diffZ = cloud->at(ind+l +u*pc_width).z - cloud->at(ind+ l + 1  +u*pc_width).z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                {
                    break;
                }

                cloudNeighborPicked[u][ind+l] = 1;
            }            


            }
        }
    }
    int pointsCorner = corner_points->size();
    int pointsEdge = PureEdge_points->size();
    //ROS_INFO("corner points:%i",pointsCorner);
    //ROS_INFO("edge points:%i",pointsEdge);

    // sensor_msgs::PointCloud2 laserCloudOutMsg;
    // pcl::toROSMsg(*cloud, laserCloudOutMsg);
    // laserCloudOutMsg.header.stamp = cloud_msg->header.stamp;
    // laserCloudOutMsg.header.frame_id = cloud_msg->header.frame_id;
    // pub.publish(laserCloudOutMsg);//正常点云

    // sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    // pcl::toROSMsg(*corner_points, cornerPointsSharpMsg);
    // cornerPointsSharpMsg.header.stamp = cloud_msg->header.stamp;
    // cornerPointsSharpMsg.header.frame_id = cloud_msg->header.frame_id;
    // pubCorner.publish(cornerPointsSharpMsg);//纯粹角点

    // sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    // pcl::toROSMsg(*PureEdge_points, cornerPointsLessSharpMsg);
    // cornerPointsLessSharpMsg.header.stamp = cloud_msg->header.stamp;
    // cornerPointsLessSharpMsg.header.frame_id = cloud_msg->header.frame_id;
    // publessEdge.publish(cornerPointsLessSharpMsg);//纯粹边

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(*edge_points, surfPointsFlat2);
    surfPointsFlat2.header.stamp = cloud_msg->header.stamp;
    surfPointsFlat2.header.frame_id = "world";
    pubEdge.publish(surfPointsFlat2);//边与角
    *output = *edge_points;



}




//检查点云是否转换成功
// void MsckfVio::FeatureExtract(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg, pcl::PointCloud<pcl::PointXYZ>::Ptr output) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::fromROSMsg(*cloud_msg, *cloud);  // 函数的参数要么复制副本要么直接用本体，效率来说用本体（指针解引用）更好。

//     // 定义从雷达坐标系到IMU坐标系的变换矩阵
//     Eigen::Isometry3d T_imu_lidar = Eigen::Isometry3d::Identity();
//     // T_imu_lidar << 0.99, 0, 0, 1,
//     //                0, 0.99, 0, 2,
//     //                0, 0, 0.99, 3,
//     //                0, 0, 0, 1;
//     T_imu_lidar.linear() = state_server.imu_state.R_imu_lidar.transpose();//T_imu_lidar：从lidar到imu
//     T_imu_lidar.translation() = state_server.imu_state.t_lidar_imu;
//     Eigen::Matrix4f T_imu_lidar_matrix = T_imu_lidar.matrix().cast<float>();

    
//     // 对点云进行变换
//     pcl::transformPointCloud(*cloud, *cloud, T_imu_lidar_matrix);

//     pcl::PointCloud<pcl::PointXYZ>::Ptr corner_points(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr PureEdge_points(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr edge_points(new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::PointCloud<pcl::PointXYZ>::Ptr edgeless_points(new pcl::PointCloud<pcl::PointXYZ>);
    
//     float cloudCurvature[250][250];
//     int cloudSortInd[250][250];
//     float cloudNeighborPicked[250][250];
//     float cloudLabel[250][250];
    
//     // 二、curvature calculate
//     vector<int> curvatureList;
//     for (int u = 3; u < (pc_height - 3); u++) {  // u代表行
//         for (int v = 3; v < (pc_width - 3); v++) {  // v代表列
//             pcl::PointXYZ pt = cloud->at(u * pc_width + v);
//             if (pt.z == pt.z) {
//                 float diffX = cloud->at((v - 3) + u * pc_width).x + cloud->at((v - 2) + u * pc_width).x + cloud->at((v - 1) + u * pc_width).x - 6 * cloud->at(v + u * pc_width).x + cloud->at(v + 1 + u * pc_width).x + cloud->at(v + 2 + u * pc_width).x + cloud->at(v + 3 + u * pc_width).x;
//                 float diffY = cloud->at((v - 3) + u * pc_width).y + cloud->at((v - 2) + u * pc_width).y + cloud->at((v - 1) + u * pc_width).y - 6 * cloud->at(v + u * pc_width).y + cloud->at(v + 1 + u * pc_width).y + cloud->at(v + 2 + u * pc_width).y + cloud->at(v + 3 + u * pc_width).y;
//                 float diffZ = cloud->at((v - 3) + u * pc_width).z + cloud->at((v - 2) + u * pc_width).z + cloud->at((v - 1) + u * pc_width).z - 6 * cloud->at(v + u * pc_width).z + cloud->at(v + 1 + u * pc_width).z + cloud->at(v + 2 + u * pc_width).z + cloud->at(v + 3 + u * pc_width).z;
                
//                 cloudCurvature[u][v] = diffX * diffX + diffY * diffY + diffZ * diffZ;  // 计算每个点的曲率，通过前后3个点总共6个点。
//                 cloudSortInd[u][v] = v;  // 记住其在每行的索引
//                 cloudNeighborPicked[u][v] = 0;  // 初始化其邻居
//                 cloudLabel[u][v] = 0;  // 初始化其标签           
//             }
//         }
        
//         std::sort(&cloudSortInd[u][3], &cloudSortInd[u][pc_width - 3], [u, &cloudCurvature](int i, int j) {
//             return cloudCurvature[u][i] > cloudCurvature[u][j];
//         });
//         // 将每行的曲率进行排序，大的曲率点索引在前，注意只是将cloud里的索引数字根据其对应的曲率进行排序，大的排在前面
//     }
    
//     frame++;
    
//     for (int u = 3; u < (pc_height - 3); u++) {  // u代表行
//         int largestPickedNum = 0;  // 每行都要初始化，记录角点的挑选数

//         for (int v = 3; v < (pc_width - 3); v++) {  // v代表列
//             int ind = cloudSortInd[u][v];  // 从头开始取，意味着从大曲率开始取,得到其在第u行的索引
//             if (cloudCurvature[u][ind] > 0.002 && cloudNeighborPicked[u][ind] == 0) {  // 如果周围没被挑选过，且其曲率大于0.1
//                 largestPickedNum++;
//                 if (largestPickedNum <= 2) {  // 如果当前行的角点小于2
//                     cloudLabel[u][ind] = 2;  // 给其贴上标签，2为角
//                     corner_points->push_back(cloud->at(ind + u * pc_width));  // 存入角点点云
//                     edge_points->push_back(cloud->at(ind + u * pc_width));  // 存入疑似边缘点云
//                 } else if (largestPickedNum <= 20) {
//                     cloudLabel[u][ind] = 1;  // 标签为边。
//                     edge_points->push_back(cloud->at(ind + u * pc_width));
//                     PureEdge_points->push_back(cloud->at(ind + u * pc_width));  // 纯粹边点
//                 } else {
//                     break;  // 超过20进入下一个u
//                 }
//                 cloudNeighborPicked[u][ind] = 1;
//                 // 避免相邻点重复处理，对每个点前后进行标记，也就是角点处理完后，其周围点都要被标记一次。
//                 for (int l = 1; l <= 3; l++) {
//                     float diffX = cloud->at(ind + l + u * pc_width).x - cloud->at(ind + l - 1 + u * pc_width).x;
//                     float diffY = cloud->at(ind + l + u * pc_width).y - cloud->at(ind + l - 1 + u * pc_width).y;
//                     float diffZ = cloud->at(ind + l + u * pc_width).z - cloud->at(ind + l - 1 + u * pc_width).z;
//                     if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
//                         break;
//                     }
//                     cloudNeighborPicked[u][ind + l] = 1;
//                 }
//                 for (int l = -1; l >= -3; l--) {
//                     float diffX = cloud->at(ind + l + u * pc_width).x - cloud->at(ind + l + 1 + u * pc_width).x;
//                     float diffY = cloud->at(ind + l + u * pc_width).y - cloud->at(ind + l + 1 + u * pc_width).y;
//                     float diffZ = cloud->at(ind + l + u * pc_width).z - cloud->at(ind + l + 1 + u * pc_width).z;
//                     if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
//                         break;
//                     }
//                     cloudNeighborPicked[u][ind + l] = 1;
//                 }
//             }
//         }
//     }
    
//     int pointsCorner = corner_points->size();
//     int pointsEdge = PureEdge_points->size();
    
//     sensor_msgs::PointCloud2 surfPointsFlat2;
//     pcl::toROSMsg(*edge_points, surfPointsFlat2);
//     surfPointsFlat2.header.stamp = cloud_msg->header.stamp;
//     surfPointsFlat2.header.frame_id = "world";
//     pubEdge.publish(surfPointsFlat2);  // 边与角
//     *output = *edge_points;
// }


















void MsckfVio::featureCallback(
    const CameraMeasurementConstPtr& msg) {

  // Return if the gravity vector has not been set.
  if (!is_gravity_set) return;

  // Start the system if the first image is received.
  // The frame where the first image is received will be
  // the origin.
  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = msg->header.stamp.toSec();
  }

  static double max_processing_time = 0.0;
  static int critical_time_cntr = 0;
  double processing_start_time = ros::Time::now().toSec();

  // Propogate the IMU state.
  // that are received before the image msg.
  ros::Time start_time = ros::Time::now();
  batchImuProcessing(msg->header.stamp.toSec());
  double imu_processing_time = (
      ros::Time::now()-start_time).toSec();
  ROS_INFO("imu_processing_time:%f",imu_processing_time);
  // Augment the state vector.
  start_time = ros::Time::now();
  stateAugmentation(msg->header.stamp.toSec());
  double state_augmentation_time = (
      ros::Time::now()-start_time).toSec();
  // ROS_INFO("state_augmentation_time:%f",state_augmentation_time);
  // Add new observations for existing features or new
  // features in the map server.
  start_time = ros::Time::now();
  addFeatureObservations(msg);
  double add_observations_time = (
      ros::Time::now()-start_time).toSec();

  // Perform measurement update if necessary.
  start_time = ros::Time::now();
  removeLostFeatures();
  double remove_lost_features_time = (
      ros::Time::now()-start_time).toSec();
 //ROS_INFO("remove_lost_features_time:%f",remove_lost_features_time);
  start_time = ros::Time::now();
  pruneCamStateBuffer();
  double prune_cam_states_time = (
      ros::Time::now()-start_time).toSec();
  ROS_INFO("prune_cam_states_time:%f",prune_cam_states_time);
  // Publish the odometry.
  start_time = ros::Time::now();
  publish(msg->header.stamp);
  double publish_time = (
      ros::Time::now()-start_time).toSec();

  // Reset the system if necessary.
  onlineReset();

  double processing_end_time = ros::Time::now().toSec();
  double processing_time =
    processing_end_time - processing_start_time;
        ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
        processing_time, critical_time_cntr);
  if (processing_time > 1.0/frame_rate) {//如果时间大于你设置的帧率，比如30
    ++critical_time_cntr;
    ROS_INFO("\033[1;31mTotal processing time %f/%d...\033[0m",
        processing_time, critical_time_cntr);
    //printf("IMU processing time: %f/%f\n",
    //    imu_processing_time, imu_processing_time/processing_time);
    //printf("State augmentation time: %f/%f\n",
    //    state_augmentation_time, state_augmentation_time/processing_time);
    //printf("Add observations time: %f/%f\n",
    //    add_observations_time, add_observations_time/processing_time);
    printf("Remove lost features time: %f/%f\n",
        remove_lost_features_time, remove_lost_features_time/processing_time);
    printf("Remove camera states time: %f/%f\n",
        prune_cam_states_time, prune_cam_states_time/processing_time);
    //printf("Publish time: %f/%f\n",
    //    publish_time, publish_time/processing_time);
  }

  return;
}

void MsckfVio::mocapOdomCallback(
    const nav_msgs::OdometryConstPtr& msg) {
  static bool first_mocap_odom_msg = true;

  // If this is the first mocap odometry messsage, set
  // the initial frame.
  if (first_mocap_odom_msg) {
    Quaterniond orientation;
    Vector3d translation;
    tf::pointMsgToEigen(
        msg->pose.pose.position, translation);
    tf::quaternionMsgToEigen(
        msg->pose.pose.orientation, orientation);
    //tf::vectorMsgToEigen(
    //    msg->transform.translation, translation);
    //tf::quaternionMsgToEigen(
    //    msg->transform.rotation, orientation);
    mocap_initial_frame.linear() = orientation.toRotationMatrix();
    mocap_initial_frame.translation() = translation;
    first_mocap_odom_msg = false;
  }

  // Transform the ground truth.
  Quaterniond orientation;
  Vector3d translation;
  //tf::vectorMsgToEigen(
  //    msg->transform.translation, translation);
  //tf::quaternionMsgToEigen(
  //    msg->transform.rotation, orientation);
  tf::pointMsgToEigen(
      msg->pose.pose.position, translation);
  tf::quaternionMsgToEigen(
      msg->pose.pose.orientation, orientation);

  Eigen::Isometry3d T_b_v_gt;
  T_b_v_gt.linear() = orientation.toRotationMatrix();
  T_b_v_gt.translation() = translation;
  Eigen::Isometry3d T_b_w_gt = mocap_initial_frame.inverse() * T_b_v_gt;

  //Eigen::Vector3d body_velocity_gt;
  //tf::vectorMsgToEigen(msg->twist.twist.linear, body_velocity_gt);
  //body_velocity_gt = mocap_initial_frame.linear().transpose() *
  //  body_velocity_gt;

  // Ground truth tf.measurementJacobian
  if (publish_tf) {
    tf::Transform T_b_w_gt_tf;
    tf::transformEigenToTF(T_b_w_gt, T_b_w_gt_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_gt_tf, msg->header.stamp, fixed_frame_id, child_frame_id+"_mocap"));
  }

  // Ground truth odometry.
  nav_msgs::Odometry mocap_odom_msg;
  mocap_odom_msg.header.stamp = msg->header.stamp;
  mocap_odom_msg.header.frame_id = fixed_frame_id;
  mocap_odom_msg.child_frame_id = child_frame_id+"_mocap";

  tf::poseEigenToMsg(T_b_w_gt, mocap_odom_msg.pose.pose);
  //tf::vectorEigenToMsg(body_velocity_gt,
  //    mocap_odom_msg.twist.twist.linear);

  mocap_odom_pub.publish(mocap_odom_msg);
  return;
}

void MsckfVio::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  int used_imu_msg_cntr = 0;

  for (const auto& imu_msg : imu_msg_buffer) {
    double imu_time = imu_msg.header.stamp.toSec();
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    tf::vectorMsgToEigen(imu_msg.angular_velocity, m_gyro);
    tf::vectorMsgToEigen(imu_msg.linear_acceleration, m_acc);

    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);
    ++used_imu_msg_cntr;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  return;
}

void MsckfVio::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  double dtime = time - imu_state.time;
  ROS_INFO("dtime:%f",dtime);
  // Compute discrete transition and noise covariance matrix
  Matrix<double, 27, 27> F = Matrix<double, 27, 27>::Zero();
  Matrix<double, 27, 12> G = Matrix<double, 27, 12>::Zero();

  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(
      imu_state.orientation).transpose()*skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F.block<3, 3>(12, 6) = Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(
      imu_state.orientation).transpose();
  G.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order,
  // which can be considered to be accurate enough assuming
  // dtime is within 0.01s.
  Matrix<double, 27, 27> Fdt = F * dtime;
  Matrix<double, 27, 27> Fdt_square = Fdt * Fdt;
  Matrix<double, 27, 27> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 27, 27> Phi = Matrix<double, 27, 27>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;


  ros::Time start_time = ros::Time::now();


 
  // Propogate the state using 4th order Runge-Kutta
  predictNewState(dtime, gyro, acc);
    double predictNewState_time = (
      ros::Time::now()-start_time).toSec();
  //ROS_INFO("predictNewState_time:%f",predictNewState_time);



  // Modify the transition matrix
  Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
  Phi.block<3, 3>(0, 0) =
    quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

  Vector3d u = R_kk_1 * IMUState::gravity;
  RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

  Matrix3d A1 = Phi.block<3, 3>(6, 0);
  Vector3d w1 = skewSymmetric(
      imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
  Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

  Matrix3d A2 = Phi.block<3, 3>(12, 0);
  Vector3d w2 = skewSymmetric(
      dtime*imu_state.velocity_null+imu_state.position_null-
      imu_state.position) * IMUState::gravity;
  Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

  // Propogate the state covariance matrix.
  Matrix<double, 27, 27> Q = Phi*G*state_server.continuous_noise_cov*
  G.transpose()*Phi.transpose()*dtime;
  //Matrix<double, 21, 21> Q = G*state_server.continuous_noise_cov*
   // G.transpose()*dtime;
  state_server.state_cov.block<27, 27>(0, 0) = Phi*state_server.state_cov.block<27, 27>(0, 0)*Phi.transpose() + Q;

  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(0, 27, 27, state_server.state_cov.cols()-27) =  Phi * state_server.state_cov.block(0, 27, 27, state_server.state_cov.cols()-27);
    state_server.state_cov.block(27, 0, state_server.state_cov.rows()-27, 27) = state_server.state_cov.block(27, 0, state_server.state_cov.rows()-27, 27) * Phi.transpose();
  }

  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  // Update the state correspondes to null space.
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  //ROS_INFO("DOWN IMU propagation");
  return;
}

void MsckfVio::predictNewState(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +
    IMUState::gravity;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);
  //ROS_INFO("imu preprotein (px,py,pz):%f,%f,%f",p.x(),p.y(),p.z());

  return;
}

void MsckfVio::stateAugmentation(const double& time) {

  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

  // Add a new camera state to the state server.
  Matrix3d R_w_i = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i;
  Vector3d t_c_w = state_server.imu_state.position +
    R_w_i.transpose()*t_c_i;

  state_server.cam_states[state_server.imu_state.id] =
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  cam_state.position = t_c_w;

  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
  Matrix<double, 6, 27> J = Matrix<double, 6, 27>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Matrix3d::Identity();
  J.block<3, 3>(3, 0) = skewSymmetric(R_w_i.transpose()*t_c_i);
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i);
  J.block<3, 3>(3, 12) = Matrix3d::Identity();
  J.block<3, 3>(3, 18) = Matrix3d::Identity();

  // Resize the state covariance matrix.
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 27, 27>& P11 =
    state_server.state_cov.block<27, 27>(0, 0);
  const MatrixXd& P12 =
    state_server.state_cov.block(0, 27, 27, 6);
  const MatrixXd& P13 =
    state_server.state_cov.block(0, 33, 27, old_cols-33);
  //ROS_INFO("p11row: %lu p12row %lu p13cols %lu", P11.rows(), P12.rows(),P13.cols());
  // Fill in the augmented state covariance.
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11,J*P12,J*P13;
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;
  //ROS_INFO("Cam propagation done!");
  return;
}

void MsckfVio::addFeatureObservations(
    const CameraMeasurementConstPtr& msg) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : msg->features) {
    if (map_server.find(feature.id) == map_server.end()) {
      // This is a new feature.
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =
        Vector2d(feature.u0, feature.v0);
    } else {
      // This is an old feature.
      map_server[feature.id].observations[state_id] =
        Vector2d(feature.u0, feature.v0);
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}

void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 2, 6>& H_x, Matrix<double, 2, 3>& H_f, Vector2d& r) {
  
  //fp = fopen("/home/lxh/pw.txt","w");
  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose.
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // Cam1 pose.
  //Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
  //Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
  //Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector2d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  //Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);
  //printf("t_c0_w:%f %f %f\n",t_c0_w(0),t_c0_w(1),t_c0_w(2));
  //printf("p_w:%f %f %f\n",p_w(0),p_w(1),p_w(2));
  //printf("p_c0:%f %f %f\n",p_c0(0),p_c0(1),p_c0(2));
  //fprintf(fp,"%f %f %f\n",p_w(0),p_w(1),p_w(2));
  
  // Compute the Jacobians.
  Matrix<double, 2, 3> dz_dpc0 = Matrix<double, 2, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  //Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  //dz_dpc1(2, 0) = 1 / p_c1(2);
  //dz_dpc1(3, 1) = 1 / p_c1(2);
  //dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  //dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

 // Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  //dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
 // dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
 // Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dxc;// + dz_dpc1*dpc1_dxc;
  H_f = dz_dpc0*dpc0_dpg;// + dz_dpc1*dpc1_dpg;

  // Modifty the measurement Jacobian to ensure
  // observability constrain.
  Matrix<double, 2, 6> A = H_x;
  Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  u.block<3, 1>(0, 0) = quaternionToRotation(
      cam_state.orientation_null) * IMUState::gravity;
  u.block<3, 1>(3, 0) = skewSymmetric(
      p_w-cam_state.position_null) * IMUState::gravity;
  H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  H_f = -H_x.block<2, 3>(0, 3);

  // Compute the residual.
  r = z - Vector2d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2));
  
  Vector2d tmpv = Vector2d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2));
  //printf("\n\nLK res:%f %f\n",z(0),z(1));
  //printf("predict LK res:%f %f\n",tmpv(0),tmpv(1));
  //printf("reproject error:%f %f\n\n\n",r(0),r(1));
  
 

  return;
}

void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 2* valid_cam_state_ids.size();//4

  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      33+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& cam_id : valid_cam_state_ids) {

    Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
    Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
    Vector2d r_i = Vector2d::Zero();
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<2, 6>(stack_cntr, 33+6*cam_state_cntr) = H_xi;
    H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<2>(stack_cntr) = r_i;//4
    stack_cntr += 2;//4
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);

  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(33+state_server.cam_states.size()*6);
    r_thin = r_temp.head(33+state_server.cam_states.size()*6);

    //HouseholderQR<MatrixXd> qr_helper(H);
    //MatrixXd Q = qr_helper.householderQ();
    //MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    //H_thin = Q1.transpose() * H;
    //r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

 
  // Update the IMU state.
  const VectorXd& delta_x_imu = delta_x.head<27>();
  //VectorXd delta_x_imu_tmp = delta_x.head<21>();

  //ROS_WARN("IMU position delta:delta_x:%f,delta_y:%f,delta_z:%f",delta_x_imu.segment<3>(12)[0],delta_x_imu.segment<3>(12)[1],delta_x_imu.segment<3>(12)[2]);





  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
    ROS_WARN("Update change is too large.");
   //delta_x_imu_tmp *= 0.1;
    //const VectorXd& delta_x_imu = &delta_x_imu_tmp;
    //return;
  }
  


  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  state_server.imu_state.position += delta_x_imu.segment<3>(12);
  

  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  const Vector4d lidar_dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(21));
  state_server.imu_state.R_imu_lidar = quaternionToRotation(
      lidar_dq_extrinsic) * state_server.imu_state.R_imu_lidar;
  state_server.imu_state.t_lidar_imu += delta_x_imu.segment<3>(24);



  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(33+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}

bool MsckfVio::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  //cout << dof << " " << gamma << " " <<
  //  chi_squared_test_table[dof] << " ";

  if (gamma < chi_squared_test_table[dof]) {
    //cout << "passed" << endl;
    return true;
  } else {
    //cout << "failed" << endl;
    return false;
  }
}

void MsckfVio::removeLostFeatures() {

  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> processed_feature_ids(0);
  //检查所有map_server的点，满足的点放入processed_feature_ids
  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // Pass the features that are still being tracked.
    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }

    jacobian_row_size += 2*feature.observations.size() - 3;//4
    processed_feature_ids.push_back(feature.id);
  }

  //cout << "invalid/processed feature #: " <<
  //  invalid_feature_ids.size() << "/" <<
  //  processed_feature_ids.size() << endl;
  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  if (processed_feature_ids.size() == 0) return;

  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      33+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 2000) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  measurementUpdate(H_x, r);

  ros::Time start_time = ros::Time::now();

  
  Mapping map_mapped;
  //map_mapped.clear();//清空，确保mao_mapped保存最新一帧的更新完的地图点。而非累积地图点。
  // Remove all processed features from the map.
  //ROS_INFO("processed_feature_ids(%d)",processed_feature_ids.size());
  for (const auto& feature_id : processed_feature_ids){
    Eigen::Vector3d p_c0;
    Eigen::Vector3d p_c0_orin;
    auto& feature = map_server[feature_id];
    auto& First_id = feature.observations.begin()->first;
    
    p_c0 =  feature.position_aft;//每个特征点第一帧下的3D点坐标
    p_c0_orin = feature.position;
    auto cam_state_iter = state_server.cam_states.find(First_id);
    if (cam_state_iter == state_server.cam_states.end()) {
      
      //很多时候已经没有这一阵相机的pose了！那么就是用原来的点的位姿
       map_mapped[First_id] = p_c0_orin;
    }
    else{Eigen::Isometry3d cam0_pose;//齐次变换矩阵
    cam0_pose.linear() = quaternionToRotation(//linear代表旋转部分，用这个观测（u，v）所对应的相机姿态（cam_state_iter）的旋转
      cam_state_iter->second.orientation).transpose();
    cam0_pose.translation() = cam_state_iter->second.position;//矩阵的平移部分

    Eigen::Vector3d Mapped = cam0_pose.linear() * p_c0 +cam0_pose.translation();
    map_mapped[feature_id] = Mapped;//跟踪丢失的点会加入Mapped
    }

    //ROS_WARN("orgin points:(%f,%f,%f)",p_c0_orin.x(),p_c0_orin.y(),p_c0_orin.z());
             // ,Mapped.x(),Mapped.y(),Mapped.z());
    //ROS_INFO("Mapped(%lf,%lf,%lf)",Mapped.x(),Mapped.y(),Mapped.z());
    map_server.erase(feature_id);
    
  }
  double map_time = (ros::Time::now()-start_time).toSec();
  //ROS_INFO("map_time%lf",map_time);
  start_time = ros::Time::now();
  publish_cam_points(map_mapped);
  double publish_cam_points = (ros::Time::now()-start_time).toSec();
  //ROS_INFO("KDTREE_points%lf",publish_cam_points);
    
  
  return;
}

void MsckfVio::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();
  for (int i = 0; i < 2; ++i)//4
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;
  ++cam_state_iter;
  auto first_cam_state_iter = state_server.cam_states.begin();

  // Pose of the key camera state.
  const Vector3d key_position =
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    //if (angle < 0.1745 && distance < 0.2 && tracking_rate > 0.5) {
    if (angle < 0.2618 && distance < 0.4 && tracking_rate > 0.5) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}

void MsckfVio::pruneCamStateBuffer() {

  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  //ROS_WARN("we are in pruneCamStateBuffer!");
  // Find two camera states to be removed.
  vector<StateIDType> rm_cam_state_ids(0);
  findRedundantCamStates(rm_cam_state_ids);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }

    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 2*involved_cam_state_ids.size() - 3;//4
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      33+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);

  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 33 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  } 

  return;
}

int process_increments = 0;
void MsckfVio:: map_incremental_cam()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);//feats_down_size是地图尺寸，可能需要全局定义。
    //ROS_INFO("after:feats_down_size:%d",feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    Nearest_Points.resize(feats_down_size);

    //对每个地图点mapping_ptr->points[i]：计算其在地图里最近的五个点存入对应的Nearest_Points[i]，

    for (int i = 0; i < feats_down_size; i++)//对每个下采样后的点
    {   
        /* transform to world frame */
        //pointBodyToWorld(&(feats_down_body->points[i]), &(mapping_ptr->points[i]));
        /* decide if need add to map */
        //给每个点云进行最近邻搜索：
        PointType &point_world = mapping_ptr->points[i];
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);//创建一个长度为5的float向量
        
        auto &points_near = Nearest_Points[i];//points_near的改变会反馈给Nearest_Points[i]


        /** Find the closest surfaces in the map **/
        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);//在kd树中搜索，这个world点的最近的5个点，并且更新最近点向量 和其分别的距离。
        point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;//是否满足五个点？是否距离大于5？，都满足才确认为可取点。

        
        if (!point_selected_surf[i]) continue; 
        // 能在地图上找到五个点，并且最远点的距离小于5，则继续，否则跳过 
        
        point_selected_surf[i] = false;//及时将数组复原
    }//负责给Nearest_Points赋值

      for (int i = 0; i < feats_down_size; i++)//对每个下采样后的点
    {   
        if (!Nearest_Points[i].empty())//Nearest_Points不为空，说明周围有地图点云
        {
            const PointVector &points_near = Nearest_Points[i];//points_near代表着，第i个帧点云周围的地图点。
            bool need_add = true;
            //BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(mapping_ptr->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(mapping_ptr->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(mapping_ptr->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(mapping_ptr->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(mapping_ptr->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;//如果没找到5个则跳过
                if (calc_dist(points_near[readd_i], mid_point) < dist)//
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(mapping_ptr->points[i]);//需要加的点放入PointToAdd
        }
        else//如果该点云周围没有地图点云，则直接加入PointToAdd
        {
            PointToAdd.push_back(mapping_ptr->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    double kdtree_incremental_time = omp_get_wtime() - st_time;
}


float MsckfVio:: calc_dist(PointType p1, PointType p2){
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

void MsckfVio::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;

  ROS_WARN("Start %lld online reset procedure...",
      ++online_reset_counter);
  ROS_INFO("Stardard deviation in xyz: %f, %f, %f",
      position_x_std, position_y_std, position_z_std);

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;
  nh.param<double>("initial_covariance/velocity",
      velocity_cov, 0.25);
  nh.param<double>("initial_covariance/gyro_bias",
      gyro_bias_cov, 1e-4);
  nh.param<double>("initial_covariance/acc_bias",
      acc_bias_cov, 1e-2);

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  nh.param<double>("initial_covariance/extrinsic_rotation_cov",
      extrinsic_rotation_cov, 3.0462e-4);
  nh.param<double>("initial_covariance/extrinsic_translation_cov",
      extrinsic_translation_cov, 1e-4);

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  ROS_WARN("%lld online reset complete...", online_reset_counter);
  return;
}



void MsckfVio::publish_cam_points(Mapping map_mapped){
  mapping_ptr->clear();
  mapping_ptr->header.frame_id = fixed_frame_id;
  mapping_ptr->height = 1;
  //ROS_INFO("map_mapped:%d",map_mapped.size());
  for(const auto& item : map_mapped){
    const auto feature = item.second;
    mapping_ptr->points.push_back(pcl::PointXYZ(
      feature(0), feature(1), feature(2)));
  }
  mapping_ptr->width = mapping_ptr->points.size();
  feats_down_size = mapping_ptr->points.size();
  //ROS_INFO("before:feats_down_size:%d",feats_down_size);

  /*** initialize the map kdtree ***/
  if(ikdtree.Root_Node == nullptr)//初始化地图kd树，直到第一次有5个特征点才开始建树
  {
      if(feats_down_size > 0)
      {
          //ikdtree.set_downsample_param(filter_size_map_min);//设置add_points时起作用的下采样参数,设为 0.1 米意味着每个下采样区域是一个边长为 0.1 米的立方体。在该区域内，KDT只保留一个代表点
          //点云够稀少了，不需要下采样。
          ikdtree.Build(mapping_ptr->points);//根据点云簇所有点的XYZ的范围来对树进行构建。
      }
      
  }//建树过程
  else{
    map_incremental_cam();
  }
  int tree_size = ikdtree.size();
  ROS_INFO("KD Tree Size: %d", tree_size);
  // BoxPointType tree_range = ikdtree.tree_range();
  //   ROS_INFO("Tree Range: Min(%.2f, %.2f, %.2f), Max(%.2f, %.2f, %.2f)",
  //            tree_range.vertex_min[0], tree_range.vertex_min[1], tree_range.vertex_min[2],
  //            tree_range.vertex_max[0], tree_range.vertex_max[1], tree_range.vertex_max[2]);
  map_pub.publish(mapping_ptr);
  //modified


}



void MsckfVio::publish(const ros::Time& time) {

  // Convert the IMU frame to the body frame.
  const IMUState& imu_state = state_server.imu_state;
  const TofState& tof_state = state_server.tof_state;
  Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  T_i_w.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_i_w.translation() = imu_state.position;
  Vector3d translation = IMUState::T_imu_body.translation();
  // ROS_INFO(" state_imu:RC:%f,%f,%f,RL:%f,%f,%f——————Lidar,P:%f,%f,%f ",imu_state.t_cam0_imu.x(),imu_state.t_cam0_imu.y(),imu_state.t_cam0_imu.z(),
  // imu_state.t_lidar_imu.x(),imu_state.t_lidar_imu.y(),imu_state.t_lidar_imu.z(),
  // tof_state.position.x(),tof_state.position.y(),tof_state.position.z());

  //ROS_WARN(" T_i_w:%f,%f,%f ",imu_state.position.x(),imu_state.position.y(),imu_state.position.z());
  Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
    IMUState::T_imu_body.inverse();
  Eigen::Vector3d body_velocity =
    IMUState::T_imu_body.linear() * imu_state.velocity;

  // Publish tf
  if (publish_tf) {
    tf::Transform T_b_w_tf;
    tf::transformEigenToTF(T_b_w, T_b_w_tf);
    tf_pub.sendTransform(tf::StampedTransform(
          T_b_w_tf, time, fixed_frame_id, child_frame_id));
  }

  // Publish the odometry
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = time;
  odom_msg.header.frame_id = fixed_frame_id;
  odom_msg.child_frame_id = child_frame_id;

  tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
  tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);

  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 12);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(12, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(12, 12);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;

  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
  H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose *
    P_imu_pose * H_pose.transpose();

  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);

  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(6, 6);
  Matrix3d H_vel = IMUState::T_imu_body.linear();
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

  odom_pub.publish(odom_msg);











  // Publish the 3D positions of the features that
  // has been initialized.
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_msg_ptr(
      new pcl::PointCloud<pcl::PointXYZ>());
  feature_msg_ptr->header.frame_id = fixed_frame_id;
  feature_msg_ptr->height = 1;
  for (const auto& item : map_server) {
    const auto& feature = item.second;
    if (feature.is_initialized) {
      Vector3d feature_position =
        IMUState::T_imu_body.linear() * feature.position;
      feature_msg_ptr->points.push_back(pcl::PointXYZ(
            feature_position(0), feature_position(1), feature_position(2)));
    }
  }
  feature_msg_ptr->width = feature_msg_ptr->points.size();

  feature_pub.publish(feature_msg_ptr);

  return;
}

} // namespace msckf_vio

