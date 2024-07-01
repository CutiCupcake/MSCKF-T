/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_H
#define MSCKF_VIO_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <std_srvs/Trigger.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "msckf_vio/tof_state.h"//modified
#include "tof_state.h"
#include "imu_state.h"
#include "cam_state.h"
#include "feature.hpp"
#include <msckf_vio/CameraMeasurement.h>
#include <sensor_msgs/PointCloud2.h>
#include <msckf_vio/ikd_Tree.h>
using namespace Eigen;
namespace msckf_vio {
/*
 * @brief MsckfVio Implements the algorithm in
 *    Anatasios I. Mourikis, and Stergios I. Roumeliotis,
 *    "A Multi-State Constraint Kalman Filter for Vision-aided
 *    Inertial Navigation",
 *    http://www.ee.ucr.edu/~mourikis/tech_reports/TR_MSCKF.pdf
 */
class MsckfVio {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    MsckfVio(ros::NodeHandle& pnh);
    // Disable copy and assign constructor
    MsckfVio(const MsckfVio&) = delete;
    MsckfVio operator=(const MsckfVio&) = delete;

    // Destructor
    ~MsckfVio() {}

    /*
     * @brief initialize Initialize the VIO.
     */
    bool initialize();

    /*
     * @brief reset Resets the VIO to initial status.
     */
    void reset();

    typedef boost::shared_ptr<MsckfVio> Ptr;
    typedef boost::shared_ptr<const MsckfVio> ConstPtr;

  private:
    /*
     * @brief StateServer Store one IMU states and several
     *    camera states for constructing measurement
     *    model.
     */
    struct StateServer {
      IMUState imu_state;
      TofState tof_state;//（6*1）
      CamStateServer cam_states;

      // State covariance matrix
      Eigen::MatrixXd state_cov;
      Eigen::Matrix<double, 12, 12> continuous_noise_cov;
    };


    /*
     * @brief loadParameters
     *    Load parameters from the parameter server.
     */
    bool loadParameters();

    /*
     * @brief createRosIO
     *    Create ros publisher and subscirbers.
     */
    bool createRosIO();

    /*
     * @brief imuCallback
     *    Callback function for the imu message.
     * @param msg IMU msg.
     */
    void imuCallback(const sensor_msgs::ImuConstPtr& msg);

    /*
     * @brief featureCallback
     *    Callback function for feature measurements.
     * @param msg Stereo feature measurements.
     */
    void featureCallback(const CameraMeasurementConstPtr& msg);
    
    void tofCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
    /*
     * @brief publish Publish the results of VIO.
     * @param time The time stamp of output msgs.
     */
    void publish(const ros::Time& time);

    /*
     * @brief initializegravityAndBias
     *    Initialize the IMU bias and initial orientation
     *    based on the first few IMU readings.
     */
    void initializeGravityAndBias();

    /*
     * @biref resetCallback
     *    Callback function for the reset service.
     *    Note that this is NOT anytime-reset. This function should
     *    only be called before the sensor suite starts moving.
     *    e.g. while the robot is still on the ground.
     */
    bool resetCallback(std_srvs::Trigger::Request& req,
        std_srvs::Trigger::Response& res);

    // Filter related functions
    // Propogate the state
    void batchImuProcessing(
        const double& time_bound);
    void processModel(const double& time,
        const Eigen::Vector3d& m_gyro,
        const Eigen::Vector3d& m_acc);
    void predictNewState(const double& dt,
        const Eigen::Vector3d& gyro,
        const Eigen::Vector3d& acc);

    // Measurement update
    void stateAugmentation(const double& time);
    void addFeatureObservations(const CameraMeasurementConstPtr& msg);
    // This function is used to compute the measurement Jacobian
    // for a single feature observed at a single camera frame.
    void measurementJacobian(const StateIDType& cam_state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 3>& H_f,
        Eigen::Vector2d& r);
    // This function computes the Jacobian of all measurements viewed
    // in the given camera states of this feature.
    void featureJacobian(const FeatureIDType& feature_id,
        const std::vector<StateIDType>& cam_state_ids,
        Eigen::MatrixXd& H_x, Eigen::VectorXd& r);
    void measurementUpdate(const Eigen::MatrixXd& H,
        const Eigen::VectorXd& r);
    bool gatingTest(const Eigen::MatrixXd& H,
        const Eigen::VectorXd&r, const int& dof);
    void removeLostFeatures();
    void findRedundantCamStates(
        std::vector<StateIDType>& rm_cam_state_ids);
    void pruneCamStateBuffer();
    // Reset the system online if the uncertainty is too large.
    void onlineReset();
    void map_incremental_cam();
   
    typedef pcl::PointXYZ PointType;
    typedef pcl::PointCloud<PointType> PointCloudXYZ;
    
    typedef std::vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
    typedef Eigen::Vector3d pos_2d;
    typedef Eigen::Vector3d pos_3d;
    // Chi squared test table.
    static std::map<int, double> chi_squared_test_table;
    typedef std::map<StateIDType,pos_3d, std::less<int>,Eigen::aligned_allocator<std::pair<const StateIDType, pos_3d> > > Mapping; 
    // State vector
    void publish_cam_points(Mapping map_mapped);
    StateServer state_server;
    // Maximum number of camera states
    int max_cam_state_size;
    float calc_dist(PointType p1, PointType p2);
    void initialize_tof(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void FeatureExtract(const sensor_msgs::PointCloud2::ConstPtr &cloud_msg,pcl::PointCloud<pcl::PointXYZ>::Ptr output);
    void stateAugmentationLidar(const double& time);
    void LidarUpdate(pcl::PointCloud<pcl::PointXYZ>::ConstPtr output);//是否需要锁定output？
    // Features used
    void LidarMeasurementUpdate(const MatrixXd& H, const VectorXd& r); 
    MapServer map_server;
    
    // IMU data buffer
    // This is buffer is used to handle the unsynchronization or
    // transfer delay between IMU and Image messages.
    std::vector<sensor_msgs::Imu> imu_msg_buffer;
    KD_TREE<PointType> ikdtree;//添加一个新的树的结构
    // Indicate if the gravity vector is set.
    bool is_gravity_set;

    // Indicate if the received image is the first one. The
    // system will start after receiving the first image.
    bool is_first_img;
    bool first_tof;
    // The position uncertainty threshold is used to determine
    // when to reset the system online. Otherwise, the ever-
    // increaseing uncertainty will make the estimation unstable.
    // Note this online reset will be some dead-reckoning.
    // Set this threshold to nonpositive to disable online reset.
    double position_std_threshold;

    // Tracking rate
    double tracking_rate;

    // Threshold for determine keyframes
    double translation_threshold;
    double rotation_threshold;
    double tracking_rate_threshold;
    
    double filter_size_map_min;
    
    // Ros node handle
    ros::NodeHandle nh;

    // Subscribers and publishers
    ros::Subscriber imu_sub;
    ros::Subscriber feature_sub;
    ros::Subscriber tof_sub;

    ros::Publisher pubEdge;
    ros::Publisher map_pub;
    ros::Publisher odom_pub;
    ros::Publisher feature_pub;
    tf::TransformBroadcaster tf_pub;
    ros::ServiceServer reset_srv;

    // Frame id
    std::string fixed_frame_id;
    std::string child_frame_id;

    // Whether to publish tf or not.
    bool publish_tf;

    // Framte rate of the stereo images. This variable is
    // only used to determine the timing threshold of
    // each iteration of the filter.
    double frame_rate;

    // Debugging variables and functions
    void mocapOdomCallback(
        const nav_msgs::OdometryConstPtr& msg);

    ros::Subscriber mocap_odom_sub;
    ros::Publisher mocap_odom_pub;
    geometry_msgs::TransformStamped raw_mocap_odom_msg;
    Eigen::Isometry3d mocap_initial_frame;
};

typedef MsckfVio::Ptr MsckfVioPtr;
typedef MsckfVio::ConstPtr MsckfVioConstPtr;

} // namespace msckf_vio

#endif
