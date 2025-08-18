#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>


#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>
#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <trac_ik/trac_ik.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <memory>

using namespace KDL;
using namespace std::chrono_literals;

class ImprovedTrackIKNode : public rclcpp::Node {
private:
    // Core components
    std::string urdf_string_;
    std::unique_ptr<TRAC_IK::TRAC_IK> tracik_solver_;
    std::unique_ptr<ChainFkSolverPos_recursive> fk_solver_;
    



    
    // Publishers and subscribers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_point_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_point_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Transform broadcasting
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster_;

    // Robot model
    KDL::Chain kdl_chain_;
    std::vector<std::string> joint_names_;
    std::vector<std::pair<double, double>> joint_limits_;
    
    // State variables
    JntArray current_joint_positions_;
    JntArray target_joint_positions_;
    JntArray joint_velocities_;
    Vector current_target_position_;
    
    // Control parameters
    bool first_solution_received_;
    bool target_reached_;
    double position_tolerance_;
    double joint_velocity_limit_;
    double convergence_threshold_;
    std::vector<double> joint_positions_;
    
    // Workspace analysis
    std::vector<Vector> workspace_points_;
    std::vector<Vector> test_targets_;
    size_t current_target_index_;
    
    // Performance tracking
    int successful_solutions_;
    int failed_solutions_;
    
    urdf::Model robot_model_;
    double average_solve_time_ = 0.0;


    
public:
    ImprovedTrackIKNode() : Node("improved_trackik_node"), 
                           tf_broadcaster_(this),
                           static_tf_broadcaster_(this),
                           first_solution_received_(false),
                           target_reached_(true),
                           position_tolerance_(0.01),    
                           joint_velocity_limit_(1.0),   
                           convergence_threshold_(0.01),
                           current_target_index_(0),
                           successful_solutions_(0),
                           failed_solutions_(0),
                           average_solve_time_(0.0) {

        joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_markers", 10);
        target_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("target_point", 10);
        
        target_point_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
            "target_point_cmd", 10, 
            std::bind(&ImprovedTrackIKNode::targetPointCallback, this, std::placeholders::_1));
            
        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "target_pose_cmd", 10, 
            std::bind(&ImprovedTrackIKNode::targetPoseCallback, this, std::placeholders::_1));

        joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states",    // topic ka naam
            10,                 // queue size
            std::bind(&ImprovedTrackIKNode::jointStateCallback, this, std::placeholders::_1)
);
   

        if (!loadRobotModel()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load robot model");
            return;
        }
        robot_model_.initString(urdf_string_);

        if (!initializeSolvers()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize solvers");
            return;
        }

        robot_model_.initString(urdf_string_);


        initializeTestTargets();
        
        generateWorkspaceSamples();
        
        testSolverCapabilities();

        timer_ = this->create_wall_timer(100ms, std::bind(&ImprovedTrackIKNode::timerCallback, this));

        RCLCPP_INFO(this->get_logger(), "Improved Track-IK Node initialized successfully (Position-only mode)");
    }

private:
    bool loadRobotModel() {
        std::string urdf_path = "/home/vansh/intern-ardee/src/fcl-ros2/ajgar_description/urdf/ik_arm.urdf";
        
        std::ifstream urdf_file(urdf_path);
        if (!urdf_file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open URDF file: %s", urdf_path.c_str());
            return false;
        }
        
        std::stringstream buffer;
        buffer << urdf_file.rdbuf();
        urdf_string_ = buffer.str();
        urdf_file.close();
        
        // urdf::Model robot_model;
        if (!robot_model_.initString(urdf_string_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse URDF");
            return false;
        }
        
        KDL::Tree kdl_tree;
        if (!kdl_parser::treeFromUrdfModel(robot_model_, kdl_tree)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to convert URDF to KDL Tree");
            return false;
        }
        
        std::string base_link = "base_link";
        std::string tip_link = "end__1";
        
        if (!kdl_tree.getChain(base_link, tip_link, kdl_chain_)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to extract chain from %s to %s", 
                        base_link.c_str(), tip_link.c_str());
            return false;
        }
        
        joint_names_.clear();
        for (unsigned int i = 0; i < kdl_chain_.getNrOfSegments(); ++i) {
            KDL::Segment segment = kdl_chain_.getSegment(i);
            if (segment.getJoint().getType() != KDL::Joint::None) {
                joint_names_.push_back(segment.getJoint().getName());
            }
        }
        
        if (joint_names_.empty()) {
            joint_names_ = {
            "base_joint",
            "shoulder_joint",
            "arm_joint",
            "forearm_joint",
            "end_joint",
            "suction_joint"
            };

        }
        
        RCLCPP_INFO(this->get_logger(), "Successfully loaded robot model with %d joints", 
                   kdl_chain_.getNrOfJoints());
        
        return true;
    }
    
    bool initializeSolvers() {
        
        fk_solver_ = std::make_unique<ChainFkSolverPos_recursive>(kdl_chain_);
        
        std::string base_link = "base_link";
        std::string tip_link = "end__1";
        
        double timeout = 0.1;   // Increased timeout to 10ms
        double eps = 5e-3;       
        
        RCLCPP_INFO(this->get_logger(), "Initializing Track-IK with timeout=%.3f, eps=%.6f", timeout, eps);
        
        tracik_solver_ = std::make_unique<TRAC_IK::TRAC_IK>(
            base_link, tip_link, urdf_string_, timeout, eps, TRAC_IK::Distance);
        
        KDL::Chain trac_ik_chain;
        if (!tracik_solver_->getKDLChain(trac_ik_chain)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get KDL chain from Track-IK");
            return false;
        }

        KDL::Chain trac_chain;
        tracik_solver_->getKDLChain(trac_chain);
        RCLCPP_INFO(this->get_logger(), "TRAC-IK Joint Order:");
        for (unsigned int i = 0; i < trac_chain.getNrOfSegments(); ++i) {
            if (trac_chain.getSegment(i).getJoint().getType() != KDL::Joint::None)
                RCLCPP_INFO(this->get_logger(), "  %s", trac_chain.getSegment(i).getJoint().getName().c_str());
        }

    
        
        unsigned int nj = kdl_chain_.getNrOfJoints();
        current_joint_positions_ = JntArray(nj);
        target_joint_positions_ = JntArray(nj);
        joint_velocities_ = JntArray(nj);
        
        joint_limits_.clear();
      
        for (const auto& name : joint_names_) {
            auto joint = robot_model_.getJoint(name);
            if (!joint) {
                RCLCPP_WARN(this->get_logger(), "Joint '%s' not found in URDF!", name.c_str());
            }
            else if (joint->type != urdf::Joint::CONTINUOUS && joint->limits) {
                if (joint->limits->upper == joint->limits->lower) {
                    RCLCPP_WARN(this->get_logger(), "Joint '%s' has identical upper and lower limits!", name.c_str());
                }
                joint_limits_.emplace_back(joint->limits->lower, joint->limits->upper);
            } else {
                joint_limits_.emplace_back(-M_PI, M_PI);
            }
        }
        
        if (joint_limits_.size() != kdl_chain_.getNrOfJoints()) {
            RCLCPP_ERROR(this->get_logger(), "Joint limits size (%ld) doesn't match number of joints in KDL chain (%d)",
                        joint_limits_.size(), kdl_chain_.getNrOfJoints());
        }


        for (size_t i = 0; i < current_joint_positions_.rows(); ++i) {
            auto [lo, hi] = joint_limits_[i];
            if (current_joint_positions_(i) < lo || current_joint_positions_(i) > hi) {
                RCLCPP_WARN(this->get_logger(), "Initial joint %ld value %.3f is out of limits [%.3f, %.3f]!",
                            i, current_joint_positions_(i), lo, hi);
            }
        }



        for (unsigned int i = 0; i < nj; ++i) {
            if (i < joint_limits_.size()) {
                current_joint_positions_(i) = (joint_limits_[i].first + joint_limits_[i].second) / 2.0;
            } else {
                current_joint_positions_(i) = 0.0;
            }
            joint_velocities_(i) = 0.0;
        }
        target_joint_positions_ = current_joint_positions_;
        
        
        Frame initial_frame;
        if (fk_solver_->JntToCart(current_joint_positions_, initial_frame) >= 0) {
            current_target_position_ = initial_frame.p;
            RCLCPP_INFO(this->get_logger(), "Initial EE position: [%.3f, %.3f, %.3f]", 
                       current_target_position_.x(), current_target_position_.y(), current_target_position_.z());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to compute initial forward kinematics");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "All solvers initialized successfully");
        return true;
    }
    

     void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
        {
            if (msg->position.size() >= 6) {  
                joint_positions_.assign(msg->position.begin(), msg->position.begin() + 6);
            } else {
                RCLCPP_WARN(this->get_logger(), "JointState message me 6 se kam joints hain!");
            }
        }




    void initializeTestTargets() {
        Frame current_frame;
        if (fk_solver_->JntToCart(current_joint_positions_, current_frame) >= 0) {
            Vector base_pos = current_frame.p;
            
            RCLCPP_INFO(this->get_logger(), "Current EE position: [%.3f, %.3f, %.3f]", 
                       base_pos.x(), base_pos.y(), base_pos.z());

                       
            
            // Create conservative test targets around current position
            // test_targets_ = {
            //     base_pos,                                    // Current position
            //     base_pos + Vector(0.0, 0.1, 0.0),         // Small +Y movement
            //     base_pos + Vector(0.0, -0.05, 0.0),        // Small -Y movement
            //     base_pos + Vector(0.0, 0.0, 0.05),         // Small +Z movement
            //     base_pos + Vector(0.0, 0.0, -0.1),
            //           // Small -Z movement
               
            // };
            
            test_targets_ = {
                    Vector(0.4,0.3,0.2),
                    Vector(-0.4,0.3,0.2),
                    
                   
                    
                };

            
            // Test all targets and keep only reachable ones
            std::vector<Vector> reachable_targets;
            for (size_t i = 0; i < test_targets_.size(); ++i) {
                const auto& target = test_targets_[i];
                RCLCPP_INFO(this->get_logger(), "Testing target %zu: [%.3f, %.3f, %.3f]", 
                           i, target.x(), target.y(), target.z());
                
                if (isTargetReachable(target)) {
                    // Do a quick IK test
                    JntArray test_result(kdl_chain_.getNrOfJoints());
                    
                    if (solveIKPositionOnly(target, current_joint_positions_, test_result)) {
                        reachable_targets.push_back(target);
                        RCLCPP_INFO(this->get_logger(), "✓ Target %zu is reachable", i);
                    } else {
                        RCLCPP_WARN(this->get_logger(), "✗ Target %zu failed IK test", i);
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), "✗ Target %zu failed reachability test", i);
                }
            }
            
            test_targets_ = reachable_targets;
            
            RCLCPP_INFO(this->get_logger(), "Initialized %zu reachable test targets", test_targets_.size());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to get current EE position for test target initialization");
        }
    }
    
    bool isTargetReachable(const Vector& target) {
        // More comprehensive reachability check
        double distance = target.Norm();
        // double dist = target_frame.p.Norm();  // Euclidean distance
        RCLCPP_INFO(get_logger(), "Target distance from base: %.3f", distance);

        // Check distance bounds
        if (distance < 0.01) {
            RCLCPP_WARN(this->get_logger(), "Target too close to origin: %.3f m", distance);
            return false;
        }
        
        if (distance > 1.3) {  // Conservative upper bound
            RCLCPP_WARN(this->get_logger(), "Target too far from origin: %.3f m", distance);
            return false;
        }
        
        // Check if target is not too close to base
        if (std::abs(target.z()) < 0.001) {
            RCLCPP_WARN(this->get_logger(), "Target too close to base plane: z=%.3f", target.z());
            return false;
        }
        
        return true;
    }
    
    void generateWorkspaceSamples() {
        RCLCPP_INFO(this->get_logger(), "Generating workspace samples...");

        workspace_points_.clear();
        srand(42);
        int num_samples = 160000;

        for (int i = 0; i < num_samples; ++i) {
            JntArray q(kdl_chain_.getNrOfJoints());

            // Generate random joint angles within limits
            for (unsigned int j = 0; j < q.rows(); ++j) {
                if (j < joint_limits_.size()) {
                    double lower = joint_limits_[j].first;
                    double upper = joint_limits_[j].second;
                    q(j) = lower + ((double)rand() / RAND_MAX) * (upper - lower);
                } else {
                    q(j) = ((double)rand() / RAND_MAX - 0.5) * 2 * M_PI;
                }
            }

            // Forward Kinematics
            Frame result;
            if (fk_solver_->JntToCart(q, result) >= 0) {
                workspace_points_.push_back(result.p);  // Save only after valid FK

                // Print the newly added point
                // const KDL::Vector& p = result.p;
                // RCLCPP_INFO(this->get_logger(), "Point %d: [%.3f, %.3f, %.3f]", 
                //             i, p.x(), p.y(), p.z());
            }
        }

        RCLCPP_INFO(this->get_logger(), "Generated %zu valid workspace points", workspace_points_.size());
    }

    
    void testSolverCapabilities() {
        RCLCPP_INFO(this->get_logger(), "Testing solver capabilities...");
        
        // Test current position
        Frame current_frame;
        if (fk_solver_->JntToCart(current_joint_positions_, current_frame) >= 0) {
            JntArray test_result(kdl_chain_.getNrOfJoints());
            
            auto start_time = std::chrono::high_resolution_clock::now();
            bool success = solveIKPositionOnly(current_frame.p, current_joint_positions_, test_result);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            
            if (success) {
                RCLCPP_INFO(this->get_logger(), "✓ Self-test passed (%.3f ms)", duration.count() / 1000.0);
            } else {
                // RCLCPP_ERROR(this->get_logger(), "✗ Self-test failed");
            }
        }
    }
    
    bool solveIKPositionOnly(const Vector& target_pos, const JntArray& q_init, JntArray& q_result) {
        double distance_from_origin = target_pos.Norm();
        RCLCPP_INFO(this->get_logger(), "Target position: [%.3f, %.3f, %.3f], distance: %.3f", 
                   target_pos.x(), target_pos.y(), target_pos.z(), distance_from_origin);
        
        if (distance_from_origin < 0.01) {
            RCLCPP_ERROR(this->get_logger(), "Target too close to origin: %.3f m", distance_from_origin);
            return false;
        }
        
        if (distance_from_origin > 1.3) {
            RCLCPP_ERROR(this->get_logger(), "Target too far from origin: %.3f m", distance_from_origin);
            return false;
        }
        
        Frame target_frame;
        target_frame.p = target_pos;
        // target_frame.M = Rotation::Identity();  // Don't constrain orientation
        
        q_result = JntArray(kdl_chain_.getNrOfJoints());
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        int result = tracik_solver_->CartToJnt(q_init, target_frame, q_result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (result >= 0) {
            successful_solutions_++;
            average_solve_time_ = (average_solve_time_ * (successful_solutions_ - 1) + 
                                  duration.count() / 1000.0) / successful_solutions_;
            
            Frame verify_frame;
            if (fk_solver_->JntToCart(q_result, verify_frame) >= 0) {
                Vector pos_error = target_pos - verify_frame.p;
                double error_norm = pos_error.Norm();
                
                RCLCPP_INFO(this->get_logger(), "Solution found! Position error: %.6f m", error_norm);
                
                if (error_norm > position_tolerance_) {
                    RCLCPP_WARN(this->get_logger(), "Large position error: %.6f m", error_norm);
                }
            }
            
            return true;
        } else {
            std::cout<<result<<std::endl;
            failed_solutions_++;
            
            // More detailed error reporting
            // RCLCPP_ERROR(this->get_logger(), "Track-IK failed for target [%.3f, %.3f, %.3f]", 
            //             target_pos.x(), target_pos.y(), target_pos.z());
            
            switch (result) {
                case -1:
                    RCLCPP_ERROR(this->get_logger(), "Track-IK failed: Timeout occurred (increase timeout or reduce precision)");
                    break;
                case -2:
                    RCLCPP_ERROR(this->get_logger(), "Track-IK failed: No solution within tolerance (target may be unreachable)");
                    break;
                // case -3:
                //     RCLCPP_ERROR(this->get_logger(), "Track-IK failed: Invalid inputs (check URDF or joint limits)");
                //     break;
                // default:
                //     RCLCPP_ERROR(this->get_logger(), "Track-IK failed: Unknown error code %d", result);
                //     break;
            }
            
            // Try to find current end-effector position for debugging
            Frame current_frame;
            if (fk_solver_->JntToCart(q_init, current_frame) >= 0) {
                RCLCPP_INFO(this->get_logger(), "Current EE position: [%.3f, %.3f, %.3f]", 
                           current_frame.p.x(), current_frame.p.y(), current_frame.p.z());
            }
            
            return false;
        }
    }
    
    // New callback for point-only commands
    void targetPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        current_target_position_ = Vector(msg->point.x, msg->point.y, msg->point.z);
        
        target_reached_ = false;
        RCLCPP_INFO(this->get_logger(), "New target point received: [%.3f, %.3f, %.3f]", 
                   current_target_position_.x(), current_target_position_.y(), current_target_position_.z());
    }
    
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_target_position_ = Vector(msg->pose.position.x, 
                                         msg->pose.position.y, 
                                         msg->pose.position.z);
        
        target_reached_ = false;
        RCLCPP_INFO(this->get_logger(), "New target position received (orientation ignored): [%.3f, %.3f, %.3f]", 
                   current_target_position_.x(), current_target_position_.y(), current_target_position_.z());
    }
    
    void timerCallback() {
       
        bool joints_moving = false;
        KDL::Frame current_pose;
        KDL::Frame target_pose;
    
        std::vector<double> integral_error(current_joint_positions_.rows(), 0.0);
        std::vector<double> prev_error_a(current_joint_positions_.rows(), 0.0);
        double prev_error=0.0;
        
        
    
        double Kp = 0.3489;
        double Ki = 0.0;
        double Kd = 0.00;
        double integral=0.0;
        double dt = 0.1; 
        
        // for (unsigned int i = 0; i < current_joint_positions_.rows(); ++i) {
        for (unsigned int i = 5; i < 6; ++i) {
            double target = target_joint_positions_(i);
            std::cout<<"target"<<target<<std::endl;
            std::cout << "Joint Positions: ";
           double pos;
        if (i < joint_positions_.size()) {
                pos = joint_positions_[i];
                
                std::cout << pos << " "<<std::endl;
            } else {
                std::cerr << "Index out of range: " << i << std::endl;
            }

            double delta=target-pos;
       


            // double current_pos=joint_positions_[i];
            // std::cout<<"joint"<<current_pos<<std::endl;
            bool oscillating = false;      // oscillation flag
            int oscillation_count = 0;     
             
            static std::chrono::steady_clock::time_point last_oscillation_time;
            static bool first_oscillation = true;

            if ((prev_error_a[i] * delta) < 0) {
                
                oscillating = true;
                oscillation_count++;
                auto now = std::chrono::steady_clock::now();

                if (!first_oscillation) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_oscillation_time).count();
                    std::cout << ">>> Oscillation detected at joint " << i 
                            << " | Count = " << oscillation_count 
                            << " | Time since last oscillation = " << duration << " ms" 
                            << std::endl;
                } else {
                    first_oscillation = false;
                }

                last_oscillation_time = now;
            }
          
            double deg_delta = delta * 180.0 / 3.14;
            
            double proportional= Kp * deg_delta;

            
            integral+= Ki* dt * deg_delta;
            double derivative= Kd*( deg_delta- prev_error)/dt;
            prev_error=deg_delta;
            std::cout<<"error=="<<deg_delta<<std::endl;

            
            
            double velocity= proportional+integral+derivative;
            std::cout<<"pid=="<<velocity<<std::endl;
            if (std::abs(delta) > convergence_threshold_) {
                /
                if (velocity>1.5 && std::abs(deg_delta)>5.0){
                    current_joint_positions_(i) += 1.5*dt;
                    joint_velocities_(i) = 1.5;
                    std::cout<<"vel="<<joint_velocities_(i)<<std::endl;
                    joints_moving = true;
                }
                else if(velocity<=-1.5 && std::abs(deg_delta)>5.0){
                    current_joint_positions_(i) += -1.5*dt;
                    joint_velocities_(i) = -1.5;
                    std::cout<<"vel="<<joint_velocities_(i)<<std::endl;
                    joints_moving = true;
                }   
             else if(std::abs(deg_delta)<5.0) {
                    current_joint_positions_(i) += velocity*dt;
                    joint_velocities_(i) = velocity;
                    std::cout<<"vel="<<joint_velocities_(i)<<std::endl;
                    joints_moving = true;
                }
                
                else {
                    current_joint_positions_(i) += velocity*dt;
                    joint_velocities_(i) = velocity;
                    std::cout<<"vel="<<joint_velocities_(i)<<std::endl;
                    joints_moving = true;
                }
        }
            else if(delta<=0.01){
                current_joint_positions_(i) = target_joint_positions_(i);
                joint_velocities_(i) = 0.0;
            }
            
            
        }
        publishJointStates();
        
        
        
        if (!joints_moving && !target_reached_) {
            target_reached_ = true;
            RCLCPP_INFO(this->get_logger(), "Target reached!");
        }
        
        if (target_reached_ && !test_targets_.empty()) {
            static auto last_target_switch = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_target_switch).count() > 3) {
                current_target_index_ = (current_target_index_ + 1) % test_targets_.size();
                current_target_position_ = test_targets_[current_target_index_];
                
                JntArray q_result(kdl_chain_.getNrOfJoints());
                if (solveIKPositionOnly(current_target_position_, current_joint_positions_, q_result)) {
                    target_joint_positions_ = q_result;
                    target_reached_ = false;
                    RCLCPP_INFO(this->get_logger(), "Switching to target %zu", current_target_index_);
                } else {
                    RCLCPP_WARN(this->get_logger(), "Failed to solve IK for target %zu", current_target_index_);
                }
                
                last_target_switch = now;
            }
        }
        
        publishVisualization();
        
        static int counter = 0;
        if (++counter % 200 == 0) { // Every 10 seconds
            RCLCPP_INFO(this->get_logger(), 
                       "Performance: Success=%d, Failed=%d, Avg solve time=%.3f ms",
                       successful_solutions_, failed_solutions_, average_solve_time_);
        }
    }
    
    void publishJointStates() {
        auto joint_state_msg = sensor_msgs::msg::JointState();
        joint_state_msg.header.stamp = this->get_clock()->now();
        joint_state_msg.header.frame_id = "base_link";
        
        joint_state_msg.name = joint_names_;
        joint_state_msg.position.resize(current_joint_positions_.rows());
        joint_state_msg.velocity.resize(current_joint_positions_.rows());
        joint_state_msg.effort.resize(current_joint_positions_.rows());
        
        for (unsigned int i = 0; i < current_joint_positions_.rows(); ++i) {
            joint_state_msg.position[i] = current_joint_positions_(i);
            joint_state_msg.velocity[i] = joint_velocities_(i);
            joint_state_msg.effort[i] = 0.0;
        }
        
        joint_state_pub_->publish(joint_state_msg);
    }
    
    void publishVisualization() {
        auto marker_array = visualization_msgs::msg::MarkerArray();
        
        Frame current_frame;
        if (fk_solver_->JntToCart(current_joint_positions_, current_frame) >= 0) {
            auto ee_marker = visualization_msgs::msg::Marker();
            ee_marker.header.frame_id = "base_link";
            ee_marker.header.stamp = this->get_clock()->now();
            ee_marker.ns = "end_effector";
            ee_marker.id = 0;
            ee_marker.type = visualization_msgs::msg::Marker::SPHERE;
            ee_marker.action = visualization_msgs::msg::Marker::ADD;
            
            ee_marker.pose.position.x = current_frame.p.x();
            ee_marker.pose.position.y = current_frame.p.y();
            ee_marker.pose.position.z = current_frame.p.z();
            ee_marker.pose.orientation.w = 1.0;
            
            ee_marker.scale.x = 0.02;
            ee_marker.scale.y = 0.02;
            ee_marker.scale.z = 0.02;
            
            ee_marker.color.r = 0.0;
            ee_marker.color.g = 1.0;
            ee_marker.color.b = 0.0;
            ee_marker.color.a = 1.0;
            
            marker_array.markers.push_back(ee_marker);
        }
        
        // Target marker
        auto target_marker = visualization_msgs::msg::Marker();
        target_marker.header.frame_id = "base_link";
        target_marker.header.stamp = this->get_clock()->now();
        target_marker.ns = "target";
        target_marker.id = 0;
        target_marker.type = visualization_msgs::msg::Marker::SPHERE;
        target_marker.action = visualization_msgs::msg::Marker::ADD;
        
        target_marker.pose.position.x = current_target_position_.x();
        target_marker.pose.position.y = current_target_position_.y();
        target_marker.pose.position.z = current_target_position_.z();
        target_marker.pose.orientation.w = 1.0;
        
        target_marker.scale.x = 0.025;
        target_marker.scale.y = 0.025;
        target_marker.scale.z = 0.025;
        
        target_marker.color.r = 1.0;
        target_marker.color.g = 0.0;
        target_marker.color.b = 0.0;
        target_marker.color.a = 0.8;
        
        marker_array.markers.push_back(target_marker);
        
        marker_pub_->publish(marker_array);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<ImprovedTrackIKNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting Improved Track-IK Node (Position-only mode)");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
    }
    
    rclcpp::shutdown();
    return 0;
}