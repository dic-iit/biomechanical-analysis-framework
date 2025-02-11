#include <BiomechanicalAnalysis/IK/InverseKinematics.h>
#include <BiomechanicalAnalysis/Logging/Logger.h>
#include <BipedalLocomotion/Conversions/ManifConversions.h>
#include <iDynTree/EigenHelpers.h>
#include <iDynTree/Model.h>

using namespace BiomechanicalAnalysis::IK;
using namespace BipedalLocomotion::ContinuousDynamicalSystem;
using namespace BipedalLocomotion::Conversions;
using namespace std::chrono_literals;

constexpr size_t WRENCH_FORCE_X = 0;
constexpr size_t WRENCH_FORCE_Y = 1;
constexpr size_t WRENCH_FORCE_Z = 2;
constexpr size_t WRENCH_TORQUE_X = 3;
constexpr size_t WRENCH_TORQUE_Y = 4;
constexpr size_t WRENCH_TORQUE_Z = 5;

bool HumanIK::initialize(std::weak_ptr<const BipedalLocomotion::ParametersHandler::IParametersHandler> handler,
                         std::shared_ptr<iDynTree::KinDynComputations> kinDyn)
{
    // set priorities variables
    constexpr std::size_t highPriority = 0;
    constexpr std::size_t lowPriority = 1;
    constexpr auto logPrefix = "[HumanIK::initialize]";

    // Check the validity of the kinDyn object
    if ((kinDyn == nullptr) || (!kinDyn->isValid()))
    {
        BiomechanicalAnalysis::log()->error("{} Invalid kinDyn object.", logPrefix);
        return false;
    }

    // Initialize kinDyn variable
    m_kinDyn = kinDyn;

    // resize kinematic variables based on DoF
    m_jointPositions.resize(m_kinDyn->getNrOfDegreesOfFreedom());
    m_jointVelocities.resize(m_kinDyn->getNrOfDegreesOfFreedom());
    m_calibrationJointPositions.resize(m_kinDyn->getNrOfDegreesOfFreedom());

    // Retrieve the state of the system
    if (!kinDyn->getRobotState(m_basePose, m_jointPositions, m_baseVelocity, m_jointVelocities, m_gravity))
    {
        BiomechanicalAnalysis::log()->error("{} Failed to get the human state.", logPrefix);
        return false;
    }

    m_system.dynamics = std::make_shared<FloatingBaseSystemKinematics>();
    m_system.dynamics->setState({m_basePose.topRightCorner<3, 1>(), toManifRot(m_basePose.topLeftCorner<3, 3>()), m_jointPositions});

    m_system.integrator = std::make_shared<ForwardEuler<FloatingBaseSystemKinematics>>();
    m_system.integrator->setDynamicalSystem(m_system.dynamics);

    // Variable for number of DoF of the model
    m_nrDoFs = kinDyn->getNrOfDegreesOfFreedom();

    auto ptr = handler.lock();
    if (ptr == nullptr)
    {
        BiomechanicalAnalysis::log()->error("{} Invalid parameters handler.", logPrefix);
        return false;
    }

    // Initialize a variable for storing the list of tasks defined in config file
    std::vector<std::string> tasks;
    if (!ptr->getParameter("tasks", tasks))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter tasks is missing", logPrefix);
        return false;
    }

    bool ok = m_qpIK.initialize(ptr->getGroup("IK"));
    auto group = ptr->getGroup("IK").lock();
    std::string variable;
    group->getParameter("robot_velocity_variable_name", variable);
    m_variableHandler.addVariable(variable, kinDyn->getNrOfDegreesOfFreedom() + 6);

    // Retrieve calibration joint positions parameter from config file, using the param handler
    Eigen::VectorXd calibrationJointPositions;
    if (ptr->getParameter("calibration_joint_positions", calibrationJointPositions))
    {
        // Check that the length of the joint positions is correct
        if (calibrationJointPositions.size() != m_kinDyn->getNrOfDegreesOfFreedom())
        {
            BiomechanicalAnalysis::log()->warn("{} Calibration joint positions vector has wrong size, setting all joints to zero",
                                               logPrefix);
            m_calibrationJointPositions.setZero();
        } else
        {
            m_calibrationJointPositions = calibrationJointPositions;
        }
    } else
    {
        // If calibration joint positions are missing, set to 0
        BiomechanicalAnalysis::log()->warn("{} Parameter calibration_joint_positions is missing, setting all joints to zero", logPrefix);
        m_calibrationJointPositions.setZero();
    }

    // Cycle on the tasks to be initialized
    for (const auto& task : tasks)
    {
        // Initialize a pointer to the task parameters
        auto taskHandler = ptr->getGroup(task).lock();
        // Check validity of the pointer
        if (taskHandler == nullptr)
        {
            BiomechanicalAnalysis::log()->error("{} Group {} is missing in the configuration file", logPrefix, task);
            return false;
        }
        // Retrieve the task type from the parameter set
        std::string taskType;

        // Check if operation successful
        if (!taskHandler->getParameter("type", taskType))
        {
            BiomechanicalAnalysis::log()->error("{} Parameter task_type of the {} task is missing", logPrefix, task);
            return false;
        }
        // Initialize SO3 task
        if (taskType == "SO3Task")
        {
            if (!initializeOrientationTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, task);
                return false;
            }
            // Initialize GravityTask
        } else if (taskType == "GravityTask")
        {
            if (!initializeGravityTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, task);
                return false;
            }
            // Initialize FloorContactTask
        } else if (taskType == "FloorContactTask")
        {
            if (!initializeFloorContactTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, task);
                return false;
            }
            // Initialize JointRegularizationTask
        } else if (taskType == "JointRegularizationTask")
        {
            if (!initializeJointRegularizationTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, task);
                return false;
            }
            // Initialize JointConstraintTask
        } else if (taskType == "JointConstraintTask")
        {
            if (!initializeJointConstraintsTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, task);
                return false;
            }
        } else if (taskType == "JointVelocityLimitsTask")
        {
            if (!initializeJointVelocityLimitsTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, task);
                return false;
            }
        } else
        {
            BiomechanicalAnalysis::log()->error("{} Invalid task type {}", logPrefix, taskType);
            return false;
        }
    }

    m_qpIK.finalize(m_variableHandler);

    return ok;
}

bool HumanIK::setDt(const double dt)
{
    // Convert time step from seconds to nanoseconds
    m_dtIntegration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

    // Set integration step for the dynamical system
    return m_system.integrator->setIntegrationStep(m_dtIntegration);
}

double HumanIK::getDt() const
{
    // Convert integration time step from nanoseconds to seconds and return
    return std::chrono::duration_cast<std::chrono::duration<double>>(m_dtIntegration).count();
}

int HumanIK::getDoFsNumber() const
{
    // Return the number of Degrees of Freedom
    return m_nrDoFs;
}

bool HumanIK::updateOrientationTask(const int node, const manif::SO3d& I_R_IMU, const manif::SO3Tangentd& I_omega_IMU)
{
    // Check if the node number is valid
    if (m_OrientationTasks.find(node) == m_OrientationTasks.end())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::updateOrientationTask] Invalid node number {}.", node);
        return false;
    }

    // Compute the rotation matrix from the world to the link frame as:
    // W_R_link = W_R_WIMU * WIMU_R_IMU * IMU_R_link
    I_R_link = m_OrientationTasks[node].calibrationMatrix * I_R_IMU * m_OrientationTasks[node].IMU_R_link;

    // Set the setpoint for the orientation task of the node
    return m_OrientationTasks[node].task->setSetPoint(I_R_link,
                                                      m_OrientationTasks[node].calibrationMatrix.rotation() * I_omega_IMU.coeffs());
}

bool HumanIK::updateGravityTask(const int node, const manif::SO3d& I_R_IMU)
{
    // check if the node number is valid
    if (m_GravityTasks.find(node) == m_GravityTasks.end())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::setNodeSetPoint] Invalid node number {}.", node);
        return false;
    }

    // compute the rotation matrix from the world to the link frame as:
    // W_R_link = W_R_WIMU * WIMU_R_IMU * IMU_R_link
    I_R_link = m_GravityTasks[node].calibrationMatrix * I_R_IMU * m_GravityTasks[node].IMU_R_link;

    // set the set point of the gravity task choosing the z direction of the link_R_W rotation
    // matrix
    return m_GravityTasks[node].task->setSetPoint((I_R_link.rotation().transpose().rightCols(1)));
}

bool HumanIK::updateFloorContactTask(const int node, const double verticalForce, const double linkHeight)
{
    bool ok{true};
    // check if the node number is valid
    if (m_FloorContactTasks.find(node) == m_FloorContactTasks.end())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::updateFloorContactTask] Invalid node number.");
        return false;
    }
    if (m_tPose == true)
    {
        m_FloorContactTasks[node].footInContact = false;
        Eigen::VectorXd jointPositions;
        jointPositions.resize(this->getDoFsNumber());
        jointPositions.setZero();
        Eigen::Matrix4d basePose;
        basePose.setIdentity();
        Eigen::VectorXd baseVelocity;
        baseVelocity.resize(6);
        baseVelocity.setZero();
        m_kinDyn->setRobotState(basePose, jointPositions, baseVelocity, m_jointVelocities, m_gravity);
    }

    // if the vertical force is greater than the threshold and if the foot is not yet in contact,
    // set the weight of the associated task to the weight of the task and set the set point of the
    // task to the position of the frame computed with the legged odometry
    if (verticalForce > m_FloorContactTasks[node].verticalForceThreshold && !m_FloorContactTasks[node].footInContact)
    {
        m_qpIK.setTaskWeight(m_FloorContactTasks[node].taskName, m_FloorContactTasks[node].weight);
        m_FloorContactTasks[node].footInContact = true;
        m_FloorContactTasks[node].setPointPosition
            = iDynTree::toEigen(m_kinDyn->getWorldTransform(m_FloorContactTasks[node].frameName).getPosition());
        m_FloorContactTasks[node].setPointPosition(2) = linkHeight;
    } else if (verticalForce < m_FloorContactTasks[node].verticalForceThreshold && m_FloorContactTasks[node].footInContact)
    {
        // if the foot is not more in contact, set the weight of the associated task to zero
        m_qpIK.setTaskWeight(m_FloorContactTasks[node].taskName, Eigen::Vector3d::Zero());
        m_FloorContactTasks[node].footInContact = false;
    }

    // if the foot is in contact, set the set point of the task
    if (m_FloorContactTasks[node].footInContact)
    {
        ok = m_FloorContactTasks[node].task->setSetPoint(m_FloorContactTasks[node].setPointPosition);
    }

    return ok;
}

bool HumanIK::updateJointRegularizationTask()
{
    // check if the joint regularization task is initialized
    if (m_jointRegularizationTask == nullptr)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::updateJointRegularizationTask] Joint regularization task not initialized.");
        return false;
    }
    // Set the set point of the joint regularization task to a zero vector of size m_nrDoFs
    return m_jointRegularizationTask->setSetPoint(Eigen::VectorXd::Zero(m_nrDoFs));
}

bool HumanIK::updateJointConstraintsTask()
{
    // check if the joint constraints task is initialized
    if (m_jointConstraintsTask == nullptr)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::updateJointConstraintsTask] Joint constraints task not initialized.");
        return false;
    }
    // Update the joint constraints task
    return m_jointConstraintsTask->update();
}

bool HumanIK::updateOrientationAndGravityTasks(const std::unordered_map<int, nodeData>& nodeStruct)
{
    // Update the orientation and gravity tasks
    for (const auto& [node, data] : nodeStruct)
    {
        if (m_OrientationTasks.find(node) != m_OrientationTasks.end())
        {
            if (!updateOrientationTask(node, data.I_R_IMU, data.I_omega_IMU))
            {
                BiomechanicalAnalysis::log()->error("[HumanIK::updateOrientationGravityTasks] "
                                                    "Error in updating the orientation task of "
                                                    "node {}",
                                                    node);
                return false;
            }
        } else if (m_GravityTasks.find(node) != m_GravityTasks.end())
        {
            if (!updateGravityTask(node, data.I_R_IMU))
            {
                BiomechanicalAnalysis::log()->error("[HumanIK::updateOrientationGravityTasks] "
                                                    "Error in updating the gravity task of node {}",
                                                    node);
                return false;
            }
        } else
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::updateOrientationGravityTasks] "
                                                "Invalid node number {}.",
                                                node);
            return false;
        }
    }
    return true;
}

bool HumanIK::updateFloorContactTasks(const std::unordered_map<int, Eigen::Matrix<double, 6, 1>>& wrenchMap, const double linkHeight)
{
    for (const auto& [node, data] : wrenchMap)
    {
        if (!updateFloorContactTask(node, data(WRENCH_FORCE_Z), linkHeight))
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::updateFloorContactTasks] Error in updating "
                                                "the floor contact task of node {}",
                                                node);
            return false;
        }
    }
    return true;
}

bool HumanIK::clearCalibrationMatrices()
{
    for (auto& [node, data] : m_OrientationTasks)
    {
        data.calibrationMatrix = manif::SO3d::Identity();
        data.IMU_R_link = m_OrientationTasks[node].IMU_R_link_init;
    }
    for (auto& [node, data] : m_GravityTasks)
    {
        data.calibrationMatrix = manif::SO3d::Identity();
        data.IMU_R_link = m_GravityTasks[node].IMU_R_link_init;
    }
    return true;
}

bool HumanIK::calibrateWorldYaw(std::unordered_map<int, nodeData> nodeStruct)
{
    // reset the robot state
    Eigen::VectorXd jointVelocities;
    jointVelocities.resize(this->getDoFsNumber());
    jointVelocities.setZero();
    Eigen::Matrix4d basePose;
    basePose.setIdentity();
    Eigen::VectorXd baseVelocity;
    baseVelocity.resize(6);
    baseVelocity.setZero();
    m_kinDyn->setRobotState(basePose, m_calibrationJointPositions, baseVelocity, jointVelocities, m_gravity);
    // Update the orientation and gravity tasks
    for (const auto& [node, data] : nodeStruct)
    {
        // check if the node number is valid
        if (m_OrientationTasks.find(node) == m_OrientationTasks.end() && m_GravityTasks.find(node) == m_GravityTasks.end())
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::calibrateWorldYaw] Invalid node number.");
            return false;
        }

        iDynTree::Rotation rpyOffset;
        if (m_OrientationTasks.find(node) != m_OrientationTasks.end())
        {
            // compute the offset between the world and the IMU world as:
            // W_R_WIMU = W_R_link * (WIMU_R_IMU * IMU_R_link)^{T}
            iDynTree::toEigen(rpyOffset) = iDynTree::toEigen(m_kinDyn->getWorldTransform(m_OrientationTasks[node].frameName).getRotation())
                                           * ((data.I_R_IMU * m_OrientationTasks[node].IMU_R_link).inverse()).rotation();
            // set the calibration matrix of the orientation task to the offset in yaw
            m_OrientationTasks[node].calibrationMatrix = manif::SO3d(0, 0, rpyOffset.asRPY()(2));
        } else
        {
            // compute the offset between the world and the IMU world as:
            // W_R_WIMU = W_R_link * (WIMU_R_IMU * IMU_R_link)^{T}
            iDynTree::toEigen(rpyOffset) = iDynTree::toEigen(m_kinDyn->getWorldTransform(m_GravityTasks[node].frameName).getRotation())
                                           * ((data.I_R_IMU * m_GravityTasks[node].IMU_R_link).inverse()).rotation();
            // set the calibration matrix of the orientation task to the offset in yaw
            m_GravityTasks[node].calibrationMatrix = manif::SO3d(0, 0, rpyOffset.asRPY()(2));
        }
    }
    return true;
}

bool HumanIK::calibrateAllWithWorld(std::unordered_map<int, nodeData> nodeStruct, std::string refFrame)
{
    // reset the robot state
    Eigen::VectorXd jointVelocities;
    jointVelocities.resize(this->getDoFsNumber());
    jointVelocities.setZero();
    Eigen::Matrix4d basePose;
    basePose.setIdentity();
    Eigen::VectorXd baseVelocity;
    baseVelocity.resize(6);
    baseVelocity.setZero();
    m_kinDyn->setRobotState(basePose, m_calibrationJointPositions, baseVelocity, jointVelocities, m_gravity);

    manif::SO3d secondaryCalib = manif::SO3d::Identity();
    // if a reference frame is provided, compute the world rotation matrix of the reference frame
    if (refFrame != "")
    {
        iDynTree::Rotation refFrameRot = m_kinDyn->getWorldTransform(refFrame).getRotation().inverse();
        secondaryCalib = manif::SO3d(refFrameRot.asRPY()(0), refFrameRot.asRPY()(1), refFrameRot.asRPY()(2));
    }

    for (const auto& [node, data] : nodeStruct)
    {
        // check if the node number is valid
        if (m_OrientationTasks.find(node) == m_OrientationTasks.end() && m_GravityTasks.find(node) == m_GravityTasks.end())
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::calibrateWorldYaw] Invalid node number.");
            return false;
        }

        if (m_OrientationTasks.find(node) != m_OrientationTasks.end())
        {
            // compute the rotation matrix from the IMU to the link frame as:
            // IMU_R_link = (W_R_WIMU * WIMU_R_IMU)^{T} * W_R_link
            Eigen::Matrix3d IMU_R_link = (m_OrientationTasks[node].calibrationMatrix * data.I_R_IMU).rotation().transpose()
                                         * iDynTree::toEigen(m_kinDyn->getWorldTransform(m_OrientationTasks[node].frameName).getRotation());
            m_OrientationTasks[node].IMU_R_link = BipedalLocomotion::Conversions::toManifRot(IMU_R_link);
            m_OrientationTasks[node].calibrationMatrix = secondaryCalib * m_OrientationTasks[node].calibrationMatrix;
        } else
        {
            // compute the rotation matrix from the IMU to the link frame as:
            // IMU_R_link = (W_R_WIMU * WIMU_R_IMU)^{T} * W_R_link
            Eigen::Matrix3d IMU_R_link = (m_GravityTasks[node].calibrationMatrix * data.I_R_IMU).rotation().transpose()
                                         * iDynTree::toEigen(m_kinDyn->getWorldTransform(m_GravityTasks[node].frameName).getRotation());
            m_GravityTasks[node].IMU_R_link = BipedalLocomotion::Conversions::toManifRot(IMU_R_link);
            m_GravityTasks[node].calibrationMatrix = secondaryCalib * m_GravityTasks[node].calibrationMatrix;
        }
    }
    // set the flag to true to reset the integration
    m_tPose = true;

    return true;
}

bool HumanIK::advance()
{
    // Initialize ok flag to true
    bool ok{true};

    // Advance the QP solver
    ok = ok && m_qpIK.advance();
    // Check if the output of the QP solver is valid
    ok = ok && m_qpIK.isOutputValid();

    // If there's an error in the QP solver, log an error and return false
    if (!ok)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::advance] Error in the QP solver.");
        return false;
    }

    // Get joint velocities and base velocities from the QP solver output
    m_jointVelocities = m_qpIK.getOutput().jointVelocity;
    m_baseVelocity = m_qpIK.getOutput().baseVelocity.coeffs();

    // Set control input to the system dynamics
    ok = ok && m_system.dynamics->setControlInput({m_baseVelocity, m_jointVelocities});
    // Integrate the system dynamics
    ok = ok && m_system.integrator->integrate(0s, m_dtIntegration);

    // If there's an error in the integration, log an error and return false
    if (!ok)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::advance] Error in the integration.");
        return false;
    }

    if (m_tPose)
    {
        Eigen::Matrix4d basePose; // Pose of the base
        Eigen::VectorXd initialJointPositions; // Initial positions of the joints
        basePose.setIdentity(); // Set the base pose to the identity matrix
        m_system.dynamics->setState(
            {basePose.topRightCorner<3, 1>(), toManifRot(basePose.topLeftCorner<3, 3>()), m_calibrationJointPositions});
        m_tPose = false;
    }

    // Get the solution (base position, base rotation, joint positions) from the integrator
    const auto& [basePosition, baseRotation, jointPosition] = m_system.integrator->getSolution();
    // Update the base pose and joint positions
    m_basePose.topRightCorner<3, 1>() = basePosition;
    m_basePose.topLeftCorner<3, 3>() = baseRotation.rotation();
    m_jointPositions = jointPosition;

    // Set the robot state to the KinDynComputations object
    m_kinDyn->setRobotState(m_basePose, jointPosition, m_baseVelocity, m_jointVelocities, m_gravity);
    // Return whether the process was successful
    return ok;
}

bool HumanIK::getJointPositions(Eigen::Ref<Eigen::VectorXd> jointPositions) const
{
    // Check if the size of the input vector matches the size of the joint positions vector
    if (jointPositions.size() != m_jointPositions.size())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::getJointPositions] Invalid size of the "
                                            "input vector.");
        return false;
    }

    // Assign joint positions to the input vector
    jointPositions = m_jointPositions;

    return true;
}

bool HumanIK::getJointVelocities(Eigen::Ref<Eigen::VectorXd> jointVelocities) const
{
    // Check if the size of the input vector matches the size of the joint velocities vector
    if (jointVelocities.size() != m_jointVelocities.size())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::getJointVelocities] Invalid size of the "
                                            "input vector.");
        return false;
    }

    // Assign joint velocities to the input vector
    jointVelocities = m_jointVelocities;

    return true;
}

bool HumanIK::getBasePosition(Eigen::Ref<Eigen::Vector3d> basePosition) const
{
    // Assign the base position to the input vector
    basePosition = m_basePose.topRightCorner<3, 1>();

    return true;
}

bool HumanIK::getBaseLinearVelocity(Eigen::Ref<Eigen::Vector3d> baseVelocity) const
{
    // Assign the linear velocity of the base to the input vector
    baseVelocity = m_baseVelocity.topRows<3>();

    return true;
}

bool HumanIK::getBaseOrientation(Eigen::Ref<Eigen::Matrix3d> baseOrientation) const
{
    // Assign the orientation of the base to the input matrix
    baseOrientation = m_basePose.topLeftCorner<3, 3>();

    return true;
}

bool HumanIK::getBaseAngularVelocity(Eigen::Ref<Eigen::Vector3d> baseAngularVelocity) const
{
    // Assign the angular velocity of the base to the input vector
    baseAngularVelocity = m_baseVelocity.bottomRows<3>();

    return true;
}

bool HumanIK::initializeOrientationTask(const std::string& taskName,
                                        const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    // Log prefix for error messages
    constexpr auto logPrefix = "[HumanIK::initializeOrientationTask]";

    // Variable to store the node number
    int nodeNumber;
    // Flag to indicate successful initialization
    bool ok{true};

    // Retrieve node number parameter from config file, using the task handler
    if (!taskHandler->getParameter("node_number", nodeNumber))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter node_number of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Retrieve frame name parameter from config file, using the task handler
    if (!taskHandler->getParameter("frame_name", m_OrientationTasks[nodeNumber].frameName))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter frame_name of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Retrieve weight parameter from config file, using the task handler
    std::vector<double> weight;
    if (!taskHandler->getParameter("weight", weight))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter weight of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Check that the weight is a 3D vector
    if (weight.size() != 3)
    {
        BiomechanicalAnalysis::log()->error("{} The size of the parameter weight of the {} task is "
                                            "{}, it should be 3",
                                            logPrefix,
                                            taskName,
                                            weight.size());
        return false;
    }

    // Map weight vector to Eigen::Vector3d and assign it to the corresponding task
    m_OrientationTasks[nodeNumber].weight = Eigen::Map<Eigen::Vector3d>(weight.data());

    // Set node number for the orientation task
    m_OrientationTasks[nodeNumber].nodeNumber = nodeNumber;

    // Create an SO3Task object for the orientation task
    m_OrientationTasks[nodeNumber].task = std::make_shared<BipedalLocomotion::IK::SO3Task>();

    //*****************************************************************************************************
    // Retrieve Rotation matrix  IMU-to-link from configuration file exampleIK.ini
    //*****************************************************************************************************

    std::vector<double> rotation_matrix;
    if (taskHandler->getParameter("rotation_matrix", rotation_matrix))
    {
        // Convert rotation matrix to ManifRot and assign it to IMU_R_link
        m_OrientationTasks[nodeNumber].IMU_R_link_init
            = BipedalLocomotion::Conversions::toManifRot(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation_matrix.data()));
    } else
    {
        // If rotation_matrix parameter is missing, set IMU_R_link to identity
        std::string frame_name;
        taskHandler->getParameter("frame_name", frame_name);
        BiomechanicalAnalysis::log()->warn("{} Parameter rotation_matrix of the {} task is "
                                           "missing, setting the rotation "
                                           "matrix from the IMU to the frame {} to identity",
                                           logPrefix,
                                           taskName,
                                           frame_name);
        m_OrientationTasks[nodeNumber].IMU_R_link_init.setIdentity();
    }
    m_OrientationTasks[nodeNumber].IMU_R_link = m_OrientationTasks[nodeNumber].IMU_R_link_init;
    //*****************************************************************************************************

    // Initialize the SO3Task object
    ok = ok && m_OrientationTasks[nodeNumber].task->setKinDyn(m_kinDyn);
    ok = ok && m_OrientationTasks[nodeNumber].task->initialize(taskHandler);

    // Add the orientation task to the QP solver
    ok = ok && m_qpIK.addTask(m_OrientationTasks[nodeNumber].task, taskName, 1, m_OrientationTasks[nodeNumber].weight);

    // Check if initialization was successful
    if (!ok)
    {
        BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task", logPrefix, taskName);
        return false;
    }

    return ok;
}

bool HumanIK::initializeGravityTask(const std::string& taskName,
                                    const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    // Log prefix for error messages
    constexpr auto logPrefix = "[HumanIK::initializeGravityTask]";

    // Variable to store the node number
    int nodeNumber;
    // Flag to indicate successful initialization
    bool ok{true};

    // Default weight for the gravity task
    Eigen::Vector2d Weight;
    Weight.setConstant(10.0);

    // Retrieve node number parameter from the config file, using the task handler
    if (!taskHandler->getParameter("node_number", nodeNumber))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter node_number of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Retrieve frame name parameter from the config file, using the task handler
    if (!taskHandler->getParameter("target_frame_name", m_GravityTasks[nodeNumber].frameName))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter frame_name of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Retrieve weight parameter from the config file, using the task handler
    std::vector<double> weight;
    if (!taskHandler->getParameter("weight", weight))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter weight of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Check that the weight is a 2D vector
    if (weight.size() != 2)
    {
        BiomechanicalAnalysis::log()->error("{} The size of the parameter weight of the {} task is "
                                            "{}, it should be 2",
                                            logPrefix,
                                            taskName,
                                            weight.size());
        return false;
    }

    //*****************************************************************************************************
    // Retrieve Rotation matrix  IMU-to-link from configuration file exampleIK.ini
    //*****************************************************************************************************

    std::vector<double> rotation_matrix;
    if (taskHandler->getParameter("rotation_matrix", rotation_matrix))
    {
        // Convert rotation matrix to ManifRot and assign it to IMU_R_link
        m_GravityTasks[nodeNumber].IMU_R_link_init
            = BipedalLocomotion::Conversions::toManifRot(Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(rotation_matrix.data()));
    } else
    {
        // If rotation_matrix parameter is missing, set IMU_R_link to identity
        std::string frame_name;
        taskHandler->getParameter("frame_name", frame_name);
        BiomechanicalAnalysis::log()->warn("{} Parameter rotation_matrix of the {} task is "
                                           "missing, setting the rotation "
                                           "matrix from the IMU to the frame {} to identity",
                                           logPrefix,
                                           taskName,
                                           frame_name);
        m_GravityTasks[nodeNumber].IMU_R_link_init.setIdentity();
    }
    m_GravityTasks[nodeNumber].IMU_R_link = m_GravityTasks[nodeNumber].IMU_R_link_init;

    //*****************************************************************************************************

    // Map weight vector to Eigen::Vector2d and assign it to the corresponding task
    m_GravityTasks[nodeNumber].weight = Eigen::Map<Eigen::Vector2d>(weight.data());

    // Set node number and task name for the gravity task
    m_GravityTasks[nodeNumber].nodeNumber = nodeNumber;
    m_GravityTasks[nodeNumber].taskName = taskName;

    // Create an GravityTask object for the gravity task
    m_GravityTasks[nodeNumber].task = std::make_shared<BipedalLocomotion::IK::GravityTask>();

    // Initialize the GravityTask object
    ok = ok && m_GravityTasks[nodeNumber].task->setKinDyn(m_kinDyn);
    ok = ok && m_GravityTasks[nodeNumber].task->initialize(taskHandler);

    // Add the gravity task to the QP solver
    ok = ok && m_qpIK.addTask(m_GravityTasks[nodeNumber].task, taskName, 1, m_GravityTasks[nodeNumber].weight);

    // Check if initialization was successful
    return ok;
}

bool HumanIK::initializeFloorContactTask(const std::string& taskName,
                                         const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    // Log prefix for error messages
    constexpr auto logPrefix = "[HumanIK::initializeFloorContactTask]";

    // Variable to store the node number
    int nodeNumber;
    // Flag to indicate successful initialization
    bool ok{true};

    // Retrieve node number parameter from the task handler
    if (!taskHandler->getParameter("node_number", nodeNumber))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter node_number of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Retrieve frame name parameter from the task handler and assign it to the corresponding
    // FloorContactTask
    if (!taskHandler->getParameter("frame_name", m_FloorContactTasks[nodeNumber].frameName))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter frame_name of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Retrieve weight parameter from the task handler
    std::vector<double> weight;
    if (!taskHandler->getParameter("weight", weight))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter weight of the {} task is missing", logPrefix, taskName);
        return false;
    }

    // Check that the weight is a 3D vector
    if (weight.size() != 3)
    {
        BiomechanicalAnalysis::log()->error("{} The size of the parameter weight of the {} task is "
                                            "{}, it should be 3",
                                            logPrefix,
                                            taskName,
                                            weight.size());
        return false;
    }

    // Retrieve vertical force threshold parameter from the task handler and assign it to the
    // corresponding FloorContactTask
    if (!taskHandler->getParameter("vertical_force_threshold", m_FloorContactTasks[nodeNumber].verticalForceThreshold))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter vertical_force_threshold of the {} task "
                                            "is missing",
                                            logPrefix,
                                            taskName);
        return false;
    }

    // Map weight vector to Eigen::Vector3d and assign it to the corresponding FloorContactTask
    m_FloorContactTasks[nodeNumber].weight = Eigen::Map<Eigen::Vector3d>(weight.data());

    // Set node number and task name for the FloorContactTask
    m_FloorContactTasks[nodeNumber].nodeNumber = nodeNumber;
    m_FloorContactTasks[nodeNumber].taskName = taskName;

    // Create an R3Task object for the floor contact task
    m_FloorContactTasks[nodeNumber].task = std::make_shared<BipedalLocomotion::IK::R3Task>();

    // Initialize the R3Task object
    ok = ok && m_FloorContactTasks[nodeNumber].task->setKinDyn(m_kinDyn);
    ok = ok && m_FloorContactTasks[nodeNumber].task->initialize(taskHandler);

    // Add the floor contact task to the QP solver
    ok = ok && m_qpIK.addTask(m_FloorContactTasks[nodeNumber].task, taskName, 1, m_FloorContactTasks[nodeNumber].weight);

    // Check if initialization was successful
    return ok;
}

bool HumanIK::initializeJointRegularizationTask(const std::string& taskName,
                                                const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    // Flag to indicate successful initialization
    bool ok{true};

    // Initialize a vector of proportional gains (kp) with zeros
    std::vector<double> kp(m_kinDyn->getNrOfDegreesOfFreedom(), 0.0);

    // Set the proportional gains (kp) parameter for the task
    taskHandler->setParameter("kp", kp);

    // Retrieve the weight parameter from the task handler
    double weight;
    if (!taskHandler->getParameter("weight", weight))
    {
        // If weight parameter is missing, log an error and return false
        BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointRegularizationTask] "
                                            "Parameter 'weight' of the {} task is missing",
                                            taskName);
        return false;
    }

    // Create a JointTrackingTask object for joint regularization
    m_jointRegularizationTask = std::make_shared<BipedalLocomotion::IK::JointTrackingTask>();

    // Initialize the JointTrackingTask object
    ok = ok && m_jointRegularizationTask->setKinDyn(m_kinDyn);
    ok = ok && m_jointRegularizationTask->initialize(taskHandler);

    // Create a weight vector with constant values based on the weight parameter
    Eigen::VectorXd weightVector(m_kinDyn->getNrOfDegreesOfFreedom());
    weightVector.setConstant(weight);

    std::vector<std::string> jointsList;
    if (taskHandler->getParameter("joints_list", jointsList))
    {
        std::vector<double> Jointsweights;
        if (!taskHandler->getParameter("joints_weights", Jointsweights))
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointRegularizationTask] "
                                                "Parameter 'joints_weights' of the {} task is missing; this parameter is only ",
                                                "required if 'joints_list' is provided. Please provide the 'joints_weights' parameter",
                                                "or remove the 'joints_list' parameter from the {} task",
                                                taskName);
            return false;
        }
        if (jointsList.size() != Jointsweights.size())
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointRegularizationTask] "
                                                "The size of the parameter 'joints_weights' of the {} "
                                                "task is {}, it should be equal to the size of the "
                                                "parameter 'joints_list' that is {}",
                                                taskName,
                                                Jointsweights.size(),
                                                jointsList.size());
            return false;
        }
        for (int i = 0; i < jointsList.size(); i++)
        {
            auto index = m_kinDyn->model().getJointIndex(jointsList[i]);
            if (!m_kinDyn->model().isValidJointIndex(index))
            {
                BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointRegularizationTask] "
                                                    "Joint {} is not present in the model",
                                                    jointsList[i]);
                return false;
            }

            if (m_kinDyn->model().getJoint(index)->getNrOfDOFs() != 1)
            {
                BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointRegularizationTask] "
                                                    "Joint {} has a number of dofs different from 1",
                                                    jointsList[i]);
                return false;
            }
            auto dofOffset = m_kinDyn->model().getJoint(index)->getDOFsOffset();
            weightVector[dofOffset] = Jointsweights[i];
        }
    }

    // Add the joint regularization task to the QP solver with the specified weight vector
    ok = ok && m_qpIK.addTask(m_jointRegularizationTask, taskName, 1, weightVector);

    // Return true if initialization was successful, otherwise return false
    return ok;
}

bool HumanIK::initializeJointConstraintsTask(const std::string& taskName,
                                             const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    // Flag to indicate successful initialization
    bool ok{true};

    // Variable to store whether to use model limits or custom joint constraints
    bool useModelLimits;

    // Check if the 'use_model_limits' parameter is present in the task handler
    if (!taskHandler->getParameter("use_model_limits", useModelLimits))
    {
        // If 'use_model_limits' parameter is missing, log an error and return false
        BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointConstraintsTask] Parameter "
                                            "'use_model_limits' of the {} task is missing",
                                            taskName);
        return false;
    }

    // Create a JointLimitsTask object for joint constraints
    m_jointConstraintsTask = std::make_shared<BipedalLocomotion::IK::JointLimitsTask>();

    // Set the KinDyn object for the JointLimitsTask
    ok = ok && m_jointConstraintsTask->setKinDyn(m_kinDyn);

    // Variable to store the value of 'k_limits' parameter
    double k_lim;

    // Check if the 'k_limits' parameter is present in the task handler
    if (!taskHandler->getParameter("k_limits", k_lim))
    {
        // If 'k_limits' parameter is missing, log an error and return false
        BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointConstraintsTask] Parameter "
                                            "'k_limits' of the {} task is missing",
                                            taskName);
        return false;
    }

    // Create a vector of k_limits with size equal to the number of degrees of freedom
    std::vector<double> klim(m_kinDyn->getNrOfDegreesOfFreedom(), k_lim);

    // Set the 'klim' parameter for the task
    taskHandler->setParameter("klim", klim);

    // If 'useModelLimits' is true, initialize the JointLimitsTask with model limits
    if (useModelLimits)
    {
        ok = ok && m_jointConstraintsTask->initialize(taskHandler);
    } else
    {
        // Otherwise, initialize the JointLimitsTask with custom joint constraints

        // Retrieve the list of joint names from the task handler
        std::vector<std::string> jointNamesList;
        if (!taskHandler->getParameter("joints_list", jointNamesList))
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointConstraintsTask] "
                                                "Parameter 'joints_list' of the {} task is missing",
                                                taskName);
            return false;
        }

        // Retrieve the lower and upper bounds for the custom joint constraints
        std::vector<double> lowerBounds;
        std::vector<double> upperBounds;
        if (!taskHandler->getParameter("lower_bounds", lowerBounds) || !taskHandler->getParameter("upper_bounds", upperBounds))
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointConstraintsTask] "
                                                "Parameter 'lower_bounds' and/or 'upper_bounds' of "
                                                "the {} "
                                                "task is missing",
                                                taskName);
            return false;
        }

        // Check if the size of the lists matches
        if (jointNamesList.size() != lowerBounds.size() || jointNamesList.size() != upperBounds.size())
        {
            BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointConstraintsTask] "
                                                "The size of the parameter 'lower_bounds' and "
                                                "'upper_bounds' of the {} "
                                                "task are {}, {}, they should be equal to the size "
                                                "of the parameters 'joints_list' that is {}",
                                                taskName,
                                                lowerBounds.size(),
                                                upperBounds.size(),
                                                jointNamesList.size());
            return false;
        }

        // Get the joint indices corresponding to the joint names
        std::vector<int> jointIndices;
        for (const auto& jointName : jointNamesList)
        {
            auto index = m_kinDyn->model().getJointIndex(jointName);
            if (!m_kinDyn->model().isValidJointIndex(index))
            {
                BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointConstraintsTask] "
                                                    "Joint {} is not present in the model",
                                                    jointName);
                return false;
            }
            jointIndices.emplace_back(index);
        }

        // Create vectors to store lower and upper limits
        std::vector<double> lowerLimits(m_kinDyn->getNrOfDegreesOfFreedom());
        std::vector<double> upperLimits(m_kinDyn->getNrOfDegreesOfFreedom());

        // Populate the lower and upper limits with model limits
        for (int i = 0; i < m_kinDyn->getNrOfDegreesOfFreedom(); i++)
        {
            lowerLimits[i] = m_kinDyn->model().getJoint(i)->getMinPosLimit(i);
            upperLimits[i] = m_kinDyn->model().getJoint(i)->getMaxPosLimit(i);
        }

        // Update the lower and upper limits with custom values for specified joints
        for (std::size_t i = 0; i < jointIndices.size(); i++)
        {
            lowerLimits[jointIndices[i]] = lowerBounds[i];
            upperLimits[jointIndices[i]] = upperBounds[i];
        }

        // Set the 'lower_limits' and 'upper_limits' parameters for the task
        taskHandler->setParameter("lower_limits", lowerLimits);
        taskHandler->setParameter("upper_limits", upperLimits);

        // Initialize the JointLimitsTask with custom joint constraints
        ok = ok && m_jointConstraintsTask->initialize(taskHandler);
    }

    // Add the joint constraints task to the QP solver
    ok = ok && m_qpIK.addTask(m_jointConstraintsTask, taskName, 0);

    // Return true if initialization was successful, otherwise return false
    return ok;
}

bool HumanIK::initializeJointVelocityLimitsTask(const std::string& taskName,
                                                const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    // Flag to indicate successful initialization
    bool ok{true};

    // Create a JointVelocityLimitsTask object for joint velocity limits
    m_jointVelocityLimitsTask = std::make_shared<BipedalLocomotion::IK::JointVelocityLimitsTask>();
    double upperLimit;
    double lowerLimit;
    if (!taskHandler->getParameter("upper_limit", upperLimit) || !taskHandler->getParameter("lower_limit", lowerLimit))
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointVelocityLimitsTask] "
                                            "Parameter 'upper_limit' and/or 'lower_limit' of the {} task is missing",
                                            taskName);
        return false;
    }
    if (upperLimit < lowerLimit)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::initializeJointVelocityLimitsTask] "
                                            "The upper limit is less than the lower limit");
        return false;
    }
    std::vector<double> upperLimitsVec(m_kinDyn->getNrOfDegreesOfFreedom(), upperLimit);
    std::vector<double> lowerLimitsVec(m_kinDyn->getNrOfDegreesOfFreedom(), lowerLimit);

    taskHandler->setParameter("upper_limits", upperLimitsVec);
    taskHandler->setParameter("lower_limits", lowerLimitsVec);

    // Set the KinDyn object for the JointVelocityLimitsTask
    ok = ok && m_jointVelocityLimitsTask->setKinDyn(m_kinDyn);

    // Initialize the JointVelocityLimitsTask object
    ok = ok && m_jointVelocityLimitsTask->initialize(taskHandler);

    // Add the joint velocity limits task to the QP solver
    ok = ok && m_qpIK.addTask(m_jointVelocityLimitsTask, taskName, 0);

    // Return true if initialization was successful, otherwise return false
    return ok;
}
