#include <BiomechanicalAnalysis/IK/InverseKinematics.h>
#include <BiomechanicalAnalysis/Logging/Logger.h>
#include <BipedalLocomotion/Conversions/ManifConversions.h>
#include <iDynTree/EigenHelpers.h>

using namespace BiomechanicalAnalysis::IK;
using namespace BipedalLocomotion::ContinuousDynamicalSystem;
using namespace BipedalLocomotion::Conversions;
using namespace std::chrono_literals;

bool HumanIK::initialize(
    std::weak_ptr<const BipedalLocomotion::ParametersHandler::IParametersHandler> handler,
    std::shared_ptr<iDynTree::KinDynComputations> kinDyn)
{
    constexpr std::size_t highPriority = 0;
    constexpr std::size_t lowPriority = 1;
    constexpr auto logPrefix = "[HumanIK::initialize]";

    if ((kinDyn == nullptr) || (!kinDyn->isValid()))
    {
        BiomechanicalAnalysis::log()->error("{} Invalid kinDyn object.", logPrefix);
        return false;
    }

    m_kinDyn = kinDyn;

    m_jointPositions.resize(m_kinDyn->getNrOfDegreesOfFreedom());
    m_jointVelocities.resize(m_kinDyn->getNrOfDegreesOfFreedom());

    kinDyn->getRobotState(m_basePose,
                          m_jointPositions,
                          m_baseVelocity,
                          m_jointVelocities,
                          m_gravity);

    m_system.dynamics = std::make_shared<FloatingBaseSystemKinematics>();
    m_system.dynamics->setState({m_basePose.topRightCorner<3, 1>(),
                                 toManifRot(m_basePose.topLeftCorner<3, 3>()),
                                 m_jointPositions});

    m_system.integrator = std::make_shared<ForwardEuler<FloatingBaseSystemKinematics>>();
    m_system.integrator->setDynamicalSystem(m_system.dynamics);

    Eigen::Vector3d Weight;
    Weight.setConstant(10.0);

    m_nrDoFs = kinDyn->getNrOfDegreesOfFreedom();

    auto ptr = handler.lock();
    if (ptr == nullptr)
    {
        BiomechanicalAnalysis::log()->error("{} Invalid parameters handler.", logPrefix);
        return false;
    }

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

    for (const auto& task : tasks)
    {
        auto taskHandler = ptr->getGroup(task).lock();
        if (taskHandler == nullptr)
        {
            BiomechanicalAnalysis::log()->error("{} Group {} is missing in the configuration file",
                                                logPrefix,
                                                task);
            return false;
        }
        std::string taskType;
        if (!taskHandler->getParameter("type", taskType))
        {
            BiomechanicalAnalysis::log()->error("{} Parameter task_type of the {} task is missing",
                                                logPrefix,
                                                task);
            return false;
        }
        if (taskType == "SO3Task")
        {
            int nodeNumber;
            if (!taskHandler->getParameter("node_number", nodeNumber))
            {
                BiomechanicalAnalysis::log()->error("{} Parameter node_number of the {} task is "
                                                    "missing",
                                                    logPrefix,
                                                    task);
                return false;
            }
            m_OrientationTasks[nodeNumber].nodeNumber = nodeNumber;
            m_OrientationTasks[nodeNumber].task
                = std::make_shared<BipedalLocomotion::IK::SO3Task>();
            std::vector<double> rotation_matrix;
            if (taskHandler->getParameter("rotation_matrix", rotation_matrix))
            {
                m_OrientationTasks[nodeNumber].IMU_R_link
                    = BipedalLocomotion::Conversions::toManifRot(
                        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
                            rotation_matrix.data()));
            } else
            {
                std::string frame_name;
                taskHandler->getParameter("frame_name", frame_name);
                BiomechanicalAnalysis::log()->warn("{} Parameter rotation_matrix of the {} task is "
                                                   "missing, setting the rotation matrix from the "
                                                   "IMU to the frame {} to identity",
                                                   logPrefix,
                                                   task,
                                                   frame_name);
                m_OrientationTasks[nodeNumber].IMU_R_link.setIdentity();
            }
            ok = ok && m_OrientationTasks[nodeNumber].task->setKinDyn(kinDyn);
            ok = ok && m_OrientationTasks[nodeNumber].task->initialize(taskHandler);
            ok = ok
                 && m_qpIK.addTask(m_OrientationTasks[nodeNumber].task, task, lowPriority, Weight);
            if (!ok)
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task",
                                                    logPrefix,
                                                    task);
                return false;
            }
        } else if (taskType == "GravityTask")
        {
            if (!initializeGravityTask(task, taskHandler))
            {
                BiomechanicalAnalysis::log()->error("{} Error in the initialization of the {} task",
                                                    logPrefix,
                                                    task);
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
    m_dtIntegration
        = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(dt));

    return m_system.integrator->setIntegrationStep(m_dtIntegration);
}

double HumanIK::getDt() const
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(m_dtIntegration).count();
}

int HumanIK::getDoFsNumber() const
{
    return m_nrDoFs;
}

bool HumanIK::setNodeSetPoint(const int node,
                              const manif::SO3d& I_R_IMU,
                              const manif::SO3Tangentd& I_omega_IMU)
{
    bool ok;
    if (m_OrientationTasks.find(node) == m_OrientationTasks.end())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::setNodeSetPoint] Invalid node number.");
        return false;
    }
    I_R_link = m_OrientationTasks[node].calibrationMatrix * I_R_IMU
               * m_OrientationTasks[node].IMU_R_link;
    ok = m_OrientationTasks[node].task->setSetPoint(I_R_link, I_omega_IMU);
    return ok;
}

bool HumanIK::TPoseCalibrationNode(const int node, const manif::SO3d& I_R_IMU)
{
    if (m_OrientationTasks.find(node) == m_OrientationTasks.end())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::setNodeSetPoint] Invalid node number.");
        return false;
    }
    m_OrientationTasks[node].calibrationMatrix
        = calib_R_link * (I_R_IMU * m_OrientationTasks[node].IMU_R_link).inverse();

    return true;
}

bool HumanIK::advance()
{
    bool ok{true};
    ok = ok && m_qpIK.advance();
    ok = ok && m_qpIK.isOutputValid();

    if (!ok)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::advance] Error in the QP solver.");
        return false;
    }
    m_jointVelocities = m_qpIK.getOutput().jointVelocity;
    m_baseVelocity = m_qpIK.getOutput().baseVelocity.coeffs();

    ok = ok && m_system.dynamics->setControlInput({m_baseVelocity, m_jointVelocities});
    ok = ok && m_system.integrator->integrate(0s, m_dtIntegration);

    if (!ok)
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::advance] Error in the integration.");
        return false;
    }
    const auto& [basePosition, baseRotation, jointPosition] = m_system.integrator->getSolution();
    m_basePose.topRightCorner<3, 1>() = basePosition;
    m_basePose.topLeftCorner<3, 3>() = baseRotation.rotation();
    m_jointPositions = jointPosition;
    m_kinDyn->setRobotState(m_basePose, jointPosition, m_baseVelocity, m_jointVelocities, m_gravity);

    return ok;
}

bool HumanIK::getJointPositions(Eigen::Ref<Eigen::VectorXd> jointPositions) const
{
    if (jointPositions.size() != m_jointPositions.size())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::getJointPositions] Invalid size of the "
                                            "input vector.");
        return false;
    }
    jointPositions = m_jointPositions;

    return true;
}

bool HumanIK::getJointVelocities(Eigen::Ref<Eigen::VectorXd> jointVelocities) const
{
    if (jointVelocities.size() != m_jointVelocities.size())
    {
        BiomechanicalAnalysis::log()->error("[HumanIK::getJointVelocities] Invalid size of the "
                                            "input vector.");
        return false;
    }
    jointVelocities = m_jointVelocities;

    return true;
}

bool HumanIK::getBasePosition(Eigen::Ref<Eigen::Vector3d> basePosition) const
{
    basePosition = m_basePose.topRightCorner<3, 1>();

    return true;
}

bool HumanIK::getBaseLinearVelocity(Eigen::Ref<Eigen::Vector3d> baseVelocity) const
{
    baseVelocity = m_baseVelocity.topRows<3>();

    return true;
}

bool HumanIK::getBaseOrientation(Eigen::Ref<Eigen::Matrix3d> baseOrientation) const
{
    baseOrientation = m_basePose.topLeftCorner<3, 3>();

    return true;
}

bool HumanIK::getBaseAngularVelocity(Eigen::Ref<Eigen::Vector3d> baseAngularVelocity) const
{
    baseAngularVelocity = m_baseVelocity.bottomRows<3>();

    return true;
}

bool HumanIK::initializeGravityTask(
    const std::string& taskName,
    const std::shared_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> taskHandler)
{
    constexpr auto logPrefix = "[HumanIK::initialize]";
    int nodeNumber;
    bool ok{true};
    if (!taskHandler->getParameter("node_number", nodeNumber))
    {
        BiomechanicalAnalysis::log()->error("{} Parameter node_number of the {} task is "
                                            "missing",
                                            logPrefix,
                                            taskName);
        return false;
    }
    m_GravityTasks[nodeNumber].nodeNumber = nodeNumber;
    m_GravityTasks[nodeNumber].task = std::make_shared<BipedalLocomotion::IK::GravityTask>();
    m_GravityTasks[nodeNumber].weightProvider
        = std::make_shared<BipedalLocomotion::ContinuousDynamicalSystem::MultiStateWeightProvider>();
    if (!m_GravityTasks[nodeNumber].weightProvider->initialize(taskHandler))
    {
        BiomechanicalAnalysis::log()->error("{} Error in the initialization of the "
                                            "MultiStateWeightProvider of {} task",
                                            logPrefix,
                                            taskName);
        return false;
    }
    ok = ok && m_GravityTasks[nodeNumber].task->setKinDyn(m_kinDyn);
    ok = ok && m_GravityTasks[nodeNumber].task->initialize(taskHandler);
    ok = ok
         && m_qpIK.addTask(m_GravityTasks[nodeNumber].task,
                           taskName,
                           1,
                           m_GravityTasks[nodeNumber].weightProvider);
    return ok;
}
