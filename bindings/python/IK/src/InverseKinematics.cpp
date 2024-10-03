/**
 * @file InverseKinematics.cpp
 * @authors Evelyn D'Elia
 * @copyright 2024 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <BiomechanicalAnalysis/IK/InverseKinematics.h>
#include <BiomechanicalAnalysis/bindings/type_caster/swig.h>
#include <BipedalLocomotion/ParametersHandler/IParametersHandler.h>

namespace BiomechanicalAnalysis
{
namespace bindings
{
namespace IK
{

void CreateInverseKinematics(pybind11::module& module)
{
    namespace py = ::pybind11;

    using namespace BiomechanicalAnalysis::IK;
    using namespace BipedalLocomotion::ParametersHandler;

    py::class_<nodeData>(module, "nodeData")
        .def(py::init<>())
        .def_readwrite("I_R_IMU", &nodeData::I_R_IMU)
        .def_readwrite("I_omega_IMU", &nodeData::I_omega_IMU);

    py::class_<HumanIK>(module, "HumanIK")
        .def(py::init())
        .def(
            "initialize",
            [](HumanIK& ik, std::shared_ptr<const IParametersHandler> handler, py::object& obj) -> bool {
                std::shared_ptr<iDynTree::KinDynComputations>* cls
                    = py::detail::swig_wrapped_pointer_to_pybind<std::shared_ptr<iDynTree::KinDynComputations>>(obj);

                if (cls == nullptr)
                {
                    throw ::pybind11::value_error("Invalid input for the function. Please provide "
                                                  "an iDynTree::KinDynComputations object.");
                }

                return ik.initialize(handler, *cls);
            },
            py::arg("param_handler"),
            py::arg("kin_dyn"))
        .def("setDt", &HumanIK::setDt, py::arg("dt"))
        .def("getDt", &HumanIK::getDt)
        .def("getDoFsNumber", &HumanIK::getDoFsNumber)
        .def("updateOrientationTask",
             &HumanIK::updateOrientationTask,
             py::arg("node"),
             py::arg("I_R_IMU"),
             py::arg("I_omega_IMU") = manif::SO3d::Tangent::Zero())
        .def("updateGravityTask", &HumanIK::updateGravityTask, py::arg("node"), py::arg("I_R_IMU"))
        .def("updateFloorContactTask", &HumanIK::updateFloorContactTask, py::arg("node"), py::arg("verticalForce"))
        .def("clearCalibrationMatrices", &HumanIK::clearCalibrationMatrices)
        .def("calibrateWorldYaw", &HumanIK::calibrateWorldYaw, py::arg("nodeStruct"))
        .def("calibrateAllWithWorld", &HumanIK::calibrateAllWithWorld, py::arg("nodeStruct"), py::arg("frameName"))
        .def("calibrateWorldYaw", &HumanIK::calibrateWorldYaw, py::arg("nodeStruct"))
        .def("calibrateAllWithWorld", &HumanIK::calibrateAllWithWorld, py::arg("nodeStruct"), py::arg("frameName"))
        .def("advance", &HumanIK::advance)
        .def("updateOrientationGravityTasks", &HumanIK::updateOrientationAndGravityTasks, py::arg("nodeStruct"))
        .def("updateFloorContactTasks", &HumanIK::updateFloorContactTasks, py::arg("wrenchMap"))
        .def("updateJointRegularizationTask", &HumanIK::updateJointRegularizationTask)
        .def("updateJointConstraintsTask", &HumanIK::updateJointConstraintsTask)
        .def("advance", &HumanIK::advance)
        .def("getJointPositions",
             [](HumanIK& ik) {
                 Eigen::VectorXd jointPositions(ik.getDoFsNumber());
                 bool ok = ik.getJointPositions(jointPositions);
                 return std::make_tuple(ok, jointPositions);
             })
        .def("getJointVelocities", [](HumanIK& ik) {
            Eigen::VectorXd jointVelocities(ik.getDoFsNumber());
            bool ok = ik.getJointVelocities(jointVelocities);
            return std::make_tuple(ok, jointVelocities);
        });
}

} // namespace IK
} // namespace bindings
} // namespace BiomechanicalAnalysis
