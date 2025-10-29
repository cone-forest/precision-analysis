#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/EulerAngles>
#include <vector>
#include <string>

struct Point {
    Eigen::Vector3d m_position;
    Eigen::EulerAngles<double, Eigen::EulerSystemZYX> m_angles;

    Point(Eigen::Vector3d position, Eigen::Vector3d angles)
        : m_position(position) {
            m_angles = Eigen::EulerAngles<double, Eigen::EulerSystemZYX>(angles);
        }
    Point(Eigen::Vector3d position, Eigen::EulerAngles<double, Eigen::EulerSystemZYX> angles)
        : m_position(position), m_angles(angles){}
};

std::vector<Point> parser(std::string data) {
    return {};
}

PYBIND11_MODULE(Point, m) {
    pybind11::class_<Point>(m, "Point")
        .def(pybind11::init<Eigen::Vector3d, Eigen::Vector3d>())
        .def_readwrite("m_position", &Point::m_position)
        .def_readwrite("m_angles", &Point::m_angles);

    m.def("parser", &parser, "parse data (num, position, angels)");
}
