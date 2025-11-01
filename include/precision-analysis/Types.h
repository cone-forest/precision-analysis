#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/EulerAngles>
#include <vector>
#include <string>

namespace PrecisionAnalysis {
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


std::vector<Point> ParseData(const std::string& data);

}