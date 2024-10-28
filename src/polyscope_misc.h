#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

void generate_world_axes(Eigen::Vector3d pos);

void polyscope_plot_vectors(std::string name, Eigen::Vector3d pos, std::vector<Eigen::Vector3d> dirs, std::vector<Eigen::Vector3d> colors);