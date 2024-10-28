#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "../deps/libigl/include/igl/octree.h"
#include "../deps/libigl/include/igl/bounding_box.h"

#include "oct_helper.h"

#include <queue>
#include <unordered_map>

Eigen::VectorXd solveFaraday(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &is_cage, Eigen::VectorXi &depths, Eigen::VectorXd bdry_vals);

Eigen::VectorXd solveDirichletProblem(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &depths, Eigen::VectorXd bdry_vals, int laplacian);