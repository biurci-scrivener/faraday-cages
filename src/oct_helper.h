#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <numeric>
#include <algorithm>

struct CellNeighbors {
    std::vector<int> right;
    std::vector<int> left;
    std::vector<int> top;
    std::vector<int> bottom;
    std::vector<int> front;
    std::vector<int> back;
    std::vector<int> all;
};

Eigen::VectorXd solveDirichletProblem(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &depths, Eigen::VectorXd bdry_vals, int laplacian);

Eigen::VectorXi splitBoundaryInteriorCells(std::vector<struct CellNeighbors> neighs);

std::vector<struct CellNeighbors> createLeafNeighbors(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges);


