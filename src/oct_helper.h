#include <Eigen/Dense>
#include <iostream>

struct CellNeighbors {
    std::vector<int> right;
    std::vector<int> left;
    std::vector<int> top;
    std::vector<int> bottom;
    std::vector<int> front;
    std::vector<int> back;
};

// Eigen::VectorXi onSide(int leaf, int other, Eigen::MatrixXd &CN_l, Eigen::VectorXd &W_l);

std::vector<struct CellNeighbors> createOctreeNeighbors(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges);

