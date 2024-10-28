#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "../deps/libigl/include/igl/octree.h"
#include "../deps/libigl/include/igl/knn.h"
#include "../deps/libigl/include/igl/bounding_box.h"

#include <queue>
#include <unordered_map>
#include <stdexcept>

struct CellNeighbors {
    std::vector<int> right;
    std::vector<int> left;
    std::vector<int> top;
    std::vector<int> bottom;
    std::vector<int> front;
    std::vector<int> back;
    std::vector<int> all;
};

inline const Eigen::MatrixXd ico_pts = (Eigen::MatrixXd(12, 3) << 
            0,      0.52573,      0.85065,
            0,     -0.52573,      0.85065,
            0,      0.52573,     -0.85065,
            0,     -0.52573,     -0.85065,
      0.52573,      0.85065,            0,
     -0.52573,      0.85065,            0,
      0.52573,     -0.85065,            0,
     -0.52573,     -0.85065,            0,
      0.85065,            0,      0.52573,
     -0.85065,            0,      0.52573,
      0.85065,            0,     -0.52573,
     -0.85065,            0,     -0.52573).finished();

double getNeighDepthDelta(int leaf, std::vector<int> &neighs, Eigen::VectorXi &depths);

double getDistanceFromDelta(int leaf, int depth_gap, Eigen::VectorXd &W);

std::tuple<std::vector<int>, Eigen::VectorXd, double> getNClosestNeighs(int leaf, std::vector<int> &neighs, int n, Eigen::MatrixXd &CN);

std::tuple<std::vector<struct CellNeighbors>, Eigen::VectorXi, Eigen::VectorXi> createLeafNeighbors(std::vector<std::vector<int>> &PI, Eigen::MatrixXd &CN, Eigen::VectorXd &W, Eigen::VectorXi &is_cage_point, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges, Eigen::MatrixXd &bb);

std::tuple< std::vector<std::vector<int>>,
            Eigen::MatrixXd, 
            Eigen::VectorXd,
            std::unordered_map<int, int>, 
            std::unordered_map<int, int>,
            Eigen::VectorXi,
            Eigen::VectorXi,
            Eigen::VectorXi,
            Eigen::VectorXi> 
getLeaves(Eigen::MatrixXd &P,
            std::vector<std::vector<int>> &PI, 
            Eigen::MatrixXi &CH, 
            Eigen::MatrixXd &CN, 
            Eigen::VectorXd &W);

std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::MatrixXd> appendBoundaryAndCage(Eigen::MatrixXd &P, Eigen::MatrixXd &N);

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2i>> visOctree(Eigen::MatrixXd CN, Eigen::VectorXd W);

