#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/curve_network.h"

#include "args/args.hxx"
#include "imgui.h"

#include "io.h"
#include "oct_helper.h"

#include "../deps/libigl/include/igl/octree.h"
#include "../deps/libigl/include/igl/bounding_box.h"

#include <Eigen/Dense>

#include <queue>
#include <unordered_map>

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {


}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector2i>> visOctree(Eigen::MatrixXd CN, Eigen::VectorXd W) {
  std::vector<Eigen::Vector3d> points;
  std::vector<Eigen::Vector2i> edges;
  size_t idx_base = 0;

  for (int leaf = 0; leaf < CN.rows(); leaf++){
    Eigen::Vector3d center = CN.row(leaf);
    double width = W.coeffRef(leaf) / 2.;

    // generate 8 points
    
    points.push_back({center[0] - width, center[1] - width, center[2] - width}); // 0
    points.push_back({center[0] - width, center[1] - width, center[2] + width}); // 1
    points.push_back({center[0] - width, center[1] + width, center[2] - width}); // 2
    points.push_back({center[0] - width, center[1] + width, center[2] + width}); // 3
    points.push_back({center[0] + width, center[1] - width, center[2] - width}); // 4
    points.push_back({center[0] + width, center[1] - width, center[2] + width}); // 5
    points.push_back({center[0] + width, center[1] + width, center[2] - width}); // 6
    points.push_back({center[0] + width, center[1] + width, center[2] + width}); // 7

    // generate 12 edges
    edges.push_back({idx_base + 0, idx_base + 1});
    edges.push_back({idx_base + 0, idx_base + 2});
    edges.push_back({idx_base + 0, idx_base + 4});
    edges.push_back({idx_base + 1, idx_base + 3});
    edges.push_back({idx_base + 1, idx_base + 5});
    edges.push_back({idx_base + 2, idx_base + 3});
    edges.push_back({idx_base + 2, idx_base + 6});
    edges.push_back({idx_base + 3, idx_base + 7});
    edges.push_back({idx_base + 4, idx_base + 5});
    edges.push_back({idx_base + 4, idx_base + 6});
    edges.push_back({idx_base + 5, idx_base + 7});
    edges.push_back({idx_base + 6, idx_base + 7});

    idx_base += 8;
  }

  return std::make_tuple(points, edges);
}

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
            Eigen::VectorXd &W) {
  
  int nCells = CN.rows();

  Eigen::VectorXi parents(nCells);
  parents[0] = -1;
  Eigen::VectorXi depths = Eigen::VectorXi::Ones(nCells) * -1;
  depths[0] = 0; // root is 1

  // assign a depth to each cell 
  
  std::queue<int> bfs;
  bfs.push(0);
  int leaf_count = 0;
    
  for (; !bfs.empty(); bfs.pop()) {
    int curr = bfs.front();
    int curr_depth = depths[curr];

    Eigen::VectorXi children = CH.row(curr);

    if (children[0] == - 1) {
      // leaf;
      leaf_count++;
      continue;
    }

    for (auto child: children) {
      parents[child] = curr;
      depths[child] = curr_depth + 1;
      bfs.push(child);
    }
  }

  std::cout << "Assigned depths and parents" << std::endl;

  // set aside and reindex leaf cells
  std::vector<std::vector<int>> PI_l(leaf_count);
  Eigen::MatrixXd CN_l(leaf_count, 3);
  Eigen::VectorXd W_l(leaf_count);
  std::unordered_map<int, int> leaf_to_all;
  std::unordered_map<int, int> all_to_leaf;
  Eigen::VectorXi depths_l(leaf_count);
  Eigen::VectorXi parents_l(leaf_count);

  int leaf_idx = 0;
  for (int idx = 0; idx < nCells; idx++) {
    if (CH(idx, 0) == -1) {
      // idx is a leaf node

      leaf_to_all[leaf_idx] = idx;
      all_to_leaf[idx] = leaf_idx;

      for (int child: PI[idx]) {
        PI_l[leaf_idx].push_back(child);
      } 

      CN_l.row(leaf_idx) = CN.row(idx);
      W_l[leaf_idx] = W(idx);
      depths_l[leaf_idx] = depths[idx];
      parents_l[leaf_idx] = parents[idx];

      leaf_idx++;
    }
  }
  
  return std::make_tuple(PI_l, CN_l, W_l, leaf_to_all, all_to_leaf, depths, depths_l, parents, parents_l);
}

void appendBBPts(Eigen::MatrixXd &P, Eigen::MatrixXd &N) {
  Eigen::MatrixXd BV;
  Eigen::MatrixXi BF;
  
  igl::bounding_box(P, BV, BF);

  double PADDING = 0.5;

  Eigen::Vector3d bb_max = BV.row(0);
  Eigen::Vector3d bb_min = BV.row(7);

  Eigen::Vector3d delta = bb_max - bb_min * PADDING;
  delta = delta.cwiseAbs();

  P.conservativeResize(P.rows() + 8, P.cols());
  P.row(P.rows() - 8) << bb_min[0] - delta[0], bb_min[1] - delta[1], bb_min[2] - delta[2];
  P.row(P.rows() - 7) << bb_min[0] - delta[0], bb_min[1] - delta[1], bb_max[2] + delta[2];
  P.row(P.rows() - 6) << bb_min[0] - delta[0], bb_max[1] + delta[1], bb_min[2] - delta[2];
  P.row(P.rows() - 5) << bb_min[0] - delta[0], bb_max[1] + delta[1], bb_max[2] + delta[2];
  P.row(P.rows() - 4) << bb_max[0] + delta[0], bb_min[1] - delta[1], bb_min[2] - delta[2];
  P.row(P.rows() - 3) << bb_max[0] + delta[0], bb_min[1] - delta[1], bb_max[2] + delta[2];
  P.row(P.rows() - 2) << bb_max[0] + delta[0], bb_max[1] + delta[1], bb_min[2] - delta[2];
  P.row(P.rows() - 1) << bb_max[0] + delta[0], bb_max[1] + delta[1], bb_max[2] + delta[2];

  // add dummy normals for corners
  N.conservativeResize(N.rows() + 8, N.cols());
  N.row(N.rows() - 8) << 0., 0., 0.;
  N.row(N.rows() - 7) << 0., 0., 0.;
  N.row(N.rows() - 6) << 0., 0., 0.;
  N.row(N.rows() - 5) << 0., 0., 0.;
  N.row(N.rows() - 4) << 0., 0., 0.;
  N.row(N.rows() - 3) << 0., 0., 0.;
  N.row(N.rows() - 2) << 0., 0., 0.;
  N.row(N.rows() - 1) << 0., 0., 0.;
  
}

int main(int argc, char **argv) {

  // Configure the argument parser
  args::ArgumentParser parser("3D Faraday cage test project");
  args::Positional<std::string> inputFilename(parser, "pc", "A point cloud");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &h) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  // Make sure a mesh name was given
  if (!inputFilename) {
    std::cerr << "Please specify a mesh file as argument" << std::endl;
    return EXIT_FAILURE;
  }

  std::string filename = args::get(inputFilename);
  std::cout << filename << std::endl;

  Eigen::MatrixXd P;
  Eigen::MatrixXd N;

  // load point cloud
  std::tie(P, N) = parsePLY(filename);

  std::cout << "Loaded file" << std::endl;

  appendBBPts(P, N);

  // Initialize polyscope
  polyscope::init();

  // Set the callback function
  polyscope::state::userCallback = myCallback;
  
  // octree
  std::vector<std::vector<int >> PI; // associates points to octree cells
  Eigen::MatrixXi CH; // cells * 8, lists children
  Eigen::MatrixXd CN; // cell centers
  Eigen::VectorXd W; //  cell widths

  igl::octree(P, PI, CH, CN, W);

  std::cout << "Built octree" << std::endl;

  std::vector<std::vector<int >> PI_l;
  // no Eigen::MatrixXi CH_l because it's not needed & annoying to compute
  Eigen::MatrixXd CN_l;
  Eigen::VectorXd W_l;
  std::unordered_map<int, int> leaf_to_all;
  std::unordered_map<int, int> all_to_leaf;
  Eigen::VectorXi depths;
  Eigen::VectorXi depths_l;
  Eigen::VectorXi parents;
  Eigen::VectorXi parents_l;
  
  std::tie(PI_l, CN_l, W_l, leaf_to_all, all_to_leaf, depths, depths_l, parents, parents_l) = getLeaves(P, PI, CH, CN, W);

  std::cout << "Extracted leaves" << std::endl;

  std::vector<struct CellNeighbors> neighs = createOctreeNeighbors(CN_l, W_l);

  std::cout << "Built leaf connectivity" << std::endl;

  std::cout << "Octree statistics:" << std::endl
            << "\tNumber of cells: " << CN.rows() << std::endl
            << "\tNumber of leaf cells: " << CN_l.rows() << std::endl;

  // build octree visualization
  std::vector<Eigen::Vector3d> oc_pts;
  std::vector<Eigen::Vector2i> oc_edges;
  std::tie(oc_pts, oc_edges) = visOctree(CN_l, W_l);
  auto pc_oc = polyscope::registerCurveNetwork("Octree edges", oc_pts, oc_edges);


  auto pc = polyscope::registerPointCloud("Points", P);
  pc->addVectorQuantity("True Normals", N);
  auto pc_cn = polyscope::registerPointCloud("Octree cell centers (leaves)", CN_l);
  pc_cn->addScalarQuantity("Depth", depths_l);
  pc_cn->addScalarQuantity("Parents", parents_l);

  // test octree neighbors
  int test_cell = 20;
  Eigen::VectorXi right_neighs = Eigen::VectorXi::Zero(CN_l.rows());
  right_neighs[test_cell] = 2;
  Eigen::VectorXi left_neighs = Eigen::VectorXi::Zero(CN_l.rows());
  left_neighs[test_cell] = 2;
  Eigen::VectorXi top_neighs = Eigen::VectorXi::Zero(CN_l.rows());
  top_neighs[test_cell] = 2;
  Eigen::VectorXi bottom_neighs = Eigen::VectorXi::Zero(CN_l.rows());
  bottom_neighs[test_cell] = 2;
  Eigen::VectorXi front_neighs = Eigen::VectorXi::Zero(CN_l.rows());
  front_neighs[test_cell] = 2;
  Eigen::VectorXi back_neighs = Eigen::VectorXi::Zero(CN_l.rows());
  back_neighs[test_cell] = 2;
  for (int n: neighs[test_cell].right) {
    right_neighs[n] = 1;
  }
  for (int n: neighs[test_cell].left) {
    left_neighs[n] = 1;
  }
  for (int n: neighs[test_cell].top) {
    top_neighs[n] = 1;
  }
  for (int n: neighs[test_cell].bottom) {
    bottom_neighs[n] = 1;
  }
  for (int n: neighs[test_cell].front) {
    front_neighs[n] = 1;
  }
  for (int n: neighs[test_cell].back) {
    back_neighs[n] = 1;
  }

  pc_cn->addScalarQuantity("right", right_neighs);
  pc_cn->addScalarQuantity("left", left_neighs);
  pc_cn->addScalarQuantity("top", top_neighs);
  pc_cn->addScalarQuantity("bottom", bottom_neighs);
  pc_cn->addScalarQuantity("front", front_neighs);
  pc_cn->addScalarQuantity("back", back_neighs);

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
