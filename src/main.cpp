#include "args/args.hxx"
#include "imgui.h"

#include "io.h"
#include "polyscope_misc.h"
#include "solve.h"


// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {


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

  Eigen::VectorXi is_boundary_point;
  Eigen::VectorXi is_cage_point;
  Eigen::MatrixXd bb;
  std::tie(is_boundary_point, is_cage_point, bb) = appendBoundaryAndCage(P, N);

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
  
  std::vector<Eigen::Vector3d> oc_pts;
  std::vector<Eigen::Vector2i> oc_edges;
  std::tie(oc_pts, oc_edges) = visOctree(CN_l, W_l);

  std::cout << "Built octree vis." << std::endl;
  
  std::vector<struct CellNeighbors> neighs;
  Eigen::VectorXi is_boundary_cell;
  Eigen::VectorXi is_cage_cell;
  std::tie(neighs, is_boundary_cell, is_cage_cell) = createLeafNeighbors(PI_l, CN_l, W_l, is_cage_point, oc_pts, oc_edges, bb);

  std::cout << "Built leaf connectivity" << std::endl;

  std::cout << "Octree statistics:" << std::endl
            << "\tNumber of cells: " << CN.rows() << std::endl
            << "\tNumber of leaf cells: " << CN_l.rows() << std::endl;

  // auto pc_ = polyscope::registerPointCloud("Points", P);
  // pc_->addVectorQuantity("True Normals", N);
  // pc_->setEnabled(true);

  // auto pc_cn_ = polyscope::registerPointCloud("Octree cell centers (leaves)", CN_l);
  // pc_cn_->addScalarQuantity("Depth", depths_l);
  // pc_cn_->addScalarQuantity("Parents", parents_l);
  // pc_cn_->addScalarQuantity("Boundary cells", is_boundary_cell);
  // pc_cn_->addScalarQuantity("cage cells", is_cage_cell);

  // auto pc_oc_ = polyscope::registerCurveNetwork("Octree edges", oc_pts, oc_edges);
  // pc_oc_->setEnabled(false);

  // polyscope::show();

  // solve Dirichlet problem, as a test

  Eigen::VectorXd bdry_vals = Eigen::VectorXd::Zero(CN_l.rows());

  Eigen::Vector3d dir = {1.,1.,1.};
  dir.normalize();

  for (int leaf = 0; leaf < CN_l.rows(); leaf++) {
    bdry_vals[leaf] = dir.dot(CN_l.row(leaf));
    // bdry_vals[leaf] = CN_l(leaf, 0); // + CN_l(leaf, 1) + CN_l(leaf, 2);
  }

  std::cout << "Solving Dirichlet problem" << std::endl;

  Eigen::VectorXd u_uni = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 0);
  std::cout << "\tUniform done" << std::endl;
  Eigen::VectorXd u_weight = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 2);
  std::cout << "\tNeighs = 2 done" << std::endl;
  Eigen::VectorXd u_weight_4 = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 4);
  std::cout << "\tNeighs = 4 done" << std::endl;

  std::cout << "Solved Dirichlet problem" << std::endl;

  // Visualize!

  auto pc = polyscope::registerPointCloud("Points", P);
  pc->addVectorQuantity("True Normals", N);
  pc->setEnabled(false);
  auto pc_cn = polyscope::registerPointCloud("Octree cell centers (leaves)", CN_l);
  pc_cn->addScalarQuantity("Depth", depths_l);
  pc_cn->addScalarQuantity("Parents", parents_l);

  auto pc_oc = polyscope::registerCurveNetwork("Octree edges", oc_pts, oc_edges);
  pc_oc->setEnabled(false);

  pc_cn->addScalarQuantity("Boundary cells", is_boundary_cell);

  pc_cn->addScalarQuantity("Laplacian solve (uniform)", u_uni);
  Eigen::VectorXd laplacian_error_uniform = (u_uni - bdry_vals).cwiseAbs();
  auto vs_le_u = pc_cn->addScalarQuantity("Laplacian error (uniform)", laplacian_error_uniform);

  pc_cn->addScalarQuantity("Laplacian solve (weighted)", u_weight);
  Eigen::VectorXd laplacian_error_weighted = (u_weight - bdry_vals).cwiseAbs();
  auto vs_le_w = pc_cn->addScalarQuantity("Laplacian error (weighted)", laplacian_error_weighted);

  pc_cn->addScalarQuantity("Laplacian solve (weighted, 4 neighbors)", u_weight_4);
  Eigen::VectorXd laplacian_error_weighted_4 = (u_weight_4 - bdry_vals).cwiseAbs();
  auto vs_le_w_4 = pc_cn->addScalarQuantity("Laplacian error (weighted, 4 neighbors)", laplacian_error_weighted_4);
  
  vs_le_u->setColorMap("blues");
  vs_le_u->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));
  vs_le_w->setColorMap("blues");
  vs_le_w->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));
  vs_le_w_4->setColorMap("blues");
  vs_le_w_4->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));

  generate_world_axes({3.,0.,0.});

  // Give control to the polyscope gui
  polyscope::show();

  return EXIT_SUCCESS;
}
