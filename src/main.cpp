#include "args/args.hxx"
#include "imgui.h"

#include "io.h"
#include "polyscope_misc.h"
#include "solve.h"

static float x_dir = 1.;
static float y_dir = 1.;
static float z_dir = 1.;

Eigen::SparseLU<Eigen::SparseMatrix<double>> faraday_solver;

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
void myCallback() {

	ImGui::InputFloat("##X coord.", &x_dir);
	ImGui::InputFloat("##Y coord.", &y_dir);
	ImGui::InputFloat("##Z coord.", &z_dir);

	if (ImGui::Button("Solve")) {

	// Eigen::VectorXd bdry_vals = Eigen::VectorXd::Zero(CN_l.rows());
	
	// Eigen::Vector3d dir = {x_dir, y_dir, z_dir};
	// dir.normalize();

	// for (int leaf = 0; leaf < CN_l.rows(); leaf++) {
	//   bdry_vals[leaf] = dir.dot(CN_l.row(leaf));
	// }

	// std::cout << "Solving Dirichlet problem" << std::endl;
	// Eigen::VectorXd u_uni = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 0);
	// std::cout << "\tUniform done" << std::endl;
	// Eigen::VectorXd u_weight = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 2);
	// std::cout << "\tNeighs = 2 done" << std::endl;
	// Eigen::VectorXd u_weight_4 = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 4);
	// std::cout << "\tNeighs = 4 done" << std::endl;
	// std::cout << "\tAll done" << std::endl;

	// std::cout << "Decomposing KKT Matrix" << std::endl;
	// Eigen::SparseLU<Eigen::SparseMatrix<double>> faraday_solver;
	// std::unordered_map<int, int> global_to_matrix_ordering = computeFaraday(CN_l, W_l, neighs, is_boundary_cell, is_cage_cell, depths_l, bdry_vals, faraday_solver);
	// std::cout << "\tDone" << std::endl;
	// std::cout << "Solving for one direction " << dir.transpose() << std::endl;
	// Eigen::VectorXd u_faraday = solveFaraday(CN_l, W_l, neighs, is_boundary_cell, is_cage_cell, depths_l, bdry_vals, faraday_solver, global_to_matrix_ordering);
	// std::cout << "\tDone" << std::endl;

	// auto pc_cn = polyscope::getStructure("Octree cell centers (leaves)");

	// pc_cn->addScalarQuantity("Laplacian solve (uniform)", u_uni);
	// Eigen::VectorXd laplacian_error_uniform = (u_uni - bdry_vals).cwiseAbs();
	// auto vs_le_u = pc_cn->addScalarQuantity("Laplacian error (uniform)", laplacian_error_uniform);

	// pc_cn->addScalarQuantity("Laplacian solve (weighted)", u_weight);
	// Eigen::VectorXd laplacian_error_weighted = (u_weight - bdry_vals).cwiseAbs();
	// auto vs_le_w = pc_cn->addScalarQuantity("Laplacian error (weighted, 2 neighbors)", laplacian_error_weighted);

	// pc_cn->addScalarQuantity("Laplacian solve (weighted, 4 neighbors)", u_weight_4);
	// Eigen::VectorXd laplacian_error_weighted_4 = (u_weight_4 - bdry_vals).cwiseAbs();
	// auto vs_le_w_4 = pc_cn->addScalarQuantity("Laplacian error (weighted, 4 neighbors)", laplacian_error_weighted_4);

	// pc_cn->addScalarQuantity("Base field", bdry_vals);
	// auto vs_faraday = pc_cn->addScalarQuantity("Faraday solve", u_faraday);
	// auto vs_faraday_diff = pc_cn->addScalarQuantity("Faraday, field difference", (u_faraday - bdry_vals).cwiseAbs());
	// vs_faraday_diff->setColorMap("reds");
	
	// vs_le_u->setColorMap("blues");
	// vs_le_u->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));
	// vs_le_w->setColorMap("blues");
	// vs_le_w->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));
	// vs_le_w_4->setColorMap("blues");
	// vs_le_w_4->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));


	};
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
	std::vector<std::vector<int>> my_cage_points;
	Eigen::MatrixXd bb;

	std::tie(is_boundary_point, is_cage_point, my_cage_points, bb) = appendBoundaryAndCage(P, N);

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

	Eigen::MatrixXd bb_oct = octreeBB(CN_l); 

	std::cout << "Extracted leaves" << std::endl;
	
	std::vector<Eigen::Vector3d> oc_pts;
	std::vector<Eigen::Vector2i> oc_edges;
	std::tie(oc_pts, oc_edges) = visOctree(CN_l, W_l);

	std::cout << "Built octree vis." << std::endl;

		// auto pc_ = polyscope::registerPointCloud("Points", P);
		// pc_->addVectorQuantity("True Normals", N);
		// pc_->setEnabled(true);

		// auto pc_cn_ = polyscope::registerPointCloud("Octree cell centers (leaves)", CN_l);
		// pc_cn_->addScalarQuantity("Depth", depths_l);
		// pc_cn_->addScalarQuantity("Parents", parents_l);
		// pc_cn_->addScalarQuantity("Boundary cells", is_boundary_cell);
		// pc_cn_->addScalarQuantity("cage cells", is_cage_cell);

		// // test neighborhood connectivity
		// int test_idx_ = 100;
		// Eigen::VectorXi is_test_neigh_ = Eigen::VectorXi::Zero(CN_l.rows());

		// for (int neigh: neighs[test_idx_].all) {
		// 	is_test_neigh_[neigh] = 1.;
		// }

		// pc_cn_->addScalarQuantity("Connectivity (debug, vertex " + std::to_string(test_idx_) + ")", is_test_neigh_);

		// auto pc_oc_ = polyscope::registerCurveNetwork("Octree edges", oc_pts, oc_edges);
		// pc_oc_->setEnabled(false);

		// polyscope::show();
		// exit(0);
	
	std::vector<struct CellNeighbors> neighs;
	Eigen::VectorXi is_boundary_cell;
	Eigen::VectorXi is_cage_cell;
	std::tie(neighs, is_boundary_cell, is_cage_cell) = createLeafNeighbors(PI_l, CN_l, W_l, is_cage_point, oc_pts, oc_edges, bb_oct);

	std::cout << "Built leaf connectivity" << std::endl;

	std::cout << "Octree statistics:" << std::endl
			<< "\tNumber of cells: " << CN.rows() << std::endl
			<< "\tNumber of leaf cells: " << CN_l.rows() << std::endl;

	// solve Dirichlet problem, as a test

	// Eigen::VectorXd bdry_vals = Eigen::VectorXd::Zero(CN_l.rows());

	// Eigen::Vector3d dir = {x_dir, y_dir, z_dir};
	// dir.normalize();

	// for (int leaf = 0; leaf < CN_l.rows(); leaf++) {
	// 	bdry_vals[leaf] = dir.dot(CN_l.row(leaf));
	// }

	Eigen::VectorXd bdry_vals = Eigen::VectorXd::Zero(CN_l.rows());
	Eigen::VectorXd dir_1 = ico_pts_2.row(0);
	for (size_t leaf = 0; leaf < CN_l.rows(); leaf++) bdry_vals[leaf] = dir_1.dot(CN_l.row(leaf));

	std::cout << "Solving Dirichlet problem" << std::endl;
	Eigen::VectorXd u_uni = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 0);
	std::cout << "\tUniform done" << std::endl;
	// Eigen::VectorXd u_weight = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 2);
	// std::cout << "\tNeighs = 2 done" << std::endl;
	Eigen::VectorXd u_weight_4 = solveDirichletProblem(CN_l, W_l, neighs, is_boundary_cell, depths_l, bdry_vals, 4);
	std::cout << "\tNeighs = 4 done" << std::endl;
	std::cout << "\tAll done" << std::endl;

	// Faraday
	std::cout << "Decomposing KKT Matrix" << std::endl;
	std::unordered_map<int, int> global_to_matrix_ordering = computeFaraday(CN_l, W_l, neighs, is_boundary_cell, is_cage_cell, depths_l, faraday_solver);
	std::cout << "\tDone" << std::endl;
	std::cout << "Solving for directions " << std::endl;
	Eigen::MatrixXd u_all_dirs(CN_l.rows(), ico_pts_2.rows());
	Eigen::MatrixXd u_diff_all_dirs(CN_l.rows(), ico_pts_2.rows());
	for (size_t i = 0; i < ico_pts_2.rows(); i++) {
		std::cout << "\tSolving direction " << i << std::endl;
		Eigen::VectorXd bdry_vals_dir = Eigen::VectorXd::Zero(CN_l.rows());
		Eigen::VectorXd dir = ico_pts_2.row(i);
		for (size_t leaf = 0; leaf < CN_l.rows(); leaf++) bdry_vals_dir[leaf] = dir.dot(CN_l.row(leaf));
		Eigen::VectorXd sol = solveFaraday(CN_l, W_l, neighs, is_boundary_cell, is_cage_cell, depths_l, bdry_vals_dir, faraday_solver, global_to_matrix_ordering);
		Eigen::VectorXd this_field_diff = (sol - bdry_vals_dir).cwiseAbs();
		Eigen::VectorXd this_field_diff_gradmag = grad2(W, CH, neighs, parents, all_to_leaf, leaf_to_all, this_field_diff).rowwise().norm();
		u_all_dirs.col(i) << sol;
		u_diff_all_dirs.col(i) << this_field_diff_gradmag;
	}

	Eigen::VectorXd u_max_gradmag = u_diff_all_dirs.rowwise().maxCoeff();
	std::cout << "\tTest dir is " << dir_1.transpose() << std::endl;
	Eigen::VectorXd u_faraday = u_all_dirs.col(0);
	Eigen::VectorXd field_diff = (u_faraday - bdry_vals).cwiseAbs();

	// Eigen::MatrixXd grad_faraday = grad(CN_l, W_l, neighs, is_boundary_cell, depths, bdry_vals, u_faraday);
	// Eigen::MatrixXd grad_faraday_diff = grad(CN_l, W_l, neighs, is_boundary_cell, depths, bdry_vals, field_diff);
	// Eigen::MatrixXd grad_base = grad(CN_l, W_l, neighs, is_boundary_cell, depths, bdry_vals, bdry_vals);
	// Eigen::MatrixXd grad_max_diff = grad(CN_l, W_l, neighs, is_boundary_cell, depths, bdry_vals, u_max_diff);
	/*
		Eigen::MatrixXd grad2(  Eigen::VectorXd &W_all, Eigen::MatrixXi &CH,  std::vector<struct CellNeighbors> &neighs, 
                        Eigen::VectorXi &parents, std::unordered_map<int, int> &all_to_leaf, std::unordered_map<int, int> &leaf_to_all, 
                        Eigen::VectorXd &f);
	*/ 

	// Eigen::MatrixXd grad_faraday = grad2(W, CH, neighs, parents, all_to_leaf, leaf_to_all, u_faraday);
	// Eigen::MatrixXd grad_faraday_diff = grad2(W, CH, neighs, parents, all_to_leaf, leaf_to_all, field_diff);
	Eigen::MatrixXd grad_faraday_diff = grad2(W, CH, neighs, parents, all_to_leaf, leaf_to_all, field_diff);
	Eigen::MatrixXd grad_base = grad2(W, CH, neighs, parents, all_to_leaf, leaf_to_all, bdry_vals);
	Eigen::MatrixXd grad_max_diff = grad2(W, CH, neighs, parents, all_to_leaf, leaf_to_all, u_max_gradmag);

	// compute normals
	// put cell gradients on corresponding cage points
	Eigen::MatrixXd normals_est = Eigen::MatrixXd::Zero(P.rows(), 3);
	for (size_t leaf = 0; leaf < CN_l.rows(); leaf++) {
		// assign gradient to each cage point
		if (is_cage_cell(leaf)) {
			normals_est.row(PI_l[leaf][0]) = grad_max_diff.row(leaf);
		}
	}
	for (size_t pt_idx = 0; pt_idx < P.rows(); pt_idx++) {
		// average over cage points associated with a particular point
		if (!(is_boundary_point(pt_idx)) && !(is_cage_point(pt_idx))) {
			for (int cage: my_cage_points[pt_idx]) {
				normals_est.row(pt_idx) += normals_est.row(cage) / 12;
			}
		}
	}
	Eigen::MatrixXd normals_est_pts_og = Eigen::MatrixXd::Zero(P.rows(), 3);
	for (size_t pt_idx = 0; pt_idx < P.rows(); pt_idx++) {
		if (!(is_boundary_point(pt_idx)) && !(is_cage_point(pt_idx))) {
			normals_est_pts_og.row(pt_idx) = normals_est.row(pt_idx).normalized();
		}
	}


	
	std::cout << "\tDone" << std::endl;

	// Visualize!

	auto pc = polyscope::registerPointCloud("Points", P);
	pc->addVectorQuantity("True Normals", N);
	pc->addVectorQuantity("Estimated Normals", normals_est_pts_og);
	// test: cage point mapping
	// Eigen::VectorXi cage_point_test = Eigen::VectorXi::Zero(P.rows());
	// for (int cage: my_cage_points[0]) {
	// 	std::cout << "\t" << cage << std::endl;
	// 	cage_point_test[cage] = 1;
	// }
	// pc->addScalarQuantity("Cage vertex (test)", cage_point_test);
	pc->setEnabled(false);
	auto pc_cn = polyscope::registerPointCloud("Octree cell centers (leaves)", CN_l);
	pc_cn->addScalarQuantity("Depth", depths_l);
	pc_cn->addScalarQuantity("Parents", parents_l);

	auto pc_oc = polyscope::registerCurveNetwork("Octree edges", oc_pts, oc_edges);
	pc_oc->setEnabled(false);

	pc_cn->addScalarQuantity("Boundary cells", is_boundary_cell);
	pc_cn->addScalarQuantity("Cage cells", is_cage_cell);

	pc_cn->addScalarQuantity("Laplacian solve (uniform)", u_uni);
	Eigen::VectorXd laplacian_error_uniform = (u_uni - bdry_vals).cwiseAbs();
	auto vs_le_u = pc_cn->addScalarQuantity("Laplacian error (uniform)", laplacian_error_uniform);

	// pc_cn->addScalarQuantity("Laplacian solve (weighted)", u_weight);
	// Eigen::VectorXd laplacian_error_weighted = (u_weight - bdry_vals).cwiseAbs();
	// auto vs_le_w = pc_cn->addScalarQuantity("Laplacian error (weighted, 2 neighbors)", laplacian_error_weighted);

	pc_cn->addScalarQuantity("Laplacian solve (weighted, 4 neighbors)", u_weight_4);
	Eigen::VectorXd laplacian_error_weighted_4 = (u_weight_4 - bdry_vals).cwiseAbs();
	auto vs_le_w_4 = pc_cn->addScalarQuantity("Laplacian error (weighted, 4 neighbors)", laplacian_error_weighted_4);

	pc_cn->addScalarQuantity("Base field", bdry_vals);
	auto vs_faraday = pc_cn->addScalarQuantity("Faraday solve", u_faraday);
	
	// auto vs_faraday_diff = pc_cn->addScalarQuantity("Faraday, field difference", field_diff);
	// auto vs_faraday_grad = pc_cn->addVectorQuantity("Faraday, grad", grad_faraday);
	// auto vs_faraday_diff_grad = pc_cn->addVectorQuantity("Faraday, field difference (grad)", grad_faraday_diff);
	auto vs_base_grad = pc_cn->addVectorQuantity("Base, grad", grad_base);
	Eigen::MatrixXd dir_grad(CN_l.rows(), 3);
	dir_grad.rowwise() = dir_1.transpose();
	auto vs_base_true_grad = pc_cn->addVectorQuantity("Base, true grad", dir_grad);
	// vs_faraday_diff->setColorMap("reds");

	auto vs_max_diff = pc_cn->addScalarQuantity("Magnitude of largest gradient", u_max_gradmag);
	vs_max_diff->setColorMap("reds");
	auto vs_max_diff_grad = pc_cn->addVectorQuantity("Mag. largest grad., gradient", grad_max_diff);

	vs_le_u->setColorMap("blues");
	vs_le_u->setMapRange(std::pair<double,double>(0, fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted_4.maxCoeff())));
	// vs_le_w->setColorMap("blues");
	// vs_le_w->setMapRange(std::pair<double,double>(0, fmax(fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted.maxCoeff()), laplacian_error_weighted_4.maxCoeff())));
	vs_le_w_4->setColorMap("blues");
	vs_le_w_4->setMapRange(std::pair<double,double>(0, fmax(laplacian_error_uniform.maxCoeff(), laplacian_error_weighted_4.maxCoeff())));

	// generate_world_axes({3.,0.,0.});

	// test neighborhood connectivity
	int test_idx = 11415;
	Eigen::VectorXi is_test_neigh = Eigen::VectorXi::Zero(CN_l.rows());

	for (int neigh: neighs[test_idx].all) {
		is_test_neigh[neigh] = 1.;
	}

	pc_cn->addScalarQuantity("Connectivity (debug, vertex " + std::to_string(test_idx) + ")", is_test_neigh);

	int test_idx_2 = 9335;
	Eigen::VectorXi is_test_neigh_2 = Eigen::VectorXi::Zero(CN_l.rows());

	for (int neigh: neighs[test_idx_2].all) {
		is_test_neigh_2[neigh] = 1.;
	}

	pc_cn->addScalarQuantity("Connectivity (debug, vertex " + std::to_string(test_idx_2) + ")", is_test_neigh_2);

	// Give control to the polyscope gui
	polyscope::show();

	return EXIT_SUCCESS;
}
