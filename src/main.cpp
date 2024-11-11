#include "args/args.hxx"
#include "imgui.h"

#include "io.h"
#include "polyscope_misc.h"
#include "pc.h"

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

	auto pc = polyscope::registerPointCloud("Points", P);
	pc->addVectorQuantity("True Normals", N);
	pc->setEnabled(true);

	// Give control to the polyscope gui
	polyscope::show();

	return EXIT_SUCCESS;
}
