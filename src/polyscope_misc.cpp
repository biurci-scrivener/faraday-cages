#include "polyscope_misc.h"

void generate_world_axes(Eigen::Vector3d pos) {
    std::vector<Eigen::Vector3d> colors;
    colors.push_back({1.,0.,0.});
    colors.push_back({0.,1.,0.});
    colors.push_back({0.,0.,1.});
    std::vector<Eigen::Vector3d> dirs;
    dirs.push_back({0.5,0.,0.});
    dirs.push_back({0.,0.5,0.});
    dirs.push_back({0.,0.,0.5});
    polyscope_plot_vectors("World axes", pos, dirs, colors);
}

void polyscope_plot_vectors(std::string name, Eigen::Vector3d pos, std::vector<Eigen::Vector3d> dirs, std::vector<Eigen::Vector3d> colors) {
    std::vector<Eigen::Vector3d> p;
    p.push_back(pos);
    auto _pc = polyscope::registerPointCloud(name, p);
    _pc->setPointColor({0.,0.,0.});
    _pc->setPointRadius(0.005);
    for (size_t i = 0; i < dirs.size(); i++) {
        std::vector<Eigen::Vector3d> q;
        q.push_back(dirs[i]);
        auto _vec = _pc->addVectorQuantity(std::to_string(i), q, polyscope::VectorType::AMBIENT);
        _vec->setVectorColor({colors[i][0], colors[i][1], colors[i][2]});
        _vec->setVectorRadius(0.01);
        _vec->setEnabled(true);
    }
}