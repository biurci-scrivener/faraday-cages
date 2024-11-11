#include "pc.h"

bool is_close(double a, double b) {
    return fabs(a - b) < 1e-12;
}

template <typename T> std::vector<int> sort_indexes(const std::vector<T> &v) {

    // from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes

    // initialize original index locations
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when v contains elements of equal values 
    std::stable_sort(idx.begin(), idx.end(),
        [&v](int i1, int i2) {return v[i1] < v[i2];});

    return idx;
}

template <typename T> std::vector<T> reorder_vector(const std::vector<T> &vals, const std::vector<int> &idxs) {
    std::vector<T> vals_new;
    for (int idx: idxs) {vals_new.push_back(vals[idx]);}
    return vals_new;
}

std::tuple<Eigen::VectorXi, Eigen::VectorXi, std::vector<std::vector<int>>, Eigen::MatrixXd> appendBoundaryAndCage(Eigen::MatrixXd &P, Eigen::MatrixXd &N) {
    Eigen::MatrixXd BV;
    Eigen::MatrixXi BF;

    igl::bounding_box(P, BV, BF);

    double PADDING = 0.25;

    Eigen::Vector3d bb_max = BV.row(0);
    Eigen::Vector3d bb_min = BV.row(7);

    Eigen::Vector3d pad = bb_max - bb_min * PADDING;
    pad = pad.cwiseAbs();
    Eigen::Vector3d delta = {   (bb_max[0] + pad[0]) - (bb_min[0] - pad[0]),
                                (bb_max[1] + pad[1]) - (bb_min[1] - pad[1]),
                                (bb_max[2] + pad[2]) - (bb_min[2] - pad[2])};
    
    Eigen::MatrixXd bb(2,3);
    bb.row(0) << (bb_min[0] - pad[0]), (bb_min[1] - pad[1]), (bb_min[2] - pad[2]);
    bb.row(1) << (bb_max[0] + pad[0]), (bb_max[1] + pad[1]), (bb_max[2] + pad[2]);

    int START_BDRY = P.rows();

    std::vector<Eigen::Vector3d> add_rows;

    // ADD BOUNDARY POINTS

        // corners
        add_rows.push_back({ bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] });
        add_rows.push_back({ bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] });
        add_rows.push_back({ bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] });
        add_rows.push_back({ bb_min[0] - pad[0], bb_max[1] + pad[1], bb_max[2] + pad[2] });
        add_rows.push_back({ bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] });
        add_rows.push_back({ bb_max[0] + pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] });
        add_rows.push_back({ bb_max[0] + pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] });
        add_rows.push_back({ bb_max[0] + pad[0], bb_max[1] + pad[1], bb_max[2] + pad[2] });

        // refine each square face
        size_t REFINE_DEG = 4;

        // bottom 3
        Eigen::Vector3d base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            for (size_t j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1] + (j * (delta[1] / REFINE_DEG)), base[2]});
            }
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            for (size_t j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            for (size_t j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        // top 3
        base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            for (size_t j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1] + (j * (delta[1] / REFINE_DEG)), base[2]});
            }
        }
        base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            for (size_t j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            for (size_t j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        
        // refine 12 edges
        base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
        base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
        base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
        base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_max[2] + pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        base = { bb_max[0] + pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        for (size_t i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
    
    int START_CAGE = P.rows() + add_rows.size();

    // ADD CAGE POINTS
    // Icosphere surrounding each interior point

    double CAGE_RADIUS = delta[0] / 100.;

    for (int i = 0; i < START_BDRY; i++) {

        Eigen::Vector3d pt = P.row(i);

        for (int j = 0; j < ico_pts.rows(); j++) {
            Eigen::Vector3d ico_pt = ico_pts.row(j);
            add_rows.push_back(pt + ico_pt * CAGE_RADIUS);
        }

    }

    // append points
    P.conservativeResize(P.rows() + add_rows.size(), P.cols());
    Eigen::VectorXi is_boundary_point = Eigen::VectorXi::Zero(P.rows());
    Eigen::VectorXi is_cage_point = Eigen::VectorXi::Zero(P.rows());
    std::vector<std::vector<int>> my_cage_points(P.rows(), std::vector<int>());
    

    int i = 0;
    for (Eigen::Vector3d row_to_add: add_rows) {
        P.row(START_BDRY + i) = row_to_add;
        if (i < START_CAGE - START_BDRY) {
            is_boundary_point[START_BDRY + i] = 1;
            
        } else {
            is_cage_point[START_BDRY + i] = 1;
            my_cage_points[(i - (START_CAGE - START_BDRY)) / 12].push_back(START_BDRY + i);
        }
        i++;
    }

    // add dummy normals for bdry points and cage points
    N.conservativeResize(N.rows() + add_rows.size(), N.cols());

    i = 0;
    for (Eigen::Vector3d row_to_add: add_rows) {
        N.row(START_BDRY + i) = Eigen::Vector3d{0., 0., 0.};
        i++;
    }

    return std::make_tuple(is_boundary_point, is_cage_point, my_cage_points, bb);

}

