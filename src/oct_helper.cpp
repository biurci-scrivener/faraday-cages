#include "oct_helper.h"

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

Eigen::MatrixXd octreeBB(Eigen::MatrixXd &CN) {

    Eigen::MatrixXd BV;
    Eigen::MatrixXi BF;

    igl::bounding_box(CN, BV, BF);
    
    Eigen::MatrixXd bb(2,3);
    bb.row(0) << BV.row(7); // min
    bb.row(1) << BV.row(0); // max

    return bb;
}

std::tuple<Eigen::VectorXd, double> getNeighRep(int leaf, std::vector<int> &neighs, size_t n, Eigen::MatrixXd &CN, Eigen::VectorXd &W, int side) {
    /*
        Returns 
        - average of n-closest points
        - distance of leaf center to average along specified axis
    */

    Eigen::VectorXd ctr(3); 
    if (neighs.size() > 0) {
        ctr = Eigen::VectorXd::Zero(3);
        for (int i = 0; i < std::min(n, neighs.size()); i++) {ctr += CN.row(neighs[i]);}
        ctr /= neighs.size();
    } else {
        ctr << CN(leaf, 0), CN(leaf, 1), CN(leaf, 2);
        ctr[side / 2] += W[leaf] * (side % 2 == 0 ? 1 : -1);
    }
    
    double dist = fabs(ctr[side / 2] - CN(leaf, side / 2));

    if ((dist == 0.) || (dist != dist)) {
        std::cout << "\tLeaf: " << leaf << std::endl;
        std::cout << "\tSide: " << side << std::endl;
        std::cout << "\tneighs.size(): " << neighs.size() << std::endl;
        throw std::runtime_error("getNeighRep generated bad distance " + std::to_string(dist));
    }

    return std::make_tuple(ctr, dist);

}

std::tuple<Eigen::VectorXd, double, double> getNeighRep(int leaf, std::vector<int> &neighs, size_t n, Eigen::MatrixXd &CN, Eigen::VectorXd &W, int side, Eigen::VectorXd &f) {
    /*
        Returns 
        - average of n-closest points
        - distance of leaf center to average along specified axis
    */

    Eigen::VectorXd ctr(3); 
    double f_val = 0;
    if (neighs.size() > 0) {
        ctr = Eigen::VectorXd::Zero(3);
        for (int i = 0; i < std::min(n, neighs.size()); i++) {
            ctr += CN.row(neighs[i]);
            f_val += f[neighs[i]];
        }
        ctr /= neighs.size();
        f_val /= neighs.size();
    } else {
        ctr << CN(leaf, 0), CN(leaf, 1), CN(leaf, 2);
        ctr[side / 2] += W[leaf] * (side % 2 == 0 ? 1 : -1);
        f_val = f[leaf];
    }
    
    double dist = fabs(ctr[side / 2] - CN(leaf, side / 2));

    if ((dist == 0.) || (dist != dist)) {
        std::cout << "\tLeaf: " << leaf << std::endl;
        std::cout << "\tSide: " << side << std::endl;
        std::cout << "\tneighs.size(): " << neighs.size() << std::endl;
        throw std::runtime_error("getNeighRep generated bad distance " + std::to_string(dist));
    }

    return std::make_tuple(ctr, dist, f_val);

}

std::vector<int> sortNeighbors(int leaf, std::vector<int> &neighs, Eigen::MatrixXd &CN) {

    // sort by distance
    std::vector<double> dists;
    for (int neigh: neighs) {
        dists.push_back((CN.row(leaf) - CN.row(neigh)).array().pow(2).sum());
    }
    std::vector<int> idxs = sort_indexes(dists);
    return reorder_vector(neighs, idxs);

}

void searchForNeighbors(int leaf, std::vector<std::vector<int>> &PI, Eigen::VectorXi &search, Eigen::MatrixXd &CN, Eigen::VectorXd &W, Eigen::VectorXi &is_cage_point, 
                        Eigen::MatrixXd &bb_oct, Eigen::VectorXi &is_boundary_cell, Eigen::VectorXi &is_cage_cell, std::vector<struct CellNeighbors> &neighs) {

    if (    (CN(leaf, 0) - (W[leaf] / 2.) <= bb_oct(0, 0)) || (CN(leaf, 0) + (W[leaf] / 2.) >= bb_oct(1, 0)) ||
            (CN(leaf, 1) - (W[leaf] / 2.) <= bb_oct(0, 1)) || (CN(leaf, 1) + (W[leaf] / 2.) >= bb_oct(1, 1)) ||
            (CN(leaf, 2) - (W[leaf] / 2.) <= bb_oct(0, 2)) || (CN(leaf, 2) + (W[leaf] / 2.) >= bb_oct(1, 2)) ) is_boundary_cell[leaf] = 1;

    for (int child_p: PI[leaf]) {
        if (is_cage_point[child_p]) {
            is_cage_cell[leaf] = 1;
        }
    }

    for (int other = leaf + 1; other < CN.rows(); other++) {

        /*
            given that other is on a particular side of leaf,
            check to see if it touches leaf's cell on that side:
        */

        bool inXLimits = (fabs(CN(leaf, 0) - CN(other, 0)) < ((W[leaf] / 2.) + (W[other] / 2.)));
        bool inYLimits = (fabs(CN(leaf, 1) - CN(other, 1)) < ((W[leaf] / 2.) + (W[other] / 2.)));
        bool inZLimits = (fabs(CN(leaf, 2) - CN(other, 2)) < ((W[leaf] / 2.) + (W[other] / 2.)));
        bool onRightPlane = is_close((CN(other, 0) - CN(leaf, 0)), ((W[leaf] / 2.) + (W[other] / 2.)));
        bool onLeftPlane = is_close((CN(leaf, 0) - CN(other, 0)), ((W[leaf] / 2.) + (W[other] / 2.)));
        bool onTopPlane = is_close((CN(other, 1) - CN(leaf, 1)), ((W[leaf] / 2.) + (W[other] / 2.)));
        bool onBottomPlane = is_close((CN(leaf, 1) - CN(other, 1)), ((W[leaf] / 2.) + (W[other] / 2.)));
        bool onFrontPlane = is_close((CN(other, 2) - CN(leaf, 2)), ((W[leaf] / 2.) + (W[other] / 2.)));
        bool onBackPlane = is_close((CN(leaf, 2) - CN(other, 2)), ((W[leaf] / 2.) + (W[other] / 2.)));

        bool match_right = (onRightPlane && inYLimits && inZLimits);   
        bool match_left = (onLeftPlane && inYLimits && inZLimits);   
        bool match_top = (onTopPlane && inXLimits && inZLimits);   
        bool match_bottom = (onBottomPlane && inXLimits && inZLimits);   
        bool match_front = (onFrontPlane && inYLimits && inXLimits);   
        bool match_back = (onBackPlane && inYLimits && inXLimits);   

        if (match_right) {
            neighs[leaf].right.push_back(other);
            neighs[other].left.push_back(leaf);
        } if (match_left) {
            neighs[leaf].left.push_back(other);
            neighs[other].right.push_back(leaf);
        } if (match_top) {
            neighs[leaf].top.push_back(other);
            neighs[other].bottom.push_back(leaf);
        } if (match_bottom) {
            neighs[leaf].bottom.push_back(other);
            neighs[other].top.push_back(leaf);
        } if (match_front) {
            neighs[leaf].front.push_back(other);
            neighs[other].back.push_back(leaf);
        } if (match_back) {
            neighs[leaf].back.push_back(other);
            neighs[other].front.push_back(leaf);
        }

    }
}

std::tuple<std::vector<struct CellNeighbors>, Eigen::VectorXi, Eigen::VectorXi> createLeafNeighbors(std::vector<std::vector<int>> &PI, Eigen::MatrixXd &CN, Eigen::VectorXd &W, Eigen::VectorXi &is_cage_point, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges, Eigen::MatrixXd &bb_oct) {
    /*
    
        Takes CN_l and W_l (computes neighbor relationships for leaves only)

    */
    
    std::vector<struct CellNeighbors> neighs(CN.rows()); 
    Eigen::VectorXi is_boundary_cell = Eigen::VectorXi::Zero(CN.rows());
    Eigen::VectorXi is_cage_cell = Eigen::VectorXi::Zero(CN.rows());

    Eigen::VectorXi search_null(1);

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        searchForNeighbors(leaf, PI, search_null, CN, W, is_cage_point, 
                        bb_oct, is_boundary_cell, is_cage_cell, neighs);

        if ((   (neighs[leaf].right.size() == 0) || (neighs[leaf].left.size() == 0) || (neighs[leaf].top.size() == 0) || 
                (neighs[leaf].bottom.size() == 0) || (neighs[leaf].front.size() == 0) || (neighs[leaf].back.size() == 0)) && (!is_boundary_cell(leaf))) {
                std::cout << "ERROR at leaf: " << leaf << std::endl;
                std::cout << "\tRight neighs: ";
                for (int n: neighs[leaf].right) {
                    std::cout << n << " ";
                }
                std::cout << std::endl;

                std::cout << "\tLeft neighs: ";
                for (int n: neighs[leaf].left) {
                    std::cout << n << " ";
                }
                std::cout << std::endl;

                std::cout << "\tTop neighs: ";
                for (int n: neighs[leaf].top) {
                    std::cout << n << " ";
                }
                std::cout << std::endl;

                std::cout << "\tBottom neighs: ";
                for (int n: neighs[leaf].bottom) {
                    std::cout << n << " ";
                }
                std::cout << std::endl;

                std::cout << "\tFront neighs: ";
                for (int n: neighs[leaf].front) {
                    std::cout << n << " ";
                }
                std::cout << std::endl;

                std::cout << "\tBack neighs: ";
                for (int n: neighs[leaf].back) {
                    std::cout << n << " ";
                }
                std::cout << std::endl;

                std::cout << "\tIs boundary: " << (is_boundary_cell(leaf) ? "Yes" : "No") << std::endl;
                std::cout << "\tIs cage: " << (is_cage_cell(leaf) ? "Yes" : "No") << std::endl;
                std::cout << "\tCenter: " << CN(leaf, 0) << " " << CN(leaf, 1) << " " << CN(leaf, 2) << " " << std::endl;
                std::cout << "\tWidth: " << W(leaf) << std::endl;
                std::cout << "\tBB_oct: " << bb_oct(0, 0) << " " << bb_oct(0, 1) << " " << bb_oct(0, 2) << ", " << bb_oct(1, 0) << " " << bb_oct(1, 1) << " " << bb_oct(1, 2) << " " << std::endl;

                throw std::runtime_error("Unfortunately, " + std::to_string(leaf) + " is not connected on all sides. This is a sad day");
            }
        
        // sort each neighbor list by ascending distance
        neighs[leaf].right = sortNeighbors(leaf, neighs[leaf].right, CN);
        neighs[leaf].left = sortNeighbors(leaf, neighs[leaf].left, CN);
        neighs[leaf].top = sortNeighbors(leaf, neighs[leaf].top, CN);
        neighs[leaf].bottom = sortNeighbors(leaf, neighs[leaf].bottom, CN);
        neighs[leaf].front = sortNeighbors(leaf, neighs[leaf].front, CN);
        neighs[leaf].back = sortNeighbors(leaf, neighs[leaf].back, CN);

        for (int n: neighs[leaf].right) {
            neighs[leaf].all.push_back(n);
        }
        for (int n: neighs[leaf].left) {
            neighs[leaf].all.push_back(n);
        }
        for (int n: neighs[leaf].top) {
            neighs[leaf].all.push_back(n);
        }
        for (int n: neighs[leaf].bottom) {
            neighs[leaf].all.push_back(n);
        }
        for (int n: neighs[leaf].front) {
            neighs[leaf].all.push_back(n);
        }
        for (int n: neighs[leaf].back) {
            neighs[leaf].all.push_back(n);
        }

        if (leaf % 1000 == 0) {
            std::cout << "\tReached " << leaf << std::endl;
        }
        
    }

    return std::make_tuple(neighs, is_boundary_cell, is_cage_cell);
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

            if (PI_l[leaf_idx].size() > 1) {
                throw std::runtime_error(std::to_string(leaf_idx) + " has more than one child");
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
        double REFINE_DEG = 6.;

        // bottom 3
        Eigen::Vector3d base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            for (int j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1] + (j * (delta[1] / REFINE_DEG)), base[2]});
            }
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            for (int j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            for (int j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        // top 3
        base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            for (int j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1] + (j * (delta[1] / REFINE_DEG)), base[2]});
            }
        }
        base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            for (int j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            for (int j = 1; j < REFINE_DEG; j++) {
                add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2] + (j * (delta[2] / REFINE_DEG))});
            }
        }
        
        // refine 12 edges
        base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
        base = { bb_min[0] - pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
        base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1], base[2] + (i * (delta[2] / REFINE_DEG))});
        }
        base = { bb_min[0] - pad[0], bb_max[1] + pad[1], bb_max[2] + pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0] + (i * (delta[0] / REFINE_DEG)), base[1], base[2]});
        }
        base = { bb_max[0] + pad[0], bb_min[1] - pad[1], bb_max[2] + pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
            add_rows.push_back({base[0], base[1] + (i * (delta[1] / REFINE_DEG)), base[2]});
        }
        base = { bb_max[0] + pad[0], bb_max[1] + pad[1], bb_min[2] - pad[2] };
        for (int i = 1; i < REFINE_DEG; i++) {
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
