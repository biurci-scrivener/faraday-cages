#include "oct_helper.h"

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

std::tuple<std::vector<int>, Eigen::VectorXd, double> getNClosestNeighs(int leaf, std::vector<int> &neighs, int n, Eigen::MatrixXd &CN) {

    /* neighs are neighbors on a particular side (right, top, etc.) */
    
    std::vector<double> dists;

    for (int neigh: neighs) {
        dists.push_back((CN.row(leaf) - CN.row(neigh)).norm());
    }

    std::vector<int> idxs = sort_indexes(dists);
    std::vector<int> closest_n;
    int count = 0;
    for (int n_idx: idxs) {
        if (count == n) break;
        closest_n.push_back(neighs[n_idx]);
        count++;
    }

    Eigen::VectorXd ctr = Eigen::VectorXd::Zero(CN.cols());
    for (int closest: closest_n) {
        ctr += CN.row(closest).transpose();
    }
    ctr /= closest_n.size();

    double dist = (ctr - CN.row(leaf).transpose()).norm();

    return std::make_tuple(closest_n, ctr, dist);

}

double getNeighDepthDelta(int leaf, std::vector<int> &neighs, Eigen::VectorXi &depths) {
    int highest_neigh_depth = -1;
    for (int neigh: neighs) {
        if (depths[neigh] > highest_neigh_depth) {
            highest_neigh_depth = depths[neigh];
        }
    }
    int depth_gap = depths[leaf] - highest_neigh_depth; // negative when neighbor is smaller
    return depth_gap;
}

double getDistanceFromDelta(int leaf, int depth_gap, Eigen::VectorXd &W) {return ((W[leaf] / 2.) + ((W[leaf] * pow(2, depth_gap)) / 2.));}

void searchForNeighbors(int leaf, std::vector<std::vector<int>> &PI, Eigen::VectorXi &search, Eigen::MatrixXd &CN, Eigen::VectorXd &W, Eigen::VectorXi &is_cage_point, 
                        std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges, Eigen::MatrixXd &bb,
                        Eigen::VectorXi &is_boundary_cell, Eigen::VectorXi &is_cage_cell, std::vector<struct CellNeighbors> &neighs) {

    std::vector<Eigen::Vector3d> corners_mine = { oc_pts[leaf * 8], oc_pts[leaf * 8 + 1],
                                                oc_pts[leaf * 8 + 2], oc_pts[leaf * 8 + 3],
                                                oc_pts[leaf * 8 + 4], oc_pts[leaf * 8 + 5],
                                                oc_pts[leaf * 8 + 6], oc_pts[leaf * 8 + 7]
                                                };

    for (Eigen::Vector3d corner: corners_mine) {
        if (((corner[0] == bb.row(0)[0]) || (corner[0] == bb.row(1)[0])) ||
            ((corner[1] == bb.row(0)[1]) || (corner[1] == bb.row(1)[1])) ||
            ((corner[2] == bb.row(0)[2]) || (corner[2] == bb.row(1)[2]))) {
            is_boundary_cell[leaf] = 1;
        };
    }   

    for (int child_p: PI[leaf]) {
        if (is_cage_point[child_p]) {
            is_cage_cell[leaf] = 1;
        }
    }

    bool using_knn = (search.size() > 1);
    int MAX_SEARCH = using_knn ? search.cols() : CN.rows();
    int other;

    for (int other_knn_idx = 0; other_knn_idx < MAX_SEARCH; other_knn_idx++) {
        if (using_knn) {
            other = search[other_knn_idx];
        } else {
            other = other_knn_idx;
        }

        if (other == leaf) continue;

        /*
            given that other is on a particular side of leaf,
            check to see if it touches leaf's cell on that side:
                if any 3+ corners of other are included within leaf's "face",
                it connects on that side
        */

        // we also have corners_mine
        // corners start at oc_pts[other * 8]
        std::vector<Eigen::Vector3d> corners = { oc_pts[other * 8], oc_pts[other * 8 + 1],
                                        oc_pts[other * 8 + 2], oc_pts[other * 8 + 3],
                                        oc_pts[other * 8 + 4], oc_pts[other * 8 + 5],
                                        oc_pts[other * 8 + 6], oc_pts[other * 8 + 7]
                                        };

        int match_right = 0;    
        int match_left = 0;  
        int match_top = 0;  
        int match_bottom = 0;  
        int match_front = 0;  
        int match_back = 0;                         

        for (Eigen::Vector3d corner: corners) {
            bool inXLimits = (corner[0] >= (CN(leaf, 0) - W[leaf] / 2.)) && (corner[0] <= (CN(leaf, 0) + W[leaf] / 2.));
            bool inYLimits = (corner[1] >= (CN(leaf, 1) - W[leaf] / 2.)) && (corner[1] <= (CN(leaf, 1) + W[leaf] / 2.));
            bool inZLimits = (corner[2] >= (CN(leaf, 2) - W[leaf] / 2.)) && (corner[2] <= (CN(leaf, 2) + W[leaf] / 2.));
            bool onRightPlane = (corner[0] == (CN(leaf, 0) + W[leaf] / 2.));
            bool onLeftPlane = (corner[0] == (CN(leaf, 0) - W[leaf] / 2.));
            bool onTopPlane = (corner[1] == (CN(leaf, 1) + W[leaf] / 2.));
            bool onBottomPlane = (corner[1] == (CN(leaf, 1) - W[leaf] / 2.));
            bool onFrontPlane = (corner[2] == (CN(leaf, 2) + W[leaf] / 2.));
            bool onBackPlane = (corner[2] == (CN(leaf, 2) - W[leaf] / 2.));

            if (onRightPlane && inYLimits && inZLimits) {
                match_right++;
            }

            if (onLeftPlane && inYLimits && inZLimits) {
                match_left++;
            }

            if (onTopPlane && inXLimits && inZLimits) {
                match_top++;
            }

            if (onBottomPlane && inXLimits && inZLimits) {
                match_bottom++;
            }

            if (onFrontPlane && inXLimits && inYLimits) {
                match_front++;
            }

            if (onBackPlane && inXLimits && inYLimits) {
                match_back++;
            }

        } 

        int match_right_rev = 0;    
        int match_left_rev = 0;  
        int match_top_rev = 0;  
        int match_bottom_rev = 0;  
        int match_front_rev = 0;  
        int match_back_rev = 0;  

        for (Eigen::Vector3d corner: corners_mine) {
            bool inXLimits = (corner[0] >= (CN(other, 0) - W[other] / 2.)) && (corner[0] <= (CN(other, 0) + W[other] / 2.));
            bool inYLimits = (corner[1] >= (CN(other, 1) - W[other] / 2.)) && (corner[1] <= (CN(other, 1) + W[other] / 2.));
            bool inZLimits = (corner[2] >= (CN(other, 2) - W[other] / 2.)) && (corner[2] <= (CN(other, 2) + W[other] / 2.));
            bool onRightPlane = (corner[0] == (CN(other, 0) + W[other] / 2.));
            bool onLeftPlane = (corner[0] == (CN(other, 0) - W[other] / 2.));
            bool onTopPlane = (corner[1] == (CN(other, 1) + W[other] / 2.));
            bool onBottomPlane = (corner[1] == (CN(other, 1) - W[other] / 2.));
            bool onFrontPlane = (corner[2] == (CN(other, 2) + W[other] / 2.));
            bool onBackPlane = (corner[2] == (CN(other, 2) - W[other] / 2.));

            if (onRightPlane && inYLimits && inZLimits) {
                match_left_rev++;
            }

            if (onLeftPlane && inYLimits && inZLimits) {
                match_right_rev++;
            }

            if (onTopPlane && inXLimits && inZLimits) {
                match_bottom_rev++;
            }

            if (onBottomPlane && inXLimits && inZLimits) {
                match_top_rev++;
            }

            if (onFrontPlane && inXLimits && inYLimits) {
                match_back_rev++;
            }

            if (onBackPlane && inXLimits && inYLimits) {
                match_front_rev++;
            }

        }  

        if ((match_right >= 3) || (match_right_rev >= 3)) {

            // if (leaf == 17) {
            //     std::cout << "17 is getting a right neighbor" << std::endl;
            // } else if (other == 17) {
            //     std::cout << "17 is getting a left neighbor" << std::endl;
            // }

            neighs[leaf].right.push_back(other);

            // std::vector<int> & leaf_s = neighs[leaf].right;
            // std::vector<int> & other_s = neighs[other].left;
            // if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
            // if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
        }

        if ((match_left >= 3) || (match_left_rev >= 3)) {

            // if (leaf == 17) {
            //     std::cout << "17 is getting a left neighbor" << std::endl;
            // } else if (other == 17) {
            //     std::cout << "17 is getting a right neighbor" << std::endl;
            // }

            neighs[leaf].left.push_back(other);

            // std::vector<int> & leaf_s = neighs[leaf].left;
            // std::vector<int> & other_s = neighs[other].right;
            // if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
            // if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
        }

        if ((match_top >= 3) || (match_top_rev >= 3)) {

            // if (leaf == 17) {
            //     std::cout << "17 is getting a top neighbor" << std::endl;
            // } else if (other == 17) {
            //     std::cout << "17 is getting a bottom neighbor" << std::endl;
            // }

            neighs[leaf].top.push_back(other);

            // std::vector<int> & leaf_s = neighs[leaf].top;
            // std::vector<int> & other_s = neighs[other].bottom;
            // if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
            // if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
        }

        if ((match_bottom >= 3) || (match_bottom_rev >= 3)) {

            // if (leaf == 17) {
            //     std::cout << "17 is getting a bottom neighbor" << std::endl;
            // } else if (other == 17) {
            //     std::cout << "17 is getting a top neighbor" << std::endl;
            // }

            neighs[leaf].bottom.push_back(other);

            // std::vector<int> & leaf_s = neighs[leaf].bottom;
            // std::vector<int> & other_s = neighs[other].top;
            // if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
            // if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
        }

        if ((match_front >= 3) || (match_front_rev >= 3)) {

            // if (leaf == 17) {
            //     std::cout << "17 is getting a front neighbor" << std::endl;
            // } else if (other == 17) {
            //     std::cout << "17 is getting a back neighbor" << std::endl;
            // }

            neighs[leaf].front.push_back(other);

            // std::vector<int> & leaf_s = neighs[leaf].front;
            // std::vector<int> & other_s = neighs[other].back;
            // if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
            // if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
        }

        if ((match_back >= 3) || (match_back_rev >= 3)) {

            // if (leaf == 17) {
            //     std::cout << "17 is getting a back neighbor" << std::endl;
            // } else if (other == 17) {
            //     std::cout << "17 is getting a front neighbor" << std::endl;
            // }

            neighs[leaf].back.push_back(other);

            // std::vector<int> & leaf_s = neighs[leaf].back;
            // std::vector<int> & other_s = neighs[other].front;
            // if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
            // if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
        }

    }
}

std::tuple<std::vector<struct CellNeighbors>, Eigen::VectorXi, Eigen::VectorXi> createLeafNeighbors(std::vector<std::vector<int>> &PI, Eigen::MatrixXd &CN, Eigen::VectorXd &W, Eigen::VectorXi &is_cage_point, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges, Eigen::MatrixXd &bb) {
    /*
    
        Takes CN_l and W_l (computes neighbor relationships for leaves only)

    */
    
    std::vector<struct CellNeighbors> neighs(CN.rows()); 
    Eigen::VectorXi is_boundary_cell = Eigen::VectorXi::Zero(CN.rows());
    Eigen::VectorXi is_cage_cell = Eigen::VectorXi::Zero(CN.rows());

    // DOUBLE OCTREE TIME

    // std::vector<std::vector<int >> PI_oct;
    // Eigen::MatrixXi CH_oct;
    // Eigen::MatrixXd CN_oct;
    // Eigen::VectorXd W_oct;
    // Eigen::MatrixXi knn;

    // igl::octree(CN, PI_oct, CH_oct, CN_oct, W_oct);
    // igl::knn(CN, 100, PI_oct, CH_oct, CN_oct, W_oct, knn);

    // std::cout << "Build KNN" << std::endl;

    Eigen::VectorXi search_null(1);

    // for (int leaf = 0; leaf < CN.rows(); leaf++) {

    //     Eigen::VectorXi search = knn.row(leaf);
    //     searchForNeighbors(leaf, PI, search, CN, W, is_cage_point, 
    //                             oc_pts, oc_edges, bb,
    //                             is_boundary_cell, is_cage_cell, neighs);
    // }

    // validate that each cell has neighbors on all sides
    // if not... rerun the search over ALL cells

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        if ((   (neighs[leaf].right.size() == 0) || (neighs[leaf].left.size() == 0) || (neighs[leaf].top.size() == 0) || 
                (neighs[leaf].bottom.size() == 0) || (neighs[leaf].front.size() == 0) || (neighs[leaf].back.size() == 0)) && (!is_boundary_cell(leaf))) {
            // std::cout << "\tTrying to fix " << leaf << std::endl;
            searchForNeighbors(leaf, PI, search_null, CN, W, is_cage_point, 
                        oc_pts, oc_edges, bb,
                        is_boundary_cell, is_cage_cell, neighs);
            // check again
            if ((   (neighs[leaf].right.size() == 0) || (neighs[leaf].left.size() == 0) || (neighs[leaf].top.size() == 0) || 
                (neighs[leaf].bottom.size() == 0) || (neighs[leaf].front.size() == 0) || (neighs[leaf].back.size() == 0)) && (!is_boundary_cell(leaf))) {
                throw std::runtime_error("Unfortunately, " + std::to_string(leaf) + " is not connected on all sides. This is a sad day");
            }
        }

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
                throw std::runtime_error(leaf_idx + " has more than one child");
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

std::tuple<Eigen::VectorXi, Eigen::VectorXi, Eigen::MatrixXd> appendBoundaryAndCage(Eigen::MatrixXd &P, Eigen::MatrixXd &N) {
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
        double REFINE_DEG = 5.;

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

    int i = 0;
    for (Eigen::Vector3d row_to_add: add_rows) {
        P.row(START_BDRY + i) = row_to_add;
        if (i < START_CAGE - START_BDRY) {
            is_boundary_point[START_BDRY + i] = 1;
        } else {
            is_cage_point[START_BDRY + i] = 1;
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

    return std::make_tuple(is_boundary_point, is_cage_point, bb);

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
