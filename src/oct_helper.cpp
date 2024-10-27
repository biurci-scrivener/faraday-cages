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

Eigen::VectorXi splitBoundaryInteriorCells(std::vector<struct CellNeighbors> neighs) {

    Eigen::VectorXi is_boundary = Eigen::VectorXi::Zero(neighs.size());

    for (int leaf = 0; leaf < neighs.size(); leaf++) {
        if ((neighs[leaf].right.size() == 0)    || (neighs[leaf].left.size() == 0) ||
            (neighs[leaf].top.size() == 0)      || (neighs[leaf].bottom.size() == 0) ||
            (neighs[leaf].front.size() == 0)    || (neighs[leaf].back.size() == 0)) {
            is_boundary[leaf] = 1;
        }
    }

    return is_boundary;

}

std::tuple<std::vector<int>, Eigen::VectorXd, double> getNClosestNeighs(int leaf, std::vector<int> &neighs, int n, Eigen::MatrixXd &CN) {
    
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

Eigen::VectorXd solveDirichletProblem(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &depths, Eigen::VectorXd bdry_vals, int laplacian) {

    // bdry_vals should be CN.rows() long, but only the entries for boundary vertices will be considered

    // reindex
    std::unordered_map<int, int> global_to_interior;

    int interior_count = 0;
    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        if (!is_boundary[leaf]) {
            global_to_interior.insert({leaf, interior_count});
            interior_count++;
        }
    }

    // build Laplacian 
    Eigen::SparseMatrix<double> L(interior_count, interior_count);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(interior_count);
    Eigen::VectorXd u(interior_count);
    std::vector<Eigen::Triplet<double>> triplets;

    if (laplacian == 0) {

        // uniform Laplacian
        for (int leaf = 0; leaf < CN.rows(); leaf++) {
            if (!is_boundary(leaf)) {
                double weight_sum = 0.;
                double b_sum = 0.;
                int current_idx = global_to_interior[leaf];
                for (int neigh: neighs[leaf].all) {

                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, 1));
                    } else {
                        b_sum += bdry_vals[neigh];
                    }

                    weight_sum++;

                }
                triplets.push_back(Eigen::Triplet<double>(current_idx, current_idx, -weight_sum));
                b[current_idx] = -b_sum;
            }
        }

    } else if (laplacian == 1) {

        // weighted Laplacian
        for (int leaf = 0; leaf < CN.rows(); leaf++) {
            if (!is_boundary(leaf)) {
                double weight_sum = 0.;
                double b_sum = 0.;
                int current_idx = global_to_interior[leaf];

                // on any given side, either all of your neighbors are boundary, or none of them are
                // this is due to the fact that the octree cells on the boundary will always be evenly spaced
                // (true by the way we construct the input point set)

                // compute distances
                int right_depth_delta = getNeighDepthDelta(leaf, neighs[leaf].right, depths);
                double right_distance = getDistanceFromDelta(leaf, right_depth_delta, W);

                int left_depth_delta = getNeighDepthDelta(leaf, neighs[leaf].left, depths);
                double left_distance = getDistanceFromDelta(leaf, left_depth_delta, W);

                int top_depth_delta = getNeighDepthDelta(leaf, neighs[leaf].top, depths);
                double top_distance = getDistanceFromDelta(leaf, top_depth_delta, W);

                int bottom_depth_delta = getNeighDepthDelta(leaf, neighs[leaf].bottom, depths);
                double bottom_distance = getDistanceFromDelta(leaf, bottom_depth_delta, W);

                int front_depth_delta = getNeighDepthDelta(leaf, neighs[leaf].front, depths);
                double front_distance = getDistanceFromDelta(leaf, front_depth_delta, W);

                int back_depth_delta = getNeighDepthDelta(leaf, neighs[leaf].back, depths);
                double back_distance = getDistanceFromDelta(leaf, back_depth_delta, W);

                if (current_idx == 0) {
                    std::cout << leaf << " " << right_distance << " " << left_distance << " " << top_distance << " " 
                    << bottom_distance << " " << front_distance << " " << back_distance << std::endl;
                }

                // add right-left connections 
                for (int neigh: neighs[leaf].right) {
                    double weight = (2. / ((left_distance + right_distance) * right_distance));
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                for (int neigh: neighs[leaf].left) {
                    double weight = (2. / ((left_distance + right_distance) * left_distance));
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                weight_sum += (2. / (left_distance * right_distance));

                // add top-bottom connections 
                for (int neigh: neighs[leaf].top) {
                    double weight = (2. / ((top_distance + bottom_distance) * top_distance));
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                for (int neigh: neighs[leaf].bottom) {
                    double weight = (2. / ((top_distance + bottom_distance) * bottom_distance));
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                weight_sum += (2. / (top_distance * bottom_distance));

                // add top-bottom connections 
                for (int neigh: neighs[leaf].front) {
                    double weight = (2. / ((front_distance + back_distance) * front_distance));
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                for (int neigh: neighs[leaf].back) {
                    double weight = (2. / ((front_distance + back_distance) * back_distance));
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                weight_sum += (2. / (front_distance * back_distance));
                
                triplets.push_back(Eigen::Triplet<double>(current_idx, current_idx, -weight_sum));
                b[current_idx] = -b_sum;
            }
        }

    } else if (laplacian >= 2) {
        // weighted Laplacian which does the following:
        /*
            On each of six sides, pick the k neighboring cells whose centers are closest
            and average their positions.
            Otherwise, doesn't account for "size" of neighboring cells at all
        */

        int NUM_NEIGHS = laplacian;

        for (int leaf = 0; leaf < CN.rows(); leaf++) {
            if (!is_boundary(leaf)) {
                double weight_sum = 0.;
                double b_sum = 0.;
                int current_idx = global_to_interior[leaf];



                std::vector<int> right_closest;
                Eigen::VectorXd right_ctr(3);
                double right_distance;
                if (neighs[leaf].right.size() > 1) {
                    std::tie(right_closest, right_ctr, right_distance) = getNClosestNeighs(leaf, neighs[leaf].right, NUM_NEIGHS, CN);
                } else {
                    right_closest = neighs[leaf].right;
                    right_ctr << CN(neighs[leaf].right[0], 0), CN(leaf, 1), CN(leaf, 2);
                    right_distance = abs(right_ctr[0] - CN(leaf, 0));
                }

                std::vector<int> left_closest;
                Eigen::VectorXd left_ctr(3);
                double left_distance;
                
                if (neighs[leaf].left.size() > 1) {
                    std::tie(left_closest, left_ctr, left_distance) = getNClosestNeighs(leaf, neighs[leaf].left, NUM_NEIGHS, CN);
                } else {
                    left_closest = neighs[leaf].left;
                    left_ctr << CN(neighs[leaf].left[0], 0), CN(leaf, 1), CN(leaf, 2);
                    left_distance = abs(left_ctr[0] - CN(leaf, 0));
                }

                std::vector<int> top_closest;
                Eigen::VectorXd top_ctr(3);
                double top_distance;
                if (neighs[leaf].top.size() > 1) {
                    std::tie(top_closest, top_ctr, top_distance) = getNClosestNeighs(leaf, neighs[leaf].top, NUM_NEIGHS, CN);
                } else {
                    top_closest = neighs[leaf].top;
                    top_ctr << CN(leaf, 0), CN(neighs[leaf].top[0], 1), CN(leaf, 2);
                    top_distance = abs(top_ctr[1] - CN(leaf, 1));
                }
                
                std::vector<int> bottom_closest;
                Eigen::VectorXd bottom_ctr(3);
                double bottom_distance;
                if (neighs[leaf].bottom.size() > 1) {
                    std::tie(bottom_closest, bottom_ctr, bottom_distance) = getNClosestNeighs(leaf, neighs[leaf].bottom, NUM_NEIGHS, CN);
                } else {
                    bottom_closest = neighs[leaf].bottom;
                    bottom_ctr << CN(leaf, 0), CN(neighs[leaf].bottom[0], 1), CN(leaf, 2);
                    bottom_distance = abs(bottom_ctr[1] - CN(leaf, 1));
                }
                
                std::vector<int> front_closest;
                Eigen::VectorXd front_ctr(3);
                double front_distance;
                if (neighs[leaf].front.size() > 1) {
                    std::tie(front_closest, front_ctr, front_distance) = getNClosestNeighs(leaf, neighs[leaf].front, NUM_NEIGHS, CN);
                } else {
                    front_closest = neighs[leaf].front;
                    front_ctr << CN(leaf, 0), CN(leaf, 1), CN(neighs[leaf].front[0], 2);
                    front_distance = abs(front_ctr[2] - CN(leaf, 2));
                }

                std::vector<int> back_closest;
                Eigen::VectorXd back_ctr(3);
                double back_distance;
                if (neighs[leaf].back.size() > 1) {
                    std::tie(back_closest, back_ctr, back_distance) = getNClosestNeighs(leaf, neighs[leaf].back, NUM_NEIGHS, CN);
                } else {
                    back_closest = neighs[leaf].back;
                    back_ctr << CN(leaf, 0), CN(leaf, 0), CN(neighs[leaf].back[0], 2);
                    back_distance = abs(back_ctr[2] - CN(leaf, 2));
                }

                // ==== 

                if (current_idx == 0) {
                    std::cout << leaf << " dists " << right_distance << " " << left_distance << " " << top_distance << " " 
                    << bottom_distance << " " << front_distance << " " << back_distance << std::endl;
                }

                // add right-left connections 
                for (int neigh: right_closest) {
                    double weight = (2. / ((left_distance + right_distance) * right_distance)) / right_closest.size();
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                for (int neigh: left_closest) {
                    double weight = (2. / ((left_distance + right_distance) * left_distance)) / left_closest.size();
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                weight_sum += (2. / (left_distance * right_distance));

                // add top-bottom connections 
                for (int neigh: top_closest) {
                    double weight = (2. / ((top_distance + bottom_distance) * top_distance)) / top_closest.size();
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                for (int neigh: bottom_closest) {
                    double weight = (2. / ((top_distance + bottom_distance) * bottom_distance)) / bottom_closest.size();
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                weight_sum += (2. / (top_distance * bottom_distance));

                // add top-bottom connections 
                for (int neigh: front_closest) {
                    double weight = (2. / ((front_distance + back_distance) * front_distance)) / front_closest.size();
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                for (int neigh: back_closest) {
                    double weight = (2. / ((front_distance + back_distance) * back_distance)) / back_closest.size();
                    if (!is_boundary(neigh)) {
                        int other_idx = global_to_interior[neigh];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neigh];
                    }
                }
                weight_sum += (2. / (front_distance * back_distance));
                
                triplets.push_back(Eigen::Triplet<double>(current_idx, current_idx, -weight_sum));
                b[current_idx] = -b_sum;
            }
        }
    }

    L.setFromTriplets(triplets.begin(), triplets.end());

    L = -L;
    b = -b;

    // std::cout << L.row(0) << std::endl;

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(L);
    if (solver.info() != Eigen::Success) {
        std::cout << "ERROR: Decomposition failed!" << std::endl;
        exit(-1);
    }
    u = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cout << "ERROR: Solve failed!" << std::endl;
        exit(-1);
    }

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(CN.rows());

    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        if (!is_boundary(leaf)) {
            int current_idx = global_to_interior[leaf];
            sol[leaf] = u[current_idx];
        } else {
            sol[leaf] = bdry_vals[leaf];
        }
    }

    

    return sol;

}

std::vector<struct CellNeighbors> createLeafNeighbors(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges) {
    /*
    
        Takes CN_l and W_l (computes neighbor relationships for leaves only)

    */
    
    std::vector<struct CellNeighbors> neighs(CN.rows()); 

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        for (int other = 0; other < CN.rows(); other++) {

            if (other == leaf) continue;

            /*
                given that other is on a particular side of leaf,
                check to see if it touches leaf's cell on that side:
                    if any 3+ corners of other are included within leaf's "face",
                    it connects on that side
            */

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

            if (match_right >= 3) {
                std::vector<int> & leaf_s = neighs[leaf].right;
                std::vector<int> & other_s = neighs[other].left;
                if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
                if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
            }

            if (match_left >= 3) {
                std::vector<int> & leaf_s = neighs[leaf].left;
                std::vector<int> & other_s = neighs[other].right;
                if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
                if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
            }

            if (match_top >= 3) {
                std::vector<int> & leaf_s = neighs[leaf].top;
                std::vector<int> & other_s = neighs[other].bottom;
                if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
                if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
            }

            if (match_bottom >= 3) {
                std::vector<int> & leaf_s = neighs[leaf].bottom;
                std::vector<int> & other_s = neighs[other].top;
                if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
                if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
            }

            if (match_front >= 3) {
                std::vector<int> & leaf_s = neighs[leaf].front;
                std::vector<int> & other_s = neighs[other].back;
                if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
                if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
            }

            if (match_back >= 3) {
                std::vector<int> & leaf_s = neighs[leaf].back;
                std::vector<int> & other_s = neighs[other].front;
                if (std::find(leaf_s.begin(), leaf_s.end(), other) == leaf_s.end()) leaf_s.push_back(other);
                if (std::find(other_s.begin(), other_s.end(), leaf) == other_s.end()) other_s.push_back(leaf);
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

    return neighs;
}