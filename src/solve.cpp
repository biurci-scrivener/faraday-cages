#include "solve.h"

Eigen::VectorXd solveFaraday(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &is_cage, Eigen::VectorXi &depths, Eigen::VectorXd bdry_vals) {

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

    // weighted Laplacian which does the following:
    /*
        On each of six sides, pick the k neighboring cells whose centers are closest
        and average their positions.
        Otherwise, doesn't account for "size" of neighboring cells at all
    */

    int NUM_NEIGHS = 4.;

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

    } else if (laplacian >= 1) {
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

                // ===

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