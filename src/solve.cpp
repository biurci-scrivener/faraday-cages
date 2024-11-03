#include "solve.h"

Eigen::MatrixXd grad(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &depths, Eigen::VectorXd &bdry_vals, Eigen::VectorXd &f) {

    Eigen::MatrixXd grad_all(CN.rows(), 3);
    int NUM_NEIGHS = 64;

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        Eigen::VectorXd grad_l = Eigen::VectorXd::Zero(3);

        std::vector<int> right_closest;
        Eigen::VectorXd right_ctr(3);
        double right_distance;
        double right_f = 0;
        if (neighs[leaf].right.size() == 0) {
            right_closest = neighs[leaf].right;
            right_ctr << CN(leaf, 0) + W[leaf], CN(leaf, 1), CN(leaf, 2);
            right_distance = abs(right_ctr[0] - CN(leaf, 0));
            right_f = f[leaf];
        } else if (neighs[leaf].right.size() > 1) {
            std::tie(right_closest, right_ctr, right_distance) = getNClosestNeighs(leaf, neighs[leaf].right, NUM_NEIGHS, CN);
            for (int n: right_closest) right_f += f[n];
            right_f /= right_closest.size();
        } else {
            right_closest = neighs[leaf].right;
            right_ctr << CN(neighs[leaf].right[0], 0), CN(leaf, 1), CN(leaf, 2);
            right_distance = abs(right_ctr[0] - CN(leaf, 0));
            right_f = f[neighs[leaf].right[0]];
        }

        std::vector<int> left_closest;
        Eigen::VectorXd left_ctr(3);
        double left_distance;
        double left_f = 0;
        if (neighs[leaf].left.size() == 0) {
            left_closest = neighs[leaf].left;
            left_ctr << CN(leaf, 0) - W[leaf], CN(leaf, 1), CN(leaf, 2);
            left_distance = abs(right_ctr[0] - CN(leaf, 0));
            left_f = f[leaf];
        } else if (neighs[leaf].left.size() > 1) {
            std::tie(left_closest, left_ctr, left_distance) = getNClosestNeighs(leaf, neighs[leaf].left, NUM_NEIGHS, CN);
            for (int n: left_closest) left_f += f[n];
            left_f /= left_closest.size();
        } else {
            left_closest = neighs[leaf].left;
            left_ctr << CN(neighs[leaf].left[0], 0), CN(leaf, 1), CN(leaf, 2);
            left_distance = abs(left_ctr[0] - CN(leaf, 0));
            left_f = f[neighs[leaf].left[0]];
        }

        std::vector<int> top_closest;
        Eigen::VectorXd top_ctr(3);
        double top_distance;
        double top_f = 0;
        if (neighs[leaf].top.size() == 0) {
            top_closest = neighs[leaf].top;
            top_ctr << CN(leaf, 0), CN(leaf, 1) + W[leaf], CN(leaf, 2);
            top_distance = abs(top_ctr[1] - CN(leaf, 1));
            top_f = f[leaf];
        } else if (neighs[leaf].top.size() > 1) {
            std::tie(top_closest, top_ctr, top_distance) = getNClosestNeighs(leaf, neighs[leaf].top, NUM_NEIGHS, CN);
            for (int n: top_closest) top_f += f[n];
            top_f /= top_closest.size();
        } else {
            top_closest = neighs[leaf].top;
            top_ctr << CN(leaf, 0), CN(neighs[leaf].top[0], 1), CN(leaf, 2);
            top_distance = abs(top_ctr[1] - CN(leaf, 1));
            top_f = f[neighs[leaf].top[0]];
        }
        
        std::vector<int> bottom_closest;
        Eigen::VectorXd bottom_ctr(3);
        double bottom_distance;
        double bottom_f = 0;
        if (neighs[leaf].bottom.size() == 0) {
            bottom_closest = neighs[leaf].bottom;
            bottom_ctr << CN(leaf, 0), CN(leaf, 1) - W[leaf], CN(leaf, 2);
            bottom_distance = abs(bottom_ctr[1] - CN(leaf, 1));
            bottom_f = f[leaf];
        } else if (neighs[leaf].bottom.size() > 1) {
            std::tie(bottom_closest, bottom_ctr, bottom_distance) = getNClosestNeighs(leaf, neighs[leaf].bottom, NUM_NEIGHS, CN);
            for (int n: bottom_closest) bottom_f += f[n];
            bottom_f /= bottom_closest.size();
        } else {
            bottom_closest = neighs[leaf].bottom;
            bottom_ctr << CN(leaf, 0), CN(neighs[leaf].bottom[0], 1), CN(leaf, 2);
            bottom_distance = abs(bottom_ctr[1] - CN(leaf, 1));
            bottom_f = f[neighs[leaf].bottom[0]];
        }
        
        std::vector<int> front_closest;
        Eigen::VectorXd front_ctr(3);
        double front_distance;
        double front_f = 0;
        if (neighs[leaf].front.size() == 0) {
            front_closest = neighs[leaf].front;
            front_ctr << CN(leaf, 0), CN(leaf, 1), CN(leaf, 2) + W[leaf];
            front_distance = abs(front_ctr[2] - CN(leaf, 2));
            front_f = f[leaf];
        } else if (neighs[leaf].front.size() > 1) {
            std::tie(front_closest, front_ctr, front_distance) = getNClosestNeighs(leaf, neighs[leaf].front, NUM_NEIGHS, CN);
            for (int n: front_closest) front_f += f[n];
            front_f /= front_closest.size();
        } else {
            front_closest = neighs[leaf].front;
            front_ctr << CN(leaf, 0), CN(leaf, 1), CN(neighs[leaf].front[0], 2);
            front_distance = abs(front_ctr[2] - CN(leaf, 2));
            front_f = f[neighs[leaf].front[0]];
        }

        std::vector<int> back_closest;
        Eigen::VectorXd back_ctr(3);
        double back_distance;
        double back_f = 0;
        if (neighs[leaf].back.size() == 0) {
            back_closest = neighs[leaf].back;
            back_ctr << CN(leaf, 0), CN(leaf, 1), CN(leaf, 2) - W[leaf];
            back_distance = abs(back_ctr[2] - CN(leaf, 2));
            back_f = f[leaf];
        } else if (neighs[leaf].back.size() > 1) {
            std::tie(back_closest, back_ctr, back_distance) = getNClosestNeighs(leaf, neighs[leaf].back, NUM_NEIGHS, CN);
            for (int n: back_closest) back_f += f[n];
            back_f /= back_closest.size();
        } else {
            back_closest = neighs[leaf].back;
            back_ctr << CN(leaf, 0), CN(leaf, 0), CN(neighs[leaf].back[0], 2);
            back_distance = abs(back_ctr[2] - CN(leaf, 2));
            back_f = f[neighs[leaf].back[0]];
        }

        // ==== 

        // dx, dy, dz
        grad_l[0] = (right_f - left_f) / (right_distance + left_distance);
        grad_l[1] = (top_f - bottom_f) / (top_distance + bottom_distance);
        grad_l[2] = (front_f - back_f) / (front_distance + back_distance);

        grad_all.row(leaf) = grad_l;

        // std::cout << (right_f - left_f) / (right_distance + left_distance) << " " << (top_f - bottom_f) / (top_distance + bottom_distance) << " " << (front_f - back_f) / (front_distance + back_distance) << std::endl;
        // std::cout << grad_all.row(leaf) << std::endl;

    }

    return grad_all;

}

Eigen::VectorXd solveFaraday(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &is_cage, Eigen::VectorXi &depths, Eigen::VectorXd &bdry_vals, Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver, std::unordered_map<int, int> &global_to_matrix_ordering) {

    int constraints_size = is_boundary.sum() + is_cage.sum() - 1;

    // build RHS
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(CN.rows() + constraints_size);
    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        if (is_boundary(leaf)) {
            int current_idx = global_to_matrix_ordering[leaf];
            RHS[CN.rows() + current_idx] = bdry_vals[leaf];
        }
    }

    Eigen::VectorXd u = solver.solve(RHS);
    if (solver.info() != Eigen::Success) {
        std::cout << "ERROR: Solve failed!" << std::endl;
        exit(-1);
    }

    Eigen::VectorXd sol = Eigen::VectorXd::Zero(CN.rows());

    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        // if (!is_boundary(leaf)) {
        //     int current_idx = global_to_matrix_ordering[leaf];
        //     sol[leaf] = u[current_idx];
        // } else {
        //     sol[leaf] = bdry_vals[leaf];
        // }
        int current_idx = global_to_matrix_ordering[leaf];
        sol[leaf] = u[current_idx];
    }

    return sol;
}

std::unordered_map<int, int> computeFaraday(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &is_cage, Eigen::VectorXi &depths, Eigen::SparseLU<Eigen::SparseMatrix<double>> &solver) {

    // bdry_vals should be CN.rows() long, but only the entries for boundary vertices will be considered
    /*
        reindex all leaf cells so that they're neatly ordered as follows
        boundary, cage, interior
    */ 

    std::unordered_map<int, int> global_to_matrix_ordering;
    int matrix_count = 0;
    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        if (is_boundary(leaf)) {
            global_to_matrix_ordering.insert({leaf, matrix_count});
            matrix_count++;
        }
    }
    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        if (is_cage(leaf)) {
            global_to_matrix_ordering.insert({leaf, matrix_count});
            matrix_count++;
        }
    }
    for (int leaf = 0; leaf < CN.rows(); leaf++) {
        if ((!is_boundary(leaf)) && (!is_cage(leaf))) {
            global_to_matrix_ordering.insert({leaf, matrix_count});
            matrix_count++;
        }
    }
    if (matrix_count != CN.rows()) {throw std::runtime_error("Not every cell was indexed");}

    std::cout << "\tDone reindexing" << std::endl;

    int constraints_size = is_boundary.sum() + is_cage.sum() - 1;
    Eigen::SparseMatrix<double> KKT(CN.rows() + constraints_size, CN.rows() + constraints_size);
    Eigen::SparseMatrix<double> L(CN.rows(), CN.rows());
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<Eigen::Triplet<double>> L_triplets;

    // build Laplacian 
    // this is the Laplacian for ALL cells
    // weighted Laplacian which does the following:
    /*
        On each of six sides, pick the k neighboring cells whose centers are closest
        and average their positions.
        Otherwise, doesn't account for "size" of neighboring cells at all
    */

    int NUM_NEIGHS = 4.;

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        double weight_sum = 0.;
        int current_idx = global_to_matrix_ordering[leaf];

        std::vector<int> right_closest;
        Eigen::VectorXd right_ctr(3);
        double right_distance;
        if (neighs[leaf].right.size() == 0) {
            right_closest = neighs[leaf].right;
            right_ctr << CN(leaf, 0) + W[leaf], CN(leaf, 1), CN(leaf, 2);
            right_distance = abs(right_ctr[0] - CN(leaf, 0));
        } else if (neighs[leaf].right.size() > 1) {
            std::tie(right_closest, right_ctr, right_distance) = getNClosestNeighs(leaf, neighs[leaf].right, NUM_NEIGHS, CN);
        } else {
            right_closest = neighs[leaf].right;
            right_ctr << CN(neighs[leaf].right[0], 0), CN(leaf, 1), CN(leaf, 2);
            right_distance = abs(right_ctr[0] - CN(leaf, 0));
        }

        std::vector<int> left_closest;
        Eigen::VectorXd left_ctr(3);
        double left_distance;
        if (neighs[leaf].left.size() == 0) {
            left_closest = neighs[leaf].left;
            left_ctr << CN(leaf, 0) - W[leaf], CN(leaf, 1), CN(leaf, 2);
            left_distance = abs(right_ctr[0] - CN(leaf, 0));
        } else if (neighs[leaf].left.size() > 1) {
            std::tie(left_closest, left_ctr, left_distance) = getNClosestNeighs(leaf, neighs[leaf].left, NUM_NEIGHS, CN);
        } else {
            left_closest = neighs[leaf].left;
            left_ctr << CN(neighs[leaf].left[0], 0), CN(leaf, 1), CN(leaf, 2);
            left_distance = abs(left_ctr[0] - CN(leaf, 0));
        }

        std::vector<int> top_closest;
        Eigen::VectorXd top_ctr(3);
        double top_distance;
        if (neighs[leaf].top.size() == 0) {
            top_closest = neighs[leaf].top;
            top_ctr << CN(leaf, 0), CN(leaf, 1) + W[leaf], CN(leaf, 2);
            top_distance = abs(top_ctr[1] - CN(leaf, 1));
        } else if (neighs[leaf].top.size() > 1) {
            std::tie(top_closest, top_ctr, top_distance) = getNClosestNeighs(leaf, neighs[leaf].top, NUM_NEIGHS, CN);
        } else {
            top_closest = neighs[leaf].top;
            top_ctr << CN(leaf, 0), CN(neighs[leaf].top[0], 1), CN(leaf, 2);
            top_distance = abs(top_ctr[1] - CN(leaf, 1));
        }
        
        std::vector<int> bottom_closest;
        Eigen::VectorXd bottom_ctr(3);
        double bottom_distance;
        if (neighs[leaf].bottom.size() == 0) {
            bottom_closest = neighs[leaf].bottom;
            bottom_ctr << CN(leaf, 0), CN(leaf, 1) - W[leaf], CN(leaf, 2);
            bottom_distance = abs(bottom_ctr[1] - CN(leaf, 1));
        } else if (neighs[leaf].bottom.size() > 1) {
            std::tie(bottom_closest, bottom_ctr, bottom_distance) = getNClosestNeighs(leaf, neighs[leaf].bottom, NUM_NEIGHS, CN);
        } else {
            bottom_closest = neighs[leaf].bottom;
            bottom_ctr << CN(leaf, 0), CN(neighs[leaf].bottom[0], 1), CN(leaf, 2);
            bottom_distance = abs(bottom_ctr[1] - CN(leaf, 1));
        }
        
        std::vector<int> front_closest;
        Eigen::VectorXd front_ctr(3);
        double front_distance;
        if (neighs[leaf].front.size() == 0) {
            front_closest = neighs[leaf].front;
            front_ctr << CN(leaf, 0), CN(leaf, 1), CN(leaf, 2) + W[leaf];
            front_distance = abs(front_ctr[2] - CN(leaf, 2));
        } else if (neighs[leaf].front.size() > 1) {
            std::tie(front_closest, front_ctr, front_distance) = getNClosestNeighs(leaf, neighs[leaf].front, NUM_NEIGHS, CN);
        } else {
            front_closest = neighs[leaf].front;
            front_ctr << CN(leaf, 0), CN(leaf, 1), CN(neighs[leaf].front[0], 2);
            front_distance = abs(front_ctr[2] - CN(leaf, 2));
        }

        std::vector<int> back_closest;
        Eigen::VectorXd back_ctr(3);
        double back_distance;
        if (neighs[leaf].back.size() == 0) {
            back_closest = neighs[leaf].back;
            back_ctr << CN(leaf, 0), CN(leaf, 1), CN(leaf, 2) - W[leaf];
            back_distance = abs(back_ctr[2] - CN(leaf, 2));
        } else if (neighs[leaf].back.size() > 1) {
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
            int other_idx = global_to_matrix_ordering[neigh];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / (left_distance * right_distance)) / right_closest.size();;
        }
        for (int neigh: left_closest) {
            double weight = (2. / ((left_distance + right_distance) * left_distance)) / left_closest.size();
            int other_idx = global_to_matrix_ordering[neigh];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / (left_distance * right_distance)) / left_closest.size();
        }

        // add top-bottom connections 
        for (int neigh: top_closest) {
            double weight = (2. / ((top_distance + bottom_distance) * top_distance)) / top_closest.size();
            int other_idx = global_to_matrix_ordering[neigh];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / (top_distance * bottom_distance)) / top_closest.size();
        }
        for (int neigh: bottom_closest) {
            double weight = (2. / ((top_distance + bottom_distance) * bottom_distance)) / bottom_closest.size();
            int other_idx = global_to_matrix_ordering[neigh];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / (top_distance * bottom_distance)) / bottom_closest.size();
        }
       
        // add top-bottom connections 
        for (int neigh: front_closest) {
            double weight = (2. / ((front_distance + back_distance) * front_distance)) / front_closest.size();
            int other_idx = global_to_matrix_ordering[neigh];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / (front_distance * back_distance)) / front_closest.size();
        }
        for (int neigh: back_closest) {
            double weight = (2. / ((front_distance + back_distance) * back_distance)) / back_closest.size();
            int other_idx = global_to_matrix_ordering[neigh];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / (front_distance * back_distance)) / back_closest.size();
        }
        
        L_triplets.push_back(Eigen::Triplet<double>(current_idx, current_idx, -weight_sum));

    }

    L.setFromTriplets(L_triplets.begin(), L_triplets.end());
    L = L * L;

    for (int k=0; k<L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
            triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
        }
    }

    std::cout << "\tAdded Laplacian triplets" << std::endl;

    for (int bdry = 0; bdry < is_boundary.sum(); bdry++) {
        triplets.push_back(Eigen::Triplet<double>(CN.rows() + bdry, bdry, 1.));
        triplets.push_back(Eigen::Triplet<double>(bdry, CN.rows() + bdry, 1.));
    }
    for (int cage = is_boundary.sum(); cage < is_boundary.sum() + is_cage.sum() - 1; cage++) {
        triplets.push_back(Eigen::Triplet<double>(CN.rows() + cage, cage, 1.));
        triplets.push_back(Eigen::Triplet<double>(CN.rows() + cage, cage + 1, -1.));
        triplets.push_back(Eigen::Triplet<double>(cage, CN.rows() + cage, 1.));
        triplets.push_back(Eigen::Triplet<double>(cage + 1, CN.rows() + cage, -1.));
    }

    std::cout << "\tAdded constraint triplets. About to set KKT matrix" << std::endl;

    // set the KKT matrix
    KKT.setFromTriplets(triplets.begin(), triplets.end());

    solver.compute(KKT);
    if (solver.info() != Eigen::Success) {
        std::cout << "ERROR: Decomposition failed!" << std::endl;
        exit(-1);
    }
    
    return global_to_matrix_ordering;

}

Eigen::VectorXd solveDirichletProblem(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &depths, Eigen::VectorXd &bdry_vals, int laplacian) {

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