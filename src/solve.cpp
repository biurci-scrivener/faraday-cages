#include "solve.h"

size_t NUM_NEIGHS_GRAD = 16;
size_t NUM_NEIGHS_LAPLACE = 4;


Eigen::MatrixXd grad2(   Eigen::VectorXd &W_all, Eigen::MatrixXi &CH,  std::vector<struct CellNeighbors> &neighs, 
                        Eigen::VectorXi &parents, std::unordered_map<int, int> &all_to_leaf, std::unordered_map<int, int> &leaf_to_all, 
                        Eigen::VectorXd &f) {

    Eigen::MatrixXd grad_all(neighs.size(), 3);
    
    for (size_t leaf = 0; leaf < neighs.size(); leaf++) {

        Eigen::VectorXd grad_l = Eigen::VectorXd::Zero(3);

        double right_f = getFunctionValueAtNeighbor(leaf, neighs[leaf].right, W_all, CH, parents, all_to_leaf, leaf_to_all, f);
        double left_f = getFunctionValueAtNeighbor(leaf, neighs[leaf].left, W_all, CH, parents, all_to_leaf, leaf_to_all, f);
        double top_f = getFunctionValueAtNeighbor(leaf, neighs[leaf].top, W_all, CH, parents, all_to_leaf, leaf_to_all, f);
        double bottom_f = getFunctionValueAtNeighbor(leaf, neighs[leaf].bottom, W_all, CH, parents, all_to_leaf, leaf_to_all, f);
        double front_f = getFunctionValueAtNeighbor(leaf, neighs[leaf].front, W_all, CH, parents, all_to_leaf, leaf_to_all, f);
        double back_f = getFunctionValueAtNeighbor(leaf, neighs[leaf].back, W_all, CH, parents, all_to_leaf, leaf_to_all, f);

        // dx, dy, dz
        grad_l[0] = (right_f - left_f) / (2 * W_all[leaf_to_all[leaf]]);
        grad_l[1] = (top_f - bottom_f) / (2 * W_all[leaf_to_all[leaf]]);
        grad_l[2] = (front_f - back_f) / (2 * W_all[leaf_to_all[leaf]]);

        grad_all.row(leaf) = grad_l;

        // std::cout << (right_f - left_f) / (right_distance + left_distance) << " " << (top_f - bottom_f) / (top_distance + bottom_distance) << " " << (front_f - back_f) / (front_distance + back_distance) << std::endl;
        // std::cout << grad_all.row(leaf) << std::endl;

    }

    return grad_all;

}

Eigen::MatrixXd grad(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<struct CellNeighbors> &neighs, Eigen::VectorXi &is_boundary, Eigen::VectorXi &depths, Eigen::VectorXd &bdry_vals, Eigen::VectorXd &f) {

    Eigen::MatrixXd grad_all(CN.rows(), 3);
    
    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        Eigen::VectorXd grad_l = Eigen::VectorXd::Zero(3);
        
        Eigen::VectorXd right_ctr(3);
        double right_distance;
        double right_f = 0;
        std::tie(right_ctr, right_distance, right_f) = getNeighRep(leaf, neighs[leaf].right, NUM_NEIGHS_GRAD, CN, W, 0, f);

        Eigen::VectorXd left_ctr(3);
        double left_distance;
        double left_f = 0;
        std::tie(left_ctr, left_distance, left_f) = getNeighRep(leaf, neighs[leaf].left, NUM_NEIGHS_GRAD, CN, W, 1, f);

        Eigen::VectorXd top_ctr(3);
        double top_distance;
        double top_f = 0;
        std::tie(top_ctr, top_distance, top_f) = getNeighRep(leaf, neighs[leaf].top, NUM_NEIGHS_GRAD, CN, W, 2, f);

        Eigen::VectorXd bottom_ctr(3);
        double bottom_distance;
        double bottom_f = 0;
        std::tie(bottom_ctr, bottom_distance, bottom_f) = getNeighRep(leaf, neighs[leaf].bottom, NUM_NEIGHS_GRAD, CN, W, 3, f);

        Eigen::VectorXd front_ctr(3);
        double front_distance;
        double front_f = 0;
        std::tie(front_ctr, front_distance, front_f) = getNeighRep(leaf, neighs[leaf].front, NUM_NEIGHS_GRAD, CN, W, 4, f);

        Eigen::VectorXd back_ctr(3);
        double back_distance;
        double back_f = 0;
        std::tie(back_ctr, back_distance, back_f) = getNeighRep(leaf, neighs[leaf].back, NUM_NEIGHS_GRAD, CN, W, 5, f);

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

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        double weight_sum = 0.;
        int current_idx = global_to_matrix_ordering[leaf];

        Eigen::VectorXd right_ctr(3);
        double right_distance;
        std::tie(right_ctr, right_distance) = getNeighRep(leaf, neighs[leaf].right, NUM_NEIGHS_LAPLACE, CN, W, 0);

        Eigen::VectorXd left_ctr(3);
        double left_distance;
        std::tie(left_ctr, left_distance) = getNeighRep(leaf, neighs[leaf].left, NUM_NEIGHS_LAPLACE, CN, W, 1);

        Eigen::VectorXd top_ctr(3);
        double top_distance;
        std::tie(top_ctr, top_distance) = getNeighRep(leaf, neighs[leaf].top, NUM_NEIGHS_LAPLACE, CN, W, 2);

        Eigen::VectorXd bottom_ctr(3);
        double bottom_distance;
        std::tie(bottom_ctr, bottom_distance) = getNeighRep(leaf, neighs[leaf].bottom, NUM_NEIGHS_LAPLACE, CN, W, 3);

        Eigen::VectorXd front_ctr(3);
        double front_distance;
        std::tie(front_ctr, front_distance) = getNeighRep(leaf, neighs[leaf].front, NUM_NEIGHS_LAPLACE, CN, W, 4);

        Eigen::VectorXd back_ctr(3);
        double back_distance;
        std::tie(back_ctr, back_distance) = getNeighRep(leaf, neighs[leaf].back, NUM_NEIGHS_LAPLACE, CN, W, 5);

        // std::cout << right_distance << " " << left_distance << " " << top_distance << " " << bottom_distance << " " << front_distance << " " << back_distance << " " << std::endl;


        // ==== 

        // add right-left connections 
        double across_1 = left_distance + right_distance;
        double across_2 = left_distance * right_distance;
        for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].right.size()); i++) {
            double weight = (2. / (across_1 * right_distance)) / neighs[leaf].right.size();
            int other_idx = global_to_matrix_ordering[neighs[leaf].right[i]];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / across_2) / neighs[leaf].right.size();
        }
        for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].left.size()); i++) {
            double weight = (2. / (across_1 * left_distance)) / neighs[leaf].left.size();
            int other_idx = global_to_matrix_ordering[neighs[leaf].left[i]];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / across_2) / neighs[leaf].left.size();
        }

        // add top-bottom connections 
        across_1 = top_distance + bottom_distance;
        across_2 = top_distance * bottom_distance;
        for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].top.size()); i++) {
            double weight = (2. / (across_1 * top_distance)) / neighs[leaf].top.size();
            int other_idx = global_to_matrix_ordering[neighs[leaf].top[i]];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / across_2) / neighs[leaf].top.size();
        }
        for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].bottom.size()); i++) {
            double weight = (2. / (across_1 * bottom_distance)) / neighs[leaf].bottom.size();
            int other_idx = global_to_matrix_ordering[neighs[leaf].bottom[i]];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / across_2) / neighs[leaf].bottom.size();
        }

        // add front-back connections 
        across_1 = front_distance + back_distance;
        across_2 = front_distance * back_distance;
        for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].front.size()); i++) {
            double weight = (2. / (across_1 * front_distance)) / neighs[leaf].front.size();
            int other_idx = global_to_matrix_ordering[neighs[leaf].front[i]];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / across_2) / neighs[leaf].front.size();
        }
        for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].back.size()); i++) {
            double weight = (2. / (across_1 * back_distance)) / neighs[leaf].back.size();
            int other_idx = global_to_matrix_ordering[neighs[leaf].back[i]];
            L_triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
            weight_sum += (1. / across_2) / neighs[leaf].back.size();
        }

        if ((leaf == 11415) || (leaf == 9335)) {
            std::cout << std::endl << "Leaf: " << leaf << std::endl;
            std::cout << "Right distance: " << right_distance << std::endl;
            std::cout << "Right center: " << right_ctr.transpose() << std::endl;
            std::cout << "Right neighbors: ";
            for (int n: neighs[leaf].right) std::cout << n << " ";
            std::cout << std::endl;
            std::cout << "Left distance: " << left_distance << std::endl;
             std::cout << "Left center: " << left_ctr.transpose() << std::endl;
            std::cout << "Left neighbors: ";
            for (int n: neighs[leaf].left) std::cout << n << " ";
            std::cout << std::endl;
            std::cout << "Top distance: " << top_distance << std::endl;
             std::cout << "Top center: " << top_ctr.transpose() << std::endl;
            std::cout << "Top neighbors: ";
            for (int n: neighs[leaf].top) std::cout << n << " ";
            std::cout << std::endl;
            std::cout << "Bottom distance: " << bottom_distance << std::endl;
             std::cout << "Bottom center: " << bottom_ctr.transpose() << std::endl;
            std::cout << "Bottom neighbors: ";
            for (int n: neighs[leaf].bottom) std::cout << n << " ";
            std::cout << std::endl;
            std::cout << "Front distance: " << front_distance << std::endl;
             std::cout << "Front center: " << front_ctr.transpose() << std::endl;
            std::cout << "Front neighbors: ";
            for (int n: neighs[leaf].front) std::cout << n << " ";
            std::cout << std::endl;
            std::cout << "Back distance: " << back_distance << std::endl;
             std::cout << "Back center: " << back_ctr.transpose() << std::endl;
            std::cout << "Back neighbors: ";
            for (int n: neighs[leaf].back) std::cout << n << " ";
            std::cout << std::endl << std::endl;
        }

        L_triplets.push_back(Eigen::Triplet<double>(current_idx, current_idx, -weight_sum));

    }

    L.setFromTriplets(L_triplets.begin(), L_triplets.end());
    // std::cout << L << std::endl;
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

    std::cout << "\tSet KKT matrix. Decomposing, this may a take a while..." << std::endl;

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

        for (int leaf = 0; leaf < CN.rows(); leaf++) {
            if (!is_boundary(leaf)) {
                double weight_sum = 0.;
                double b_sum = 0.;
                int current_idx = global_to_interior[leaf];

                Eigen::VectorXd right_ctr(3);
                double right_distance;
                std::tie(right_ctr, right_distance) = getNeighRep(leaf, neighs[leaf].right, NUM_NEIGHS_LAPLACE, CN, W, 0);

                Eigen::VectorXd left_ctr(3);
                double left_distance;
                std::tie(left_ctr, left_distance) = getNeighRep(leaf, neighs[leaf].left, NUM_NEIGHS_LAPLACE, CN, W, 1);

                Eigen::VectorXd top_ctr(3);
                double top_distance;
                std::tie(top_ctr, top_distance) = getNeighRep(leaf, neighs[leaf].top, NUM_NEIGHS_LAPLACE, CN, W, 2);

                Eigen::VectorXd bottom_ctr(3);
                double bottom_distance;
                std::tie(bottom_ctr, bottom_distance) = getNeighRep(leaf, neighs[leaf].bottom, NUM_NEIGHS_LAPLACE, CN, W, 3);

                Eigen::VectorXd front_ctr(3);
                double front_distance;
                std::tie(front_ctr, front_distance) = getNeighRep(leaf, neighs[leaf].front, NUM_NEIGHS_LAPLACE, CN, W, 4);

                Eigen::VectorXd back_ctr(3);
                double back_distance;
                std::tie(back_ctr, back_distance) = getNeighRep(leaf, neighs[leaf].back, NUM_NEIGHS_LAPLACE, CN, W, 5);

                // ==== 

                // add right-left connections 
                double across_1 = left_distance + right_distance;
                double across_2 = left_distance * right_distance;
                for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].right.size()); i++) {
                    double weight = (2. / (across_1 * right_distance)) / neighs[leaf].right.size();
                    if (!is_boundary(neighs[leaf].right[i])) {
                        int other_idx = global_to_interior[neighs[leaf].right[i]];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neighs[leaf].right[i]];
                    }
                }
                for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].left.size()); i++) {
                    double weight = (2. / (across_1 * left_distance)) / neighs[leaf].left.size();
                    if (!is_boundary(neighs[leaf].left[i])) {
                        int other_idx = global_to_interior[neighs[leaf].left[i]];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neighs[leaf].left[i]];
                    }
                }
                weight_sum += (2. / across_2);

                // add top-bottom connections 
                across_1 = top_distance + bottom_distance;
                across_2 = top_distance * bottom_distance;
                for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].top.size()); i++) {
                    double weight = (2. / (across_1 * top_distance)) / neighs[leaf].top.size();
                    if (!is_boundary(neighs[leaf].top[i])) {
                        int other_idx = global_to_interior[neighs[leaf].top[i]];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neighs[leaf].top[i]];
                    }
                }
                for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].bottom.size()); i++) {
                    double weight = (2. / (across_1 * bottom_distance)) / neighs[leaf].bottom.size();
                    if (!is_boundary(neighs[leaf].bottom[i])) {
                        int other_idx = global_to_interior[neighs[leaf].bottom[i]];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neighs[leaf].bottom[i]];
                    }
                }
                weight_sum += (2. / across_2);

                // add front-back connections 
                across_1 = front_distance + back_distance;
                across_2 = front_distance * back_distance;
                for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].front.size()); i++) {
                    double weight = (2. / (across_1 * front_distance)) / neighs[leaf].front.size();
                    if (!is_boundary(neighs[leaf].front[i])) {
                        int other_idx = global_to_interior[neighs[leaf].front[i]];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neighs[leaf].front[i]];
                    }
                }
                for (size_t i= 0; i < std::min((size_t)NUM_NEIGHS_LAPLACE, neighs[leaf].back.size()); i++) {
                    double weight = (2. / (across_1 * back_distance)) / neighs[leaf].back.size();
                    if (!is_boundary(neighs[leaf].back[i])) {
                        int other_idx = global_to_interior[neighs[leaf].back[i]];
                        triplets.push_back(Eigen::Triplet<double>(current_idx, other_idx, weight));
                    } else {
                        b_sum += weight * bdry_vals[neighs[leaf].back[i]];
                    }
                }
                weight_sum += (2. / across_2);

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