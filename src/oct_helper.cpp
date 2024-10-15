#include "oct_helper.h"

// int onSide(int leaf, int other, Eigen::MatrixXd &CN, Eigen::VectorXd &W) {
//     // returns 0, 1, 2, 3, 4, 5 for directions, -1 if not on any side
//     bool inXLimits = (CN(other, 0) > (CN(leaf, 0) - W[leaf] / 2.)) && (CN(other, 0) < (CN(leaf, 0) + W[leaf] / 2.));
//     bool inYLimits = (CN(other, 1) > (CN(leaf, 1) - W[leaf] / 2.)) && (CN(other, 1) < (CN(leaf, 1) + W[leaf] / 2.));
//     bool inZLimits = (CN(other, 2) > (CN(leaf, 2) - W[leaf] / 2.)) && (CN(other, 2) < (CN(leaf, 2) + W[leaf] / 2.));
//     bool onRight = (CN(other, 0) > CN(leaf, 0));
//     bool onLeft = (CN(other, 0) < CN(leaf, 0));
//     bool onTop = (CN(other, 1) > CN(leaf, 1));
//     bool onBottom = (CN(other, 1) < CN(leaf, 1));
//     bool onFront = (CN(other, 2) > CN(leaf, 2));
//     bool onBack = (CN(other, 2) < CN(leaf, 2));
    
//     if (inYLimits && inZLimits && onRight) {
//         return 0;
//     } else if (inYLimits && inZLimits && onLeft) {
//         return 1;
//     } else if (inXLimits && inZLimits && onTop) {
//         return 2;
//     } else if (inXLimits && inZLimits && onBottom) {
//         return 3;
//     } else if (inXLimits && inYLimits && onFront) {
//         return 4;
//     } else if (inXLimits && inYLimits && onBack) {
//         return 5;
//     } else {
//         return -1;
//     }
// }

Eigen::VectorXi onSide(int leaf, int other, Eigen::MatrixXd &CN, Eigen::VectorXd &W) {

    Eigen::VectorXi sides = Eigen::VectorXi::Zero(6);
    
    if (CN(other, 0) > CN(leaf, 0)) sides[0] = 1; 
    if (CN(other, 0) < CN(leaf, 0)) sides[1] = 1;
    if (CN(other, 1) > CN(leaf, 1)) sides[2] = 1;
    if (CN(other, 1) < CN(leaf, 1)) sides[3] = 1;
    if (CN(other, 2) > CN(leaf, 2)) sides[4] = 1;
    if (CN(other, 2) < CN(leaf, 2)) sides[5] = 1;

    return sides;
}



std::vector<struct CellNeighbors> createOctreeNeighbors(Eigen::MatrixXd &CN, Eigen::VectorXd &W) {
    /*
    
        Takes CN_l and W_l (computes neighbor relationships for leaves only)

    */
    
    std::vector<struct CellNeighbors> neighs(CN.rows()); 

    for (int leaf = 0; leaf < CN.rows(); leaf++) {

        for (int other = 0; other < CN.rows(); other++) {

            if (other == leaf) continue;
            Eigen::VectorXi sides = onSide(leaf, other, CN, W);
            
            if (sides[0]) {
                neighs[leaf].right.push_back(other);
            }
            
            if (sides[1]) {
                neighs[leaf].left.push_back(other);
            }

            if (sides[2]) {
                neighs[leaf].top.push_back(other);
            }

            if (sides[3]) {
                neighs[leaf].bottom.push_back(other);
            }

            if (sides[4]) {
                neighs[leaf].front.push_back(other);
            }

            if (sides[5]) {
                neighs[leaf].back.push_back(other);
            }

        }
        
    }

    return neighs;
}