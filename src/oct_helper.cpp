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

std::vector<struct CellNeighbors> createOctreeNeighbors(Eigen::MatrixXd &CN, Eigen::VectorXd &W, std::vector<Eigen::Vector3d> &oc_pts, std::vector<Eigen::Vector2i> &oc_edges) {
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
        
    }

    return neighs;
}