import igl
import tetgen.pytetgen
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import tetgen
import ply_io
import scipy.sparse.linalg as sp

MARGIN = 0.5

def box_mesh(V):
    """
    
    Place a "box" around this mesh, to create a prismatic domain for tetgen

    V: vertices
    F: faces

    out: V, F

    """

    BV, BF = igl.bounding_box(V)
    b_min, b_max = np.apply_along_axis(np.min, 0, BV), np.apply_along_axis(np.max, 0, BV)
    padding = np.min(b_max - b_min) * MARGIN
    BV, BF = igl.bounding_box(V, padding)
    BF += V.shape[0]
    return (np.vstack((V, BV)), BF, padding)


def callback():
    pass

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Args: path")
        exit()
    path = sys.argv[1]

    ps.init()

    ps.set_user_callback(callback)

    P, N = ply_io.parse_ply(path)

    BV, BF, padding = box_mesh(P)

    tetra = tetgen.TetGen(BV, BF)
    VT, TT = tetra.tetrahedralize(switches=f'-q -a{pow(padding / 2, 3)}')

    A = igl.cotmatrix(VT, TT)
    b = np.ones(VT.shape[0])
    x = sp.spsolve(A, b)
    print(x)
    print(A @ x)


    ps_vol = ps.register_volume_mesh("Vol. mesh", VT, tets=TT)
    ps_vol.add_scalar_quantity("f", x, enabled=True)

    ps.show()