import numpy as np
import math

RES = 32

for i in range(RES):
    for j in range(RES):
        theta = i * (math.pi / RES)
        phi = j * ((2 * math.pi) / RES)
        point = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])
        print(point[0], point[1], point[2], point[0], point[1], point[2])