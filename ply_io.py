import numpy as np

def parse_ply(filename):
    with open(filename, 'r') as file:
        if file is None:
            print(f"Error reading file {filename}")
            exit(1)
            
        vertex_count = 0

        for line in file:
            line = line.strip()
            if line == "end_header":
                break
            
            if line.startswith("element vertex"):
                parts = line.split()
                vertex_count = int(parts[2])

        V = np.zeros((vertex_count, 3))
        N = np.zeros((vertex_count, 3))

        for i in range(vertex_count):
            parts = file.readline().strip().split()
            V[i, :] = [float(parts[0]), float(parts[1]), float(parts[2])]

            if len(parts) > 3:
                N[i, :] = [float(parts[3]), float(parts[4]), float(parts[5])]

    return V, N
