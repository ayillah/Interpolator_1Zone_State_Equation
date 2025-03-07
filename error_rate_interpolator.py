import numpy as np
import matplotlib.pyplot as plt

def error_rate(Nx, error):
    """Compute the error rate."""

    # Compute the logs of the nodes and the errors
    logNx = np.log(np.array(Nx))
    logError = np.log(np.array(error))
    ones = np.ones(len(logNx))

    V = np.array([ones, logNx]).transpose()

    # Solve least squares system
    A = np.matmul(V.transpose(), V)
    b = np.matmul(V.transpose(), logError)

    c = np.linalg.solve(A, b)

    return c[1]


if __name__ == '__main__':

    nodes = [32, 64, 128, 256, 512, 1024]
    error_norms = [20.050327097535273, 3.0547157308742783, 
                    0.3794896260968695, 0.04703965508687144, 
                    0.005863577566214543, 0.0007330279038187867]

    p = error_rate(nodes, error_norms)

    print('Error rate = {:6f} '.format(p))

    plt.loglog(nodes, error_norms, 'r-o')
    plt.xlabel('log(Nx)')
    plt.ylabel('log(error_norms)')
    plt.grid()
    plt.show()