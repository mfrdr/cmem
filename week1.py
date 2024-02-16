import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math as m

# Parameters
class param:
    def __init__(self):
        self.n = 1
        self.depth = 0
        self.deltaZ = self.depth/self.n
        self.u = 0
        self.D = 0


def phi_conc (n,depth):
    
    deltaZ = depth / n
    t = 200
    u = 1
    d = 1

    # Defining the grid with reversed order
    z = np.linspace(deltaZ/2, depth - deltaZ/2, n)

    # Defining phi
    phi = np.zeros(n)
    phi[1] = 10

    # Derivative function
    def ode_phi(phi, t):

        # Define and calculate advective and diffusive fluxes
        Ja = np.zeros(n+1)
        Jd = np.zeros(n+1)
        for i in range(1, n):
            Ja[i] = u * phi[i-1]
            Jd[i] = -d * (phi[i] - phi[i-1]) / deltaZ
        
        # Set boudary fluxes
        Ja[0] = Ja[n] = Jd[0] = Jd[n] = 0

        # Calculate total flux
        J = Ja + Jd

        # Calculate derivative
        dphi_dt = np.zeros(n)
        for i in range(n):
            dphi_dt[i] = -(J[i+1] - J[i]) / deltaZ

        return dphi_dt

    # Solving ODE
    times = np.arange(0, t+1, 1)
    phi_solution = odeint(ode_phi, phi, times)

    # Plotting 
    plt.imshow(phi_solution.T, aspect='auto', cmap='viridis', extent=[0, t, z[-1], z[0]])
    plt.colorbar(label='Concentration')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (z)')
    plt.title('Concentration of Phytoplankton')
    plt.show()

    return

def main():

    n = 50
    depth = 100

    phi_conc(n,depth)

    return

if __name__ == "__main__":
    main()
