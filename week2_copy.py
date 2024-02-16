import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys

# Parameters
class param:
    def __init__(self):
        self.n = 1
        self.depth = 0
        self.deltaZ = self.depth/self.n
        self.u = 0
        self.D = 0


def show_results (prints, phi_solution, I_values, growth_values, t, z, times):

    if prints != "stable" and prints != "timeseries":
        # Plotting Plankton Concentration
        plt.imshow(phi_solution.T, aspect='auto', cmap='viridis', extent=[0, t, z[-1], z[0]])
        plt.colorbar(label='Plankton Concentration [umol/m^3]')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Plankton Concentration Over Time and Depth')
        plt.show()

    if prints != "stable" and prints!= "plankton":
        # Plotting Light Intensity, Growth Rate, and Plankton Concentration
        plt.figure(figsize=(12, 8))

        # Plotting Light Intensity
        plt.subplot(3, 1, 1)
        plt.imshow(I_values.T, aspect='auto', cmap='viridis', extent=[0, t, z[-1], z[0]])
        plt.colorbar(label='Light Intensity (I)')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Light Intensity Over Time and Depth')

        # Plotting Growth Rate
        plt.subplot(3, 1, 2)
        plt.imshow(growth_values.T, aspect='auto', cmap='viridis', extent=[0, t, z[-1], z[0]])
        plt.colorbar(label='Growth Rate')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Growth Rate Over Time and Depth')

        # Plotting Plankton Concentration
        plt.subplot(3, 1, 3)
        plt.imshow(phi_solution.T, aspect='auto', cmap='viridis', extent=[0, t, z[-1], z[0]])
        plt.colorbar(label='Plankton Concentration')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Plankton Concentration Over Time and Depth')

        plt.tight_layout()
        plt.show()

    if prints != "timeseries" and prints != "plankton":
        # Plotting stable values of phi with respect to depth at the last instant
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(phi_solution[-1, :], z, color='green', linewidth=3)
        plt.xlabel('Stable Plankton Concentration [umol/m^3]')
        plt.ylabel('Depth [m]')
        plt.title('Stable Plankton Concentration at t = {}'.format(int(times[-1])))
        plt.gca().invert_yaxis()

        plt.subplot(2, 2, 2)
        plt.plot(I_values[-1, :], z, color='orange', linewidth=3)
        plt.xlabel('Stable Light Intensity [units??]')
        plt.ylabel('Depth [m]')
        plt.title('Stable Light Intensity at t = {}'.format(int(times[-1])))
        plt.gca().invert_yaxis()

        plt.subplot(2, 2, 3)
        plt.plot(growth_values[-1, :], z, color='turquoise', linewidth=3)
        plt.xlabel('Stable Growth Rate [units??]')
        plt.ylabel('Depth [m]')
        plt.title('Stable Growth Rate at t = {}'.format(int(times[-1])))
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()
    
    return


def phi_conc (n,depth,prints):
    
    # Parameters
    deltaZ = depth / n
    t = 400                    # [days]
    u = 0.96                    # [m/day]
    d = 43.2                    # [m/day]
    l = 0.24                    # [/day]
    kw = 0.2                    # [/m]
    kp = 15 * 10**(-12)         # [m^2/cell]
    Io = 30240000               # [umol photons /m^2 /day]
    pmax = 0.96                 # [/day]
    H = 2592000                 # [umol photons /m^2 /day]

    # Defining the grid
    z = np.linspace(deltaZ/2, depth - deltaZ/2, n)

    # Defining phi
    phi = np.zeros(n)
    phi[7] = 1000
    
    # Derivative function
    def ode_phi(phi, t):
        # Define and calculate advective and diffusive fluxes
        Ja = np.zeros(n+1)
        Jd = np.zeros(n+1)
        for i in range(1, n):
            Ja[i] = u * phi[i-1]
            Jd[i] = -d * (phi[i] - phi[i-1]) / deltaZ

        # Set boundary fluxes
        Ja[0] = Ja[n] = Jd[0] = Jd[n] = 0

        # Calculate total flux
        J = Ja + Jd

        # Calculate functional response to light
        #p = np.array(phi)
        dI = np.cumsum((kw + kp*phi)*deltaZ) - 1/2 * deltaZ * (kw + kp*phi)
        I = Io * np.exp(-dI)
        pI = (pmax * I) / (H + I)
        g = pI - l

        # Calculate derivative of plankton concentration
        dphi_dt = np.zeros(n)
        dJ_dz = np.zeros(n)
        for i in range(0, n):
            dJ_dz = (J[i+1] - J[i]) / deltaZ
            dphi_dt[i] = g[i] * phi[i] - dJ_dz

        return dphi_dt

    # Solving ODE for phi
    times = np.arange(0, t+1, 0.01)
    phi_solution = odeint(ode_phi, phi, times)    

    def plot_light_intensity_growth_and_phi(n, prints):
        I_values = np.zeros((len(times), n))
        growth_values = np.zeros((len(times), n))
        for i, time in enumerate(times):
            p = np.array(phi_solution[i])
            dI = np.cumsum((kw + kp*p)*deltaZ) - 1/2*deltaZ*(kw + kp*p)
            I_values[i, :] = Io * np.exp(-dI)

            # Calculate growth rate as a function of light
            pI = (pmax * I_values[i, :]) / (H + I_values[i, :])
            growth_values[i, :] = pI - l
        
        # Plots 
        show_results(prints, phi_solution, I_values, growth_values, t, z, times)

    plot_light_intensity_growth_and_phi(n,prints)
    return

def main():

    if len(sys.argv) != 4:
        print("Usage: python week2.py <n> <depth> <prints>")
        sys.exit(1)
    
    n = int(sys.argv[1])
    depth = int(sys.argv[2])
    prints = sys.argv[3]

    # n = 20
    # depth = 100    
        
    phi_conc(n,depth,prints)

    return

if __name__ == "__main__":
    main()



