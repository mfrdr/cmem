import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys

# Parameters
class Parameters:
    def __init__(self, n, depth):
        self.n = n
        self.depth = depth
        self.deltaZ = self.depth / self.n
        self.t = 400
        self.u = 0.96
        self.d = 43.2
        self.l = 0.24
        self.kw = 0.2
        self.kp = 15 * 10**(-12)
        self.Io = 30240000
        self.pmax = 0.96
        self.H = 2592000

class Solution:
    def __init__(self,n,t):
        self.z = np.zeros(n)
        self.p_values = []
        self.I_values = []
        self.I_wo_attenuation = []
        self.g_values = []

RESOLUTION = 0.01


# Compute the light intensity
def light(param: Parameters ,p):

    dI = np.cumsum((param.kw + param.kp*p)*param.deltaZ) - 1/2 * param.deltaZ * (param.kw + param.kp*p)
    I = param.Io * np.exp(-dI)

    return I

def growth(param: Parameters, I_values):

    pI = (param.pmax * I_values) / (param.H + I_values)
    g = pI - param.l

    return g


def solve_ode (parameters: Parameters, sol: Solution, times):
    
    # Parameters
    n = parameters.n
    depth = parameters.depth
    deltaZ = parameters.deltaZ
    u = parameters.u                    # [m/day]
    d = parameters.d                    # [m/day]

    # Defining the grid
    z = np.linspace(deltaZ/2, depth - deltaZ/2, n)
    sol.z = z

    # Defining p
    p = np.zeros(n)
    p[7] = 1000

    # Derivative function
    def ode(p, t):
        # Define and calculate advective and diffusive fluxes
        Ja = np.zeros(n+1)
        Jd = np.zeros(n+1)
        for i in range(1, n):
            Ja[i] = u * p[i-1]
            Jd[i] = -d * (p[i] - p[i-1]) / deltaZ

        # Set boundary fluxes
        Ja[0] = Ja[n] = Jd[0] = Jd[n] = 0

        # Calculate total flux
        J = Ja + Jd

        # Calculate functional response to light
        I = light(parameters,p)
        g = growth(parameters, I)

        # Calculate derivative of plankton concentration
        dp_dt = np.zeros(n)
        for i in range(0, n):
            dJ_dz = (J[i+1] - J[i]) / deltaZ
            dp_dt[i] = g[i] * p[i] - dJ_dz

        return dp_dt

    # Solving ODE for p
    p_solution = odeint(ode, p, times)
    sol.p_values = p_solution.copy()

    return 


def compute_light_growth(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))
    g_values = np.zeros((len(times), param.n))

    for i, time in enumerate(times):
        p = np.array(sol.p_values[i])
        I_values[i, :] = light(param, p)
        g_values[i, :] = growth(param, I_values[i, :])
        sol.I_values.append(I_values[i, :].copy())
        sol.g_values.append(g_values[i, :].copy())
    
    return


def compute_light(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))

    for i, time in enumerate(times):
        p = np.zeros(param.n)
        I_values[i, :] = light(param, p)
        sol.I_wo_attenuation.append(I_values[i, :].copy())

    return


def show_result(sol: Solution, t: int, prints: str):
    times = np.arange(0, t+1, RESOLUTION)

    if prints == "none":
        return

    if prints != "stable" and prints != "timeseries":
        # Plotting Plankton Concentration
        plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label='Plankton Concentration [umol/m^3]')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Plankton Concentration Over Time and Depth')
        plt.show()

    if prints != "stable" and prints!= "plankton":

        # Plotting TIMESERIES of Light Intensity, Growth Rate, and Plankton Concentration
        plt.figure(figsize=(12, 8))

        # Plotting Light Intensity
        plt.subplot(3, 1, 1)
        plt.imshow(np.array(sol.I_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label='Light Intensity [umol photons /m^2 /day]')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Light Intensity Over Time and Depth')

        # Plotting Growth Rate
        plt.subplot(3, 1, 2)
        plt.imshow(np.array(sol.g_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label='Growth Rate [/day]')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Growth Rate Over Time and Depth')

        # Plotting Plankton Concentration
        plt.subplot(3, 1, 3)
        plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label='Plankton Concentration [umol/m^3]')
        plt.xlabel('Time [days]')
        plt.ylabel('Depth [m]')
        plt.title('Plankton Concentration Over Time and Depth')

        plt.tight_layout()
        plt.show()

    if prints != "timeseries" and prints != "plankton":

        # Plotting STABLE values of phi with respect to depth at the last instant
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(sol.p_values[-1].T, sol.z, color='#3E9A4C', linewidth=3)
        plt.xlabel('Stable Plankton Concentration [umol/m^3]')
        plt.ylabel('Depth [m]')
        plt.title('Stable Plankton Concentration at t = {}'.format(int(times[-1])))
        plt.gca().invert_yaxis()
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(sol.I_values[-1].T, sol.z, color='#FFA500', linewidth=3, label='With Attenuation')
        plt.plot(sol.I_wo_attenuation[-1].T, sol.z, color='#A15800', linewidth=2, linestyle='--', label='Without Attenuation')
        plt.xlabel('Stable Light Intensity [umol photons /m^2 /day]')
        plt.ylabel('Depth [m]')
        plt.title('Stable Light Intensity at t = {}'.format(int(times[-1])))
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(sol.g_values[-1].T, sol.z, color='#006572', linewidth=3)
        plt.xlabel('Stable Growth Rate [/day]')
        plt.ylabel('Depth [m]')
        plt.title('Stable Growth Rate at t = {}'.format(int(times[-1])))
        plt.gca().invert_yaxis()
        plt.grid()

        plt.tight_layout()
        plt.show()

    return

def main():

    if len(sys.argv) != 4:
        print("Usage: python week2.py <n> <depth> <prints>")
        sys.exit(1)

    n = int(sys.argv[1])
    depth = int(sys.argv[2])
    prints = sys.argv[3]

    parameters = Parameters(n, depth)
    sol = Solution(n,parameters.t)

    # Solve ODE for plankton concentration
    times = np.arange(0, parameters.t+1, RESOLUTION)
    solve_ode(parameters, sol, times)

    # Compute light intensity (w/ and wo/ attenuation) and growth rate
    compute_light_growth(times, parameters, sol)
    compute_light(times,parameters,sol)

    show_result(sol, parameters.t, prints)

    return

if __name__ == "__main__":
    main()



