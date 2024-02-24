import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys
from pformat import *
import cProfile
import pstats

# Parameters
class Parameters:
    def __init__(self, n, depth):
        self.n = n
        self.depth = depth
        self.delta_z = self.depth / self.n
        self.t = 300
        self.u = 1.008
        self.d = 43.2
        self.l = 0.1
        self.kw = 0.2
        self.kp = 15 * 10**(-12)
        self.Io = 30240000
        self.HI = 30*60*60*24       # value varies between 20 and 98
        self.HN = 0.0425            # value varies between 0.0425 and 0.015
        self.miu_max = 0.96
        self.a = 1 * 10**(-9)
        self.e = 0.5
        self.Nb = 5                 # [mmol nutrient/m^3] (5-100)



class Solution:
    def __init__(self,n):
        self.z = np.zeros(n)
        self.p_values = []              # [cells/m^3]
        self.n_values = []              # [mmol nutrient/m^3]      
        self.I_values = []              # [mmol photons/m^2 day]
        self.I_wo_attenuation = []
        self.g_values = []
        self.maxI = []
        self.maxg = []
        self.maxN = []
        self.maxP = []


RESOLUTION = 0.01


# Compute the light intensity
def light(param: Parameters ,p):

    di = np.cumsum((param.kw + param.kp*p)*param.delta_z) - 1/2 * param.delta_z * (param.kw + param.kp*p)
    I = param.Io * np.exp(-di)

    return I


def growth(p: Parameters, n, I_values):

    minimum = np.array([min(n[i]/(p.HN + n[i]), I_values[i]/(p.HI + I_values[i])) for i in range(len(n))])
    miu = p.miu_max * minimum

    return miu


def solve_ode (parameters: Parameters, sol: Solution, times):
    
    # Parameters
    n = parameters.n
    depth = parameters.depth
    delta_z = parameters.delta_z
    u = parameters.u                    # [m/day]
    d = parameters.d                    # [m/day]
    a = parameters.a
    l = parameters.l
    Nb = parameters.Nb


    # Defining the grid
    z = np.linspace(delta_z/2, depth - delta_z/2, n)
    sol.z = z

    # Defining y [P(0:n) N(n:2n)]
    y = np.zeros(2*n)
    y[:n] = 10
    y[n:] = Nb

    # Derivative function
    def dydt(y,t):

        # Define and calculate advective and diffusive fluxes
        Ja = np.zeros(n+1)
        Jd = np.zeros(n+1)
        JN = np.zeros(n+1)
        for i in range(1, n):
            Ja[i] = u * y[i-1]
            Jd[i] = -d * (y[i] - y[i-1]) / delta_z
            JN[i] = -d * (y[n+i] - y[n+i-1]) / delta_z
        
        # Set boundary conditions
        Ja[0] = Ja[n] = Jd[0] = Jd[n] = JN[0] = 0
        JN[n] = -d * (Nb - y[n+i-1]) / delta_z
        J = Ja + Jd

        # Calculate functional response to light
        I = light(parameters,y[0:n])
        g = growth(parameters, y[n:], I)

        # Calculate derivative of plankton concentration
        dP_dt = np.zeros(n)
        dN_dt = np.zeros(n)
        dy_dt = np.zeros(2*n)

        for i in range(0, n-1):
            dJ_dz = (J[i+1] - J[i]) / delta_z
            dJN_dz = (JN[i+1] - JN[i]) / delta_z
            dP_dt[i] = g[i] * y[i] -l * y[i] - dJ_dz
            dN_dt[i] = - a * g[i] * y[i] + a * l * y[i] - dJN_dz
            
        dy_dt[:n] = dP_dt
        dy_dt[n:] = dN_dt
        #print(dy_dt)

        return dy_dt

    # Solving ODE for p
    solution = odeint(dydt, y, times)
    sol.p_values = solution[0][:, :n].copy()
    sol.n_values = solution[0][:, n:].copy()
    
    for i in range(len(times)):
        sol.maxP.append(find_max(sol.p_values[i]))
        sol.maxN.append(find_max(sol.n_values[i]))

def find_max(vec):
    res = 0
    for i in range(len(vec)):
        if vec[i]>res:
            res = vec[i]
    return res


def compute_light_growth(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))
    g_values = np.zeros((len(times), param.n))

    for i, time in enumerate(times):
        p = np.array(sol.p_values[i])
        I_values[i, :] = light(param, p)
        g_values[i, :] = growth(param, p, I_values[i, :])
        sol.I_values.append(I_values[i, :].copy())
        sol.g_values.append(g_values[i, :].copy())

        sol.maxI.append(find_max(I_values[i, :]))
        sol.maxg.append(find_max(g_values[i, :]))


def compute_light(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))

    for i, time in enumerate(times):
        p = np.zeros(param.n)
        I_values[i, :] = light(param, p)
        sol.I_wo_attenuation.append(I_values[i, :].copy())


def show_result(sol: Solution, t: int, prints: str):

    if prints == "none":
        return

    # Plotting Plankton and Nutrients Concentration 
    # "plankton"
    if prints == "plankton" or prints == "all":
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_PLANKTON)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_PLANKTON)

        plt.subplot(1, 2, 2)
        plt.imshow(sol.n_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_NUTRIENTS)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_NUTRIENTS)
        plt.show()

    # Plotting LIMITATION of Growth by Light Intensity and Nutrients Concentration 
    # "limitation"
    if prints == "limitation" or prints == "all":
        _, ax1 = plt.subplots(figsize=(12, 6))

        # Plot Nutrients on the left y-axis
        ax1.plot(sol.n_values[-1, :] / sol.maxN[-1], sol.z, label=L_NUTRIENTS_N, color='#006572', linewidth=3)
        ax1.plot(np.array(sol.I_values)[-1, :] / sol.maxI[-1], sol.z, label=L_LIGHT_N, color='#FFA500', linewidth=3)
        ax1.plot(sol.p_values[-1, :] / sol.maxP[-1], sol.z, label=L_PLANKTON_N, color='green', linewidth=3)

        ax1.set_xlabel(AX_NTR_LGT_PP)
        ax1.set_ylabel(AX_DEPTH)
        ax1.invert_yaxis()

        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        ax1.minorticks_on()

        plt.yticks(np.linspace(100, 0, 11))
        plt.legend()
        plt.title(T_NTR_LGT_PP)
        plt.show()

    # Plotting TIMESERIES of Light Intensity, Growth Rate, Plankton and Nutrients Concentration 
    # "timeseries"
    if prints == "timeseries" or prints == "all":
        plt.figure(figsize=(12, 8))

        # Plotting Light Intensity
        plt.subplot(3, 1, 1)
        plt.imshow(np.array(sol.I_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_LIGHT)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_LIGHT)

        # Plotting Growth Rate
        plt.subplot(3, 1, 2)
        plt.imshow(np.array(sol.g_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_GROWTH)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_GROWTH)

        # Plotting Plankton Concentration
        plt.subplot(3, 1, 3)
        plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_PLANKTON)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_PLANKTON)

        plt.tight_layout()
        plt.show()

    # Plotting STABLE values of plankton etc with respect to depth at the last instant
    # "stable"
    if prints == "stable" or prints == "all":
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(sol.p_values[-1].T, sol.z, color='#3E9A4C', linewidth=3)
        plt.xlabel(L_PLANKTON)
        plt.ylabel(AX_DEPTH)
        plt.title(T_PLANKTON)
        plt.gca().invert_yaxis()
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(sol.I_values[-1].T, sol.z, color='#FFA500', linewidth=3, label=L_W_ATTENUATION)
        plt.plot(sol.I_wo_attenuation[-1].T, sol.z, color='#A15800', linewidth=2, linestyle='--', label=L_WO_ATTENUATION)
        plt.xlabel(L_LIGHT)
        plt.ylabel(AX_DEPTH)
        plt.title(T_LIGHT)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(sol.g_values[-1].T, sol.z, color='#006572', linewidth=3)
        plt.xlabel(L_GROWTH)
        plt.ylabel(AX_DEPTH)
        plt.title(T_GROWTH)
        plt.gca().invert_yaxis()
        plt.grid()

        plt.tight_layout()
        plt.show()



def main():

    if len(sys.argv) != 4:
        print("Usage: python week2.py <n> <depth> <prints>")
        sys.exit(1)

    n = int(sys.argv[1])
    depth = int(sys.argv[2])
    prints = sys.argv[3]

    # To run without arguments:
    # n = 50
    # depth = 100
    # prints = "plankton"
    
    parameters = Parameters(n, depth)
    sol = Solution(n)

    # Solve ODE for plankton concentration
    times = np.arange(0, parameters.t+1, RESOLUTION)
    solve_ode(parameters, sol, times)

    # Compute light intensity (w/ and wo/ attenuation) and growth rate
    compute_light_growth(times, parameters, sol)
    compute_light(times,parameters,sol)

    show_result(sol, parameters.t, prints)
    
    
if __name__ == "__main__":
    main()



