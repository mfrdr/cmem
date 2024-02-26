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
        self.t = 1
        self.u = 1.008
        self.d = 5*60*60*24         # [m^2 /(x10^5) /day] Av in article
        self.kp = 15 * 10**(-12)
        self.m = 0.03               # [/day] e in article, natural mortality losses
        self.gamma = 0.7            # [m^3 /mmolN /day] grazing coeficient
        self.tau = 0.1              # [/day] remineralization coeficient
        self.kw = 0.0375            # [/m] attenuation coefficient of water
        self.kc = 0.05              # [m^2 /mmolN] self-shading of the organic material
        self.Io = 300               # [W /m^2]
        #self.HI = 30*60*60*24      # value varies between 20 and 98
        self.k_N = 0.3              # [mmolN /m^3] half-saturation constant (previous HN)
        self.miu = 0.8              # [/day] maximum specific growth rate
        self.a = 0.1                # [m^2 /W /day] slope of the PI-curve
        self.e = 1.0                # recycling coeficient (previous paper)
        self.Nb = 30                # [mmolN /m^3]
        self.w = 15                 # [m /day]



class Solution:
    def __init__(self,n):
        self.z = np.zeros(n)
        self.p_values = []              # [cells/m^3]
        self.n_values = []              # [mmol nutrient/m^3]  
        self.d_values = []    
        self.I_values = []              # [mmol photons/m^2 day]
        self.I_wo_attenuation = []
        self.g_values = []
        self.maxI = []
        self.maxg = []
        self.maxN = []
        self.maxP = []
        self.maxD = []


# Compute the light intensity
def light(param: Parameters, p, d, z):

    di = np.cumsum((param.kw + param.kc*(p+d))*param.delta_z)
    I = param.Io * np.exp(-di)

    return I


def growth(p: Parameters, n, I_values):

    sigma_N = np.array([n[i] / (p.k_N+n[i]) for i in range(len(n))])
    sigma_L = np.array([p.a*I_values[i] / np.sqrt(p.miu**2 + (p.a*I_values[i])**2) for i in range(len(I_values))])

    g = p.miu * sigma_N * sigma_L
    #print(sigma_L, sigma_N)

    return g


def solve_ode (parameters: Parameters, sol: Solution, times):
    
    # Parameters
    n = parameters.n
    depth = parameters.depth
    delta_z = parameters.delta_z
    u = parameters.u                    # [m/day]
    d = parameters.d                    # [m/day]
    m = parameters.m
    Nb = parameters.Nb
    w = parameters.w
    gamma = parameters.gamma
    tau = parameters.tau


    # Defining the grid
    z = np.linspace(delta_z/2, depth - delta_z/2, n)
    sol.z = z

    # Defining y [P(0:n) N(n:2n)]
    y = np.zeros(3*n)
    y[:n] = 10
    y[n:2*n] = Nb
    y[2*n:] = 0

    # Derivative function
    def dydt(y,t):

        # Define and calculate advective and diffusive fluxes
        Ja = np.zeros(n+1)
        Jd = np.zeros(n+1)
        JN = np.zeros(n+1)
        JD = np.zeros(n+1)
        for i in range(1, n):
            Ja[i] = u * y[i-1]
            Jd[i] = -d * (y[i] - y[i-1]) / delta_z
            JN[i] = -d * (y[n+i] - y[n+i-1]) / delta_z
            JD[i] = -d * (y[2*n+i] - y[2*n+i-1]) / delta_z
        
        # Set boundary conditions
        Ja[0] = Ja[n] = Jd[0] = Jd[n] = JN[0] = JD[0] = 0
        JN[n] = -d * (Nb - y[n+i-1]) / delta_z
        JD[n] = - w * y[-1]
        J = Ja + Jd

        # Calculate functional response to light
        I = light(parameters,y[0:n],y[2*n:],z)
        g = growth(parameters, y[n:2*n], I)

        # Calculate derivative of plankton concentration
        dP_dt = np.zeros(n)
        dN_dt = np.zeros(n)
        dD_dt = np.zeros(n)
        dy_dt = np.zeros(3*n)

        for i in range(0, n-1):
            dJ_dz = (J[i+1] - J[i]) / delta_z
            dJN_dz = (JN[i+1] - JN[i]) / delta_z
            dJD_dz = (JD[i+1] - JD[i]) / delta_z
            dD_dz = (y[2*n+i+1] - y[2*n+i]) / delta_z

            dP_dt[i] = g[i]*y[i] - m*y[i] - gamma*y[i]**2 - dJ_dz
            # print(g[i]*y[i], m*y[i], gamma*y[i]**2, dJ_dz)
            dN_dt[i] = - g[i]*y[i] + tau*y[2*n+i] - dJN_dz
            dD_dt[i] = m*y[i] + gamma*y[i]**2 - tau*y[2*n+i] - w*dD_dz - dJD_dz
            
        dy_dt[:n] = dP_dt
        dy_dt[n:2*n] = dN_dt
        dy_dt[2*n:] = dD_dt

        return dy_dt

    # Solving ODE for p
    solution = odeint(dydt, y, times)
    sol.p_values = solution[0][:, :n].copy()
    sol.n_values = solution[0][:, n:2*n].copy()
    sol.d_values = solution[0][:,2*n:].copy()
    
    for i in range(len(times)):
        sol.maxP.append(find_max(sol.p_values[i]))
        sol.maxN.append(find_max(sol.n_values[i]))
        sol.maxD.append(find_max(sol.d_values[i]))

def find_max(vec):
    res = 0
    for i in range(len(vec)):
        if vec[i]>res:
            res = vec[i]
    return res


def compute_light_growth(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))
    g_values = np.zeros((len(times), param.n))
    z = sol.z

    for i, time in enumerate(times):
        p = np.array(sol.p_values[i])
        d = np.array(sol.p_values[i])
        I_values[i, :] = light(param, p, d, z)
        g_values[i, :] = growth(param, p, I_values[i, :])
        sol.I_values.append(I_values[i, :].copy())
        sol.g_values.append(g_values[i, :].copy())

        sol.maxI.append(find_max(I_values[i, :]))
        sol.maxg.append(find_max(g_values[i, :]))


def compute_light(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))
    z = sol.z

    for i, time in enumerate(times):
        p = np.zeros(param.n)
        d = np.array(sol.p_values[i])
        I_values[i, :] = light(param, p, d, z)
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
        plt.subplot(2, 3, 1)
        plt.imshow(np.array(sol.I_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_LIGHT)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_LIGHT)

        # Plotting Growth Rate
        plt.subplot(2, 3, 2)
        plt.imshow(np.array(sol.g_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_GROWTH)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_GROWTH)

        # Plotting Plankton Concentration
        plt.subplot(2, 3, 4)
        plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_PLANKTON)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_PLANKTON)

        # Plotting Nutrient Concentration
        plt.subplot(2, 3, 5)
        plt.imshow(sol.n_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_NUTRIENTS)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_NUTRIENTS)

        # Plotting Detritus Concentration
        plt.subplot(2, 3, 6)
        plt.imshow(sol.d_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_DETRITUS)
        plt.xlabel(AX_TIME_D)
        plt.ylabel(AX_DEPTH)
        plt.title(T_DETRITUS)

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



