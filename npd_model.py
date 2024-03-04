import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pformat import *
import seaborn as sns


# Parameters
class Parameters:
    def __init__(self, n, depth,t):
        self.n = n
        self.depth = depth
        self.delta_z = self.depth / self.n
        self.t = t
        self.u = 1
        self.d = 5                  # [m^2 /(x10^5) /day] Av in article
        self.m = 0.03               # [/day] e in article, natural mortality losses
        self.gamma = 1.5            # [m^3 /mmolN /day] grazing coeficient
        self.tau = 0.1              # [/day] remineralization coeficient
        self.kw = 0.0375            # [/m] attenuation coefficient of water
        self.kc = 0.05              # [m^2 /mmolN] self-shading of the organic material
        self.Io = 250               # [W /m^2] 
        self.k_N = 0.3              # [mmolN /m^3] half-saturation constant (previous HN)
        self.mu = 0.8               # [/day] maximum specific growth rate
        self.a = 0.1                # [m^2 /W /day] slope of the PI-curve
        self.Nb = 25                # [mmolN /m^3]
        self.w = 15                 # [m /day]
        self.sigmaN = []
        self.sigmaL = []



class Solution:
    def __init__(self,n):
        self.n = 0
        self.z = np.zeros(n)
        self.p_values = []              # [cells/m^3]
        self.n_values = []              # [mmol nutrient/m^3]  
        self.d_values = []    
        self.I_values = []              # [mmol photons/m^2 day]
        self.I_wo_attenuation = []
        self.g_values = []
        
        self.maxP = []
        self.sigmaN = []
        self.sigmaL = []


# Compute the light intensity
def light(param: Parameters, p, d):

    di = np.cumsum((param.kw + param.kc*(p+d))*param.delta_z) - (param.kw + param.kc*(p+d))*param.delta_z/2
    I = param.Io * np.exp(-di)

    return I


def growth(p: Parameters, n, I_values):

    p.sigmaN = np.array([n[i]/(p.k_N+n[i]) for i in range(len(n))])
    p.sigmaL = np.array([p.a*I_values[i] / np.sqrt(p.mu**2 + (p.a*I_values[i])**2) for i in range(len(I_values))])

    g = p.mu * p.sigmaN * p.sigmaL

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

    # Defining y [P(0:n) N(n:2n) D(2n:3n)]
    y = np.zeros(3*n)
    y[5] = 0.2                #P0 [mmolN /m^3]
    y[n:2*n] = Nb           #N0 [mmolN /m^3]
    y[3] = 0.03                #D0 [mmolN /m^3]

    # Derivative function
    def dydt(y,t):

        P = y[:n]
        N = y[n:2*n]
        D = y[2*n:]

        # Define and calculate advective and diffusive fluxes
        Ja = np.zeros(n+1)
        Jd = np.zeros(n+1)
        JN = np.zeros(n+1)
        JD = np.zeros(n+1)

        Ja[1:n] = u * P[:n-1]
        Jd[1:n] = -d * (P[1:n] - P[:n-1]) / delta_z
        JN[1:n] = -d * (N[1:n] - N[:n-1]) / delta_z
        JD[1:n] = -d * (D[1:n] - D[:n-1]) / delta_z + w * D[:n-1]
        
        # Set boundary conditions
        Ja[0] = Ja[n] = Jd[0] = Jd[n] = JN[0] = JD[0] = 0
        JN[n] = -d * (Nb - N[n-1]) / delta_z
        JD[n] = w * D[-1]
        JP = Ja + Jd

        # Calculate functional response to light
        I = light(parameters,P,D)
        g = growth(parameters,N,I)

        # Calculate derivative of plankton concentration
        # dP_dt = np.zeros(n)
        # dN_dt = np.zeros(n)
        # dD_dt = np.zeros(n)
        dy_dt = np.zeros(3*n)

        dJP_dz = np.diff(JP) / delta_z
        dJN_dz = np.diff(JN) / delta_z
        dJD_dz = np.diff(JD) / delta_z

        dP_dt = g[:n]*P[:n] - m*P[:n] - gamma*P[:n]**2 - dJP_dz
        dN_dt = -g[:n]*P[:n] + tau*D[:n] - dJN_dz
        dD_dt = m*P[:n] + gamma*P[:n]**2 - tau*D[:n] - dJD_dz
            
        dy_dt[:n] = dP_dt
        dy_dt[n:2*n] = dN_dt
        dy_dt[2*n:] = dD_dt

        return dy_dt

    # Solving ODE for p
    solution = odeint(dydt, y, times)
    sol.p_values = solution[0][:,:n].copy()
    sol.n_values = solution[0][:,n:2*n].copy()
    sol.d_values = solution[0][:,2*n:].copy()
    
    #for i in range(len(times)):
    sol.maxP=(np.max(sol.p_values[-1]))
    sol.maxN=(np.max(sol.n_values[-1]))
    sol.maxD=(np.max(sol.d_values[-1]))

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
        P = np.array(sol.p_values[i])
        N = np.array(sol.n_values[i])
        D = np.array(sol.d_values[i])
        I_values[i,:] = light(param, P, D)
        g_values[i,:] = growth(param, N, I_values[i,:])

        sol.I_values.append(I_values[i,:].copy())
        sol.g_values.append(g_values[i,:].copy())

        sol.sigmaL.append(param.sigmaL.copy())
        sol.sigmaN.append(param.sigmaN.copy())


def compute_light(times, param: Parameters, sol: Solution):
    I_values = np.zeros((len(times), param.n))

    for i, time in enumerate(times):
        P = np.zeros(param.n)
        D = np.zeros(param.n)
        I_values[i, :] = light(param, P, D)
        sol.I_wo_attenuation.append(I_values[i, :].copy())


def show_result(sol: Solution, t: int, prints: str):

    if prints == "none":
        return

    # Plotting Plankton and Nutrients Concentration 
    # "plankton"
    # if prints == "plankton" or prints == "all":
    #     plt.figure(figsize=(12, 6))

    #     plt.subplot(1, 3, 1)
    #     plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
    #     plt.colorbar(label=L_PLANKTON)
    #     plt.xlabel(AX_TIME_D)
    #     plt.ylabel(AX_DEPTH)
    #     plt.title(T_PLANKTON)

    #     plt.subplot(1, 3, 2)
    #     plt.imshow(sol.n_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
    #     plt.colorbar(label=L_NUTRIENTS)
    #     plt.xlabel(AX_TIME_D)
    #     plt.ylabel(AX_DEPTH)
    #     plt.title(T_NUTRIENTS)

    #     plt.subplot(1, 3, 3)
    #     plt.imshow(sol.d_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
    #     plt.colorbar(label=L_DETRITUS)
    #     plt.xlabel(AX_TIME_D)
    #     plt.ylabel(AX_DEPTH)
    #     plt.title(T_DETRITUS)
    #     # plt.show()

    # Plotting LIMITATION of Growth by Light Intensity and Nutrients Concentration 
    # "limitation"
    if prints == "limitation" or prints == "all":
        _, ax1 = plt.subplots(figsize=(12,8))

        ax1.plot(1000*sol.p_values[-1], sol.z, label=L_P, color='green', linewidth=2)
        ax1.set_ylabel(AX_DEPTH,fontsize=18)
        ax1.set_xlabel(AX_MUMOL,fontsize=18, ha='left',x=0.9, y=0.5)
        ax1.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_facecolor('#f0f0f0')
        ax1.tick_params(axis='both', which='both', labelsize=18)
        ax1.invert_yaxis()

        secax_top = ax1.twiny()
        secax_top.plot(sol.sigmaN[-1], sol.z, label=L_N, color='blue', linewidth=2,linestyle='--',alpha=0.7)
        secax_top.plot(sol.sigmaL[-1], sol.z, label=L_I, color='orange', linewidth=2,linestyle='--',alpha=0.7)
        secax_top.set_xlim(0,1)  # Set the same limits as the main x-axis
        secax_top.tick_params(axis='both', which='both', labelsize=18)
        # ax.set_xlim(0, 30)  
        ax1.set_ylim(0, 300)

        line2, = secax_top.plot(0, 0, color='orange', linewidth=2, label=L_I,alpha=0.6)
        line3, = secax_top.plot(0, 0, color='blue', linewidth=2, label=L_N,alpha=0.6)
        line1, = ax1.plot(0, 0, color='green', linewidth=2, label=L_P)
        secax_top.invert_yaxis()
        ax1.legend([line1, line2, line3], [L_P, L_I, L_N], loc='best')

        
        plt.title(T_LIMIT,fontsize=20,fontweight='bold')
        plt.savefig("limitation.png")
        

    # Plotting TIMESERIES of Light Intensity, Growth Rate, Plankton and Nutrients Concentration 
    # "timeseries"
    if prints == "timeseries" or prints == "all":
        plt.figure(figsize=(12, 12))

        # Plotting Light Intensity
        plt.subplot(3, 2, 1)
        plt.imshow(np.array(sol.I_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_LIGHT)
        plt.xlabel(AX_TIME_D,fontsize=12)
        plt.ylabel(AX_DEPTH,fontsize=12)
        plt.title(T_LIGHT,fontsize=12,fontweight='bold')

        # Plotting Growth Rate
        plt.subplot(3, 2, 2)
        plt.imshow(np.array(sol.g_values).T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_GROWTH)
        plt.xlabel(AX_TIME_D,fontsize=12)
        plt.ylabel(AX_DEPTH,fontsize=12)
        plt.title(T_GROWTH,fontsize=12,fontweight='bold')

        # Plotting Plankton Concentration
        plt.subplot(3, 2, 3)
        plt.imshow(sol.p_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_PLANKTON)
        plt.xlabel(AX_TIME_D,fontsize=12)
        plt.ylabel(AX_DEPTH,fontsize=12)
        plt.title(T_PLANKTON,fontsize=12,fontweight='bold')

        # Plotting Nutrient Concentration
        plt.subplot(3, 2, 4)
        plt.imshow(sol.n_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_NUTRIENTS)
        plt.xlabel(AX_TIME_D,fontsize=12)
        plt.ylabel(AX_DEPTH,fontsize=12)
        plt.title(T_NUTRIENTS,fontsize=12,fontweight='bold')

        # Plotting Detritus Concentration
        plt.subplot(3, 2, 5)
        plt.imshow(sol.d_values.T, aspect='auto', cmap='viridis', extent=[0, t, sol.z[-1], sol.z[0]])
        plt.colorbar(label=L_DETRITUS)
        plt.xlabel(AX_TIME_D,fontsize=12)
        plt.ylabel(AX_DEPTH,fontsize=12)
        plt.title(T_DETRITUS,fontsize=12,fontweight='bold')

        plt.tight_layout()
        plt.savefig("timeseries.png")
        # plt.show()

    # Plotting STABLE values of plankton etc with respect to depth at the last instant
    # "stable"
    if prints == "stable" or prints == "all":

        _, ax = plt.subplots(figsize=(10, 10))

        ax.plot(sol.n_values[-1].T, sol.z, color='blue', linewidth=2,alpha=0.6)
        ax.set_ylabel(AX_DEPTH,fontsize=18)
        ax.set_xlabel(AX_MMOL,fontsize=18, ha='left',x=0.9, y=0.5)
        plt.gca().set_facecolor('#f0f0f0')  # Light gray background
        ax.grid(True, linestyle='--', alpha=0.6)
        secax_top = ax.twiny()
        secax_top.plot(1000*sol.d_values[-1].T, sol.z, color='brown', linewidth=2,alpha=0.6)
        secax_top.plot(sol.I_values[-1].T, sol.z, color='orange', linewidth=2,alpha=0.6)
        secax_top.plot(1000*sol.p_values[-1].T, sol.z, color='green', linewidth=2)
        secax_top.set_xlim(ax.get_xlim())  # Set the same limits as the main x-axis
        secax_top.set_xlabel(AX_MUMOL,fontsize=18, ha='left',x=0.9, y=0.5)
        ax.set_xlim(0, 30)  
        ax.set_ylim(0, 300)  
        secax_top.set_xlim(0, 250)
        ax.invert_yaxis()

        line2, = secax_top.plot(0, 0, color='brown', linewidth=2, label=L_D,alpha=0.6)
        line3, = ax.plot(0, 0, color='blue', linewidth=2, label=L_N,alpha=0.6)
        line4, = secax_top.plot(0, 0, color='orange', linewidth=2, label=L_I,alpha=0.6)
        line1, = secax_top.plot(0, 0, color='green', linewidth=2, label=L_P)
        ax.legend([line1, line2, line3, line4], [L_P, L_D, L_N, L_I], loc='right')
        ax.set_title(T_STEADY,fontsize=20,fontweight='bold')

        ax.tick_params(axis='both', which='both', labelsize=18)
        secax_top.tick_params(axis='both', which='both', labelsize=18)

        plt.savefig("stable.png")

    if prints == "grid":

        _, ax0 = plt.subplots(figsize=(8, 8))
        sol.maxP = [0.4994015803715579, 0.5005495263756956, 0.5009721084258041, 0.5011279500290862, 0.5011986918233604, 0.501265287546247]
        sol.n = [ 75, 150, 225, 300, 375, 450]
        ax0.plot(sol.n, sol.maxP, color='green', linewidth=2, label=L_P)
        ax0.set_ylabel(AX_PMAX,fontsize=18)
        ax0.set_xlabel(AX_N,fontsize=18)
        ax0.set_ylim(0, 1)
        plt.gca().set_facecolor('#f0f0f0')  # Light gray background
        ax0.grid(True, linestyle='--', alpha=0.6)
        ax0.set_title(T_GRID,fontsize=20,fontweight='bold')
        ax0.tick_params(axis='both', which='both', labelsize=18)

        plt.savefig("grid.png")


def print_sens_profile(p_values, target: str, t_array, depth):

    green = sns.color_palette("Greens", 5)
    _, ax = plt.subplots(figsize=(6, 12))
    for i, values in enumerate(p_values):
        ax.plot(1000*values, depth, label=rf'$m$ = {t_array[i]}', color=green[i], linewidth=1)

    ax.set_ylabel(AX_DEPTH,fontsize=14)
    ax.set_xlabel(L_PLANKTON,fontsize=14)
    ax.tick_params(axis='both', which='both', labelsize=14)
    ax.set_ylim(0, 300)
    ax.invert_yaxis()
    plt.gca().set_facecolor('#f0f0f0')  # Light gray background
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(r'$m$',fontsize=20,fontweight='bold')
    plt.legend()
    plt.savefig(f"sens_profile_{target}.png")

    


def main():

    # To run with arguments:
    # if len(sys.argv) != 4:
    #     print("Usage: python week2.py <n> <depth> <prints>")
    #     sys.exit(1)
    # n = int(sys.argv[1])
    # depth = int(sys.argv[2])
    # prints = sys.argv[3]

    # To run without arguments:
    depth = 300
    prints = "sensitivity"
    t = 6000
    n = 150

    if prints == "sensitivity":
        parameters = Parameters(n, depth,t)
        sol = Solution(n)

        # parameter to assess: d
        ten_u = parameters.m/10
        temp_u = np.round(np.linspace(parameters.m-2*ten_u,parameters.m+2*ten_u,5),decimals=2)

        p_values_ = []
        for i in range(len(temp_u)):
            print("Going for this value...")
            parameters.m = temp_u[i]
            sol = Solution(n)

            times = np.arange(0, parameters.t+1, RESOLUTION)
            print("Solving...")
            solve_ode(parameters, sol, times)
            print("Solved!")

            p_values_.append(sol.p_values[-1])

        print_sens_profile(p_values_, "m", temp_u, sol.z)
        plt.show()
        print("There you go :)")
        return


    elif prints == "grid":
        n_val = np.linspace(75,450,6)
        n_val = n_val.astype(int)

        maxP = []
        for i in range(len(n_val)):
            print("Going for this grid size...")
            n = n_val[i]
            parameters = Parameters(n, depth,t)
            sol = Solution(n)

            times = np.arange(0, parameters.t+1, RESOLUTION)
            print("Solving...")
            solve_ode(parameters, sol, times)
            print("Solved!")
            maxP.append(np.max(sol.p_values))

        sol.maxP = maxP
        sol.n = n_val

    else:
        parameters = Parameters(n, depth,t)
        sol = Solution(n)

        # Solve ODE for plankton concentration
        times = np.arange(0, parameters.t+1, RESOLUTION)
        print("Solving...")
        solve_ode(parameters, sol, times)
        print("Solved!")
        
        # Compute light intensity (w/ and wo/ attenuation) and growth rate
        print("Computing light and growth...")
        compute_light_growth(times, parameters, sol)
        compute_light(times,parameters,sol)

    print("Preparing results...")
    show_result(sol, parameters.t, prints)
    print("There you go :)")

    plt.show()
    
    
if __name__ == "__main__":
    main()



