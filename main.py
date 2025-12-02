import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==========================================
# 1. USER INPUTS & CONFIGURATION
# ==========================================
# Choose Algorithm: 'Simplex' or 'Secant'
# Note: 'Secant' here uses L-BFGS-B (a Quasi-Newton method using secant updates)
# as it is standard for parameter estimation with bounds.
ALGORITHM_CHOICE = 'Simplex' 

# Fixed Inputs
XH_INPUT = 2500.0  # Biomass COD (mg/L) - Input as requested
YH = 0.67          # Yield Coefficient (mgCOD/mgCOD)
fE = 0.08          # Endogenous Residue Fraction

# Initial Guesses for Parameters (from Table 5 or estimation)
# Order: [bh, kh, Ks, Kx, mumax, Ss0, Xs0]
# Units: 1/day for rates, mg/L for constants
initial_guess = [0.08, 1.62, 17.30, 0.10, 2.68, 222.94, 752.70]

# Parameter Bounds ((min, max)) to prevent unrealistic values
param_bounds = (
    (0.0, 1.0),      # bh (1/day)
    (0.0, 10.0),     # kh (1/day)
    (0.1, 100.0),    # Ks (mg/L)
    (0.0, 5.0),      # Kx (mg/L)
    (0.1, 20.0),     # mumax (1/day)
    (10.0, 1000.0),  # Ss0 (mg/L)
    (100.0, 5000.0)  # Xs0 (mg/L)
)

# ==========================================
# 2. DATA INPUT
# ==========================================
# 60 Data points representing 1 hour (0 to 1.0 h) derived from Table 1 image
our_data_real = np.array([
    54.0, 55.2, 56.0, 55.6, 44.6, 32.0, 29.0, 26.0, 25.0, 25.5, 25.0, 25.2,
    25.1, 24.2, 24.7, 24.0, 23.7, 23.8, 23.0, 22.5, 22.7, 21.5, 20.6, 19.6,
    19.6, 19.5, 18.1, 17.5, 15.6, 15.2, 13.2, 12.0, 10.0, 9.1, 8.3, 6.6,
    6.1, 5.5, 5.5, 5.2, 5.1, 5.0, 4.6, 4.9, 5.1, 4.4, 4.5, 4.4,
    4.3, 4.3, 4.3, 4.2, 4.1, 4.0, 3.9, 3.9, 3.8, 3.8, 3.7, 3.6
])

# Time vector (0 to 1 hour)
t_meas = np.linspace(0, 1.0, len(our_data_real))

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
def asm_model_rates(y, t, params):
    """
    Calculates derivatives [dSs/dt, dXs/dt, dXh/dt]
    """
    Ss, Xs, Xh = y
    bh, kh, Ks, Kx, mumax = params
    
    # Convert 1/day rates to 1/hour for the ODE solver
    bh_h = bh / 24.0
    kh_h = kh / 24.0
    mumax_h = mumax / 24.0
    
    # Process Rates (rho)
    # Growth: Monod kinetics. S_O term assumed ~1 (non-limiting)
    rho_growth = mumax_h * (Ss / (Ks + Ss)) * Xh
    
    # Decay: First order
    rho_decay = bh_h * Xh
    
    # Hydrolysis: Surface limitation kinetics
    if Xh > 1e-6:
        ratio = Xs / Xh
    else:
        ratio = 0
    rho_hydro = kh_h * (ratio / (Kx + ratio)) * Xh
    
    # Stoichiometry (from Matrix)
    # dSs/dt: Growth consumes (-1/YH), Hydrolysis produces (+1)
    dSs_dt = - (1.0 / YH) * rho_growth + rho_hydro
    
    # dXs/dt: Hydrolysis consumes (-1)
    dXs_dt = - rho_hydro
    
    # dXh/dt: Growth produces (+1), Decay consumes (-1)
    dXh_dt = rho_growth - rho_decay
    
    return [dSs_dt, dXs_dt, dXh_dt]

def calculate_model_our(t_points, y0, params):
    """
    Integrates ODE and calculates OUR profile
    """
    # Unpack kinetic parameters
    kinetic_params = params[:5]
    
    # Integrate ODE
    sol = odeint(asm_model_rates, y0, t_points, args=(kinetic_params,))
    Ss = sol[:, 0]
    Xs = sol[:, 1]
    Xh = sol[:, 2]
    
    # Recalculate Rates for OUR calculation
    bh, kh, Ks, Kx, mumax = kinetic_params
    bh_h = bh / 24.0
    mumax_h = mumax / 24.0
    
    rho_growth = mumax_h * (Ss / (Ks + Ss)) * Xh
    rho_decay = bh_h * Xh
    
    # OUR Calculation (mgO2/L/h)
    # OUR = Oxygen for Growth + Oxygen for Decay
    our_calc = ((1.0 - YH) / YH) * rho_growth + (1.0 - fE) * rho_decay
    
    return our_calc

# ==========================================
# 4. OPTIMIZATION
# ==========================================
def objective_function(p):
    """
    Objective: Minimize Sum of Squared Errors (SSE)
    p = [bh, kh, Ks, Kx, mumax, Ss0, Xs0]
    """
    # Hard constraint check (if algorithm doesn't support bounds well)
    if any(x < 0 for x in p): return 1e9
    
    params_kinetic = p[:5]
    Ss0, Xs0 = p[5], p[6]
    Xh0 = XH_INPUT # Fixed
    
    y0 = [Ss0, Xs0, Xh0]
    
    try:
        our_pred = calculate_model_our(t_meas, y0, params_kinetic)
        sse = np.sum((our_data_real - our_pred)**2)
        return sse
    except:
        return 1e9

print(f"--- Starting Parameter Estimation ({ALGORITHM_CHOICE}) ---")

if ALGORITHM_CHOICE == 'Simplex':
    # Nelder-Mead (Simplex)
    res = minimize(objective_function, initial_guess, method='Nelder-Mead', tol=1e-4)
elif ALGORITHM_CHOICE == 'Secant':
    # L-BFGS-B (Quasi-Newton/Secant with bounds)
    res = minimize(objective_function, initial_guess, method='L-BFGS-B', bounds=param_bounds, tol=1e-4)
else:
    print("Invalid Algorithm. Defaulting to Simplex.")
    res = minimize(objective_function, initial_guess, method='Nelder-Mead', tol=1e-4)

# ==========================================
# 5. OUTPUT & PLOTTING
# ==========================================
p_opt = res.x
print(f"Optimization Success: {res.success}")
print(f"Final SSE: {res.fun:.4f}")
print("\n--- Estimated Parameters ---")
labels = ["bh (1/d)", "kh (1/d)", "Ks (mg/L)", "Kx (mg/L)", "mumax (1/d)", "Ss0 (mg/L)", "Xs0 (mg/L)"]
for label, value in zip(labels, p_opt):
    print(f"{label}: {value:.4f}")

# Calculate Best Fit Curve
y0_opt = [p_opt[5], p_opt[6], XH_INPUT]
our_fit_curve = calculate_model_our(t_meas, y0_opt, p_opt[:5])

# Plotting
plt.figure(figsize=(10, 6))
# Plot Real Data
plt.scatter(t_meas, our_data_real, facecolors='none', edgecolors='blue', s=80, label='Measurement', zorder=2)
# Plot Model Fit
plt.plot(t_meas, our_fit_curve, 'k-', linewidth=2.5, label='Model', zorder=1)

# Formatting similar to Figure 4
plt.title('Model fitting results', fontsize=14)
plt.xlabel('Time(h)', fontsize=12)
plt.ylabel('mgO2/l/h', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='-')
plt.xlim(-0.02, 1.02)

# Save to Desktop (Adjust path if needed or just save in current dir)
import os
save_path = os.path.join(os.getcwd(), 'model_fit_results.png')
plt.savefig(save_path, dpi=300)
print(f"\nGraph saved to: {save_path}")
plt.show()