import numpy as np
from scipy.optimize import minimize

# --- FORMULE ---
def nelson_siegel_spot(t, b0, b1, b2, lam):
    """Calcola tassi spot NS (vettorizzato)"""
    term1 = np.ones_like(t)
    mask = t > 1e-6
    t_valid = t[mask]
    lam_t = lam * t_valid
    term1[mask] = (1 - np.exp(-lam_t)) / lam_t
    term2 = term1 - np.exp(-lam * t)
    return b0 + (b1 * term1) + (b2 * term2)

def nelson_siegel_svensson_spot(t, b0, b1, b2, b3, lam1, lam2):
    """Calcola tassi spot NSS (vettorizzato)"""
    term1 = np.ones_like(t)
    mask = t > 1e-6
    t_valid = t[mask]

    lam1_t = lam1 * t_valid
    term1[mask] = (1 - np.exp(-lam1_t)) / lam1_t
    term2 = term1 - np.exp(-lam1 * t)

    term3 = np.zeros_like(t)
    lam2_t = lam2 * t_valid
    term3[mask] = (1 - np.exp(-lam2_t)) / lam2_t - np.exp(-lam2_t)

    return b0 + (b1 * term1) + (b2 * term2) + (b3 * term3)

def price_bond_vectorized(cfs, taus, model_type, params):
    if model_type == 'NS':
        r_spot = nelson_siegel_spot(taus, *params)
    elif model_type == 'NSS':
        r_spot = nelson_siegel_svensson_spot(taus, *params)
    return np.sum(cfs * np.exp(-r_spot * taus))

# --- OTTIMIZZAZIONE ---
def objective_function(params, bonds_data, market_prices, model_type):
    if model_type == 'NS':
        if params[3] <= 0.01: return 1e20
    elif model_type == 'NSS':
        if params[4] <= 0.01 or params[5] <= 0.01: return 1e20

    sse = 0.0
    for i, (cfs, taus) in enumerate(bonds_data):
        if len(cfs) == 0: continue
        mod_price = price_bond_vectorized(cfs, taus, model_type, params)
        sse += (market_prices[i] - mod_price) ** 2
    return sse

def calibrate_model(bond_ids, market_prices, bonds_data, model_type):
    if model_type == 'NS':
        x0 = [0.05, 0.1, -0.1, 0.1] # b0, b1, b2, lam
        bounds = [(None,None)]*3 + [(0.01, 5.0)]
    elif model_type == 'NSS':
        x0 = [0.05, 0.1, -0.1, 0.1, 0.1, 0.2] # b0, b1, b2, b3, lam1, lam2
        bounds = [(None,None)]*4 + [(0.01, 5.0), (0.01, 5.0)]

    res = minimize(
        fun=objective_function,
        x0=x0,
        args=(bonds_data, market_prices, model_type),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'maxiter': 5000}
    )
    return res.x if res.success else None, res.fun if res.success else None
