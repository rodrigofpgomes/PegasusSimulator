import numpy as np
from itertools import product

def compute_bound(m, J, kx, kv, kR, kw, c1, c2, psi1, Bbar, kappa, sigma=0.0):
    lJmin, lJmax = np.linalg.eigvalsh(J)[[0, -1]]
    g = 9.81

    # --- 1. Enquadramento de V ---
    W11 = 0.5 * np.array([[kx, -c1], [-c1, m]])
    W12 = 0.5 * np.array([[kx,  c1], [ c1, m]])
    W21 = 0.5 * np.array([[kR, -c2], [-c2, lJmin]])
    W22 = 0.5 * np.array([[2*kR/(2-psi1), c2], [c2, lJmax]])
    
    lo_w11 = np.linalg.eigvalsh(W11).min()
    lo_w21 = np.linalg.eigvalsh(W21).min()
    if lo_w11 <= 0 or lo_w21 <= 0:
        return None # c1 ou c2 violam condicoes de enquadramento
        
    lo = min(lo_w11, lo_w21)
    hi = max(np.linalg.eigvalsh(W12).max(), np.linalg.eigvalsh(W22).max())

    # --- 2. Taxa de decrescimento (CORRIGIDO) ---
    # Removidos os termos -0.5*kx e -0.5*kR das diagonais cruzadas
    Wa = np.array([[c1*kx/m,           -0.5*(c1*kv/m)],
                   [-0.5*(c1*kv/m),     kv - c1]])
                   
    Wb = np.array([[c2*kR/lJmax,       -0.5*(c2*kw/lJmin)],
                   [-0.5*(c2*kw/lJmin), kw - c2]])
                   
    la = np.linalg.eigvalsh(Wa).min()
    lb = np.linalg.eigvalsh(Wb).min()
    if la <= 0 or lb <= 0:
        return None

    # Termo de acoplamento translacao-atitude
    Wx = np.linalg.norm(np.array([[c1*Bbar/m, 0.0], [Bbar, 0.0]]), 2)
    
    if Wx**2 >= 4 * la * lb:
        return None # Acoplamento excessivo (bound quebra)
        
    lam_m = np.linalg.eigvalsh(np.array([[la, -0.5*Wx], [-0.5*Wx, lb]])).min()
    if lam_m <= 0:
        return None

    # --- 3. Calculo do Bound Final (B) ---
    c4 = np.sqrt(m**2 + c1**2) / m
    delta = kappa * m * g
    
    B = np.sqrt(hi/lo) * c4 * (delta + sigma) / lam_m
    return B

def best_bound(kappa, **p):
    best = (np.inf, None)
    c1max = min(p['kv'], 4*p['m']*p['kx']*p['kv']/(p['kv']**2 + 4*p['m']*p['kx']),
                np.sqrt(p['kx']*p['m']))
    lJmin = np.linalg.eigvalsh(p['J'])[0]
    c2max = min(p['kw'], np.sqrt(p['kR']*lJmin))
    
    for c1, c2 in product(np.linspace(0.01, 0.99, 100) * c1max,
                          np.linspace(0.01, 0.99, 100) * c2max):
        B = compute_bound(c1=c1, c2=c2, kappa=kappa, **p)
        if B is not None and B < best[0]:
            best = (B, (c1, c2))
    return best

# ==========================================
# Execucao do Calculo
# ==========================================
if __name__ == "__main__":
    # Parametros fisicos extraidos do shuttle.usda

    params = {
        'm': 3.4207,
        'J': np.diag([0.021666666, 0.021666666, 0.04]), 
        
        # --- Ganhos de Certificação Analítica ---
        'kx': 150.0,  
        'kv': 50.0,
        'kR': 1500.0, # Mola de atitude titânica para a matemática
        'kw': 50.0,
        
        'psi1': 0.5, 
        'Bbar': 1.2 * 3.4207 * 9.81, # 40.26 N
        'sigma': 0.0
    }

    print("Calculando o limite de erro garantido B(kappa)...\\n")
    print(f"{'Kappa':<10} | {'Delta_f (N)':<15} | {'c1, c2 otimos':<25} | {'B (metros)'}")
    print("-" * 75)
    
    for kappa in [0.05, 0.1, 0.2, 0.3, 0.5]:
        best_B, best_c = best_bound(kappa=kappa, **params)
        
        if best_B != np.inf:
            delta_f = kappa * params['m'] * 9.81
            c_str = f"({best_c[0]:.2f}, {best_c[1]:.2f})"
            print(f"{kappa:<10} | {delta_f:<15.2f} | {c_str:<25} | {best_B:.4f} m")
        else:
            print(f"{kappa:<10} | {'-':<15} | {'Falha (Ganhos fracos)':<25} | None")