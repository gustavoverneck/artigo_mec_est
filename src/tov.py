# src/tov.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Constantes de Conversão
MEV_FM3_TO_MSUN_KM3 = 8.9653e-7
G_C2 = 1.4766 # km / M_sol

def integrate_star(Pc_mev, eps_array, P_array, r_span = [1e-5, 30.0], max_step=0.1):
    # 1. Conversão de Unidades (M_sol e km)
    eps_tov = eps_array * MEV_FM3_TO_MSUN_KM3
    P_tov = P_array * MEV_FM3_TO_MSUN_KM3
    Pc_tov = Pc_mev * MEV_FM3_TO_MSUN_KM3
    
    # A menor pressão válida que calculamos na nossa física
    P_min = P_tov[0] 
    
    # 2. Interpolação sem distorcer o final
    # Usamos 'extrapolate' apenas para o integrador não quebrar se der um passo milimétrico fora da tabela
    eps_interp = interp1d(P_tov, eps_tov, kind='linear', fill_value="extrapolate")

    def tov_equations(r, y):
        P, m = y
        
        # Se a pressão cair abaixo da nossa tabela física, estamos fora da estrela
        if P <= P_min:
            return [0, 0]
            
        eps = eps_interp(P)
        num = (eps + P) * (m + 4 * np.pi * r**3 * P)
        den = r * (r - 2 * G_C2 * m)
        dPdr = - G_C2 * num / den
        dmdr = 4 * np.pi * r**2 * eps
            
        return [dPdr, dmdr]

    # 3. O Evento de Superfície
    def surface_event(r, y):
        # Para a integração EXATAMENTE quando atinge o limite inferior da sua tabela EoS
        return y[0] - P_min 
    
    surface_event.terminal = True  
    surface_event.direction = -1   

    # 4. Limites da Simulação
    r_span = [1e-5, 30000.0] # 30.000 km é mais que suficiente para qualquer anã branca
    y0 = [Pc_tov, 0.0]

    # Integração com tolerâncias rígidas
    sol = solve_ivp(tov_equations, r_span, y0, events=surface_event, method='RK45', 
                    max_step=10.0, rtol=1e-8, atol=1e-20)

    R = sol.t[-1]
    M = sol.y[1][-1]
    
    return M, R

def generate_mr_curve(eps_array, P_array, r_span = [1e-5, 30.0], max_step=0.1):
    """
    Gera a curva Massa-Raio iterando sobre várias pressões centrais da EoS.
    """
    masses = []
    radii = []
    
    # Pegamos um conjunto de pressões centrais da nossa tabela EoS
    # (Ignoramos as mais baixinhas para evitar erros de limite)
    central_pressures = P_array[5:] 
    
    for Pc in central_pressures:
        M, R = integrate_star(Pc, eps_array, P_array, r_span, max_step)
        # Filtro básico: só guardamos se fez uma estrela fisicamente razoável
        if M > 0.05 and R > 2.0:
            masses.append(M)
            radii.append(R)
            
    return np.array(masses), np.array(radii)