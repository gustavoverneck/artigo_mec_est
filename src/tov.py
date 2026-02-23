# src/tov.py
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Constantes de Conversão
MEV_FM3_TO_MSUN_KM3 = 8.9653e-7
G_C2 = 1.4766 # km / M_sol

def integrate_star(Pc_mev, eps_array, P_array, r_span = [1e-5, 30.0], max_step=0.1):
    """
    Integra a equação TOV para uma única pressão central.
    """
    # 1. Converter a EoS para unidades TOV (M_sol e km)
    # Inserimos (0,0) no início para garantir que a superfície (P=0) feche com densidade zero
    eps_tov = np.insert(eps_array * MEV_FM3_TO_MSUN_KM3, 0, 0.0)
    P_tov = np.insert(P_array * MEV_FM3_TO_MSUN_KM3, 0, 0.0)
    Pc_tov = Pc_mev * MEV_FM3_TO_MSUN_KM3
    
    # 2. Criar uma função contínua para a EoS via interpolação
    # Se a pressão cair abaixo de 0, a densidade de energia será 0
    eps_interp = interp1d(P_tov, eps_tov, kind='cubic', fill_value=(0.0, eps_tov[-1]), bounds_error=False)

    def tov_equations(r, y):
        P, m = y
        
        # Se a pressão for zero ou negativa, estamos fora da estrela
        if P <= 0:
            return [0, 0]
            
        eps = eps_interp(P)
        
        # Equações TOV
        # dP/dr (Gradiente de Pressão)
        num = (eps + P) * (m + 4 * np.pi * r**3 * P)
        den = r * (r - 2 * G_C2 * m)
        dPdr = - G_C2 * num / den
        
        # dm/dr (Conservação de Massa)
        dmdr = 4 * np.pi * r**2 * eps
            
        return [dPdr, dmdr]

    # 3. Evento para parar a integração quando atingir a superfície (P = 0)
    def surface_event(r, y):
        return y[0] - 1e-10 # Para numéricamente um pouquinho acima de zero
    
    surface_event.terminal = True  # Para a integração
    surface_event.direction = -1   # Apenas quando estiver diminuindo

    # 4. Condições iniciais no centro da estrela
    # Começamos em r = 1e-5 (um raio minúsculo) para evitar a divisão por zero no centro (denominador da TOV) 
    y0 = [Pc_tov, 0.0]

    # Resolver as EDOs
    sol = solve_ivp(tov_equations, r_span, y0, events=surface_event, method='RK45', max_step=max_step)

    # O raio final e a massa final são os últimos valores calculados antes de parar
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