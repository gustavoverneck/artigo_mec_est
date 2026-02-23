import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from .constants import PI, PI2, HBAR_C, M_N, M_P, M_E

def moment_fermi(n):
    """Calcula o momento de Fermi (em MeV) a partir da densidade numérica (em fm^-3)."""
    return HBAR_C * (3 * np.pi**2 * n)**(1/3)

def integrand_energy(k, m):
    """Integrando para a densidade de energia."""
    return (k**2) * np.sqrt(k**2 + m**2)

def integrand_pressure(k, m):
    """Integrando para a pressão."""
    return (k**4) / np.sqrt(k**2 + m**2)

def calc_energy_density(kf, m):
    """Densidade de energia epsilon (MeV/fm^3)."""
    if kf == 0: return 0.0
    integral, _ = quad(integrand_energy, 0, kf, args=(m,))
    return (1 / (np.pi**2 * HBAR_C**3)) * integral

def calc_pressure(kf, m):
    """Pressão P (MeV/fm^3)."""
    if kf == 0: return 0.0
    integral, _ = quad(integrand_pressure, 0, kf, args=(m,))
    return (1 / (3 * np.pi**2 * HBAR_C**3)) * integral

# ==========================================
# Caso 1: Gás de Nêutrons Puro
# ==========================================
def solve_pure_neutrons(n_b_array):
    epsilon_list, pressure_list = [], []
    
    for n_B in n_b_array:
        kf_n = moment_fermi(n_B)
        eps = calc_energy_density(kf_n, M_N)
        P = calc_pressure(kf_n, M_N)
        
        epsilon_list.append(eps)
        pressure_list.append(P)
        
    return np.array(epsilon_list), np.array(pressure_list)

# ==========================================
# Caso 2: Gás de n, p, e (Equilíbrio Beta)
# ==========================================
def beta_equilibrium_residual(n_p, n_B):
    """Função cujo zero determina a fração de prótons correta para o equilíbrio químico."""
    n_n = n_B - n_p
    n_e = n_p # Neutralidade de carga
    
    kf_n = moment_fermi(n_n)
    kf_p = moment_fermi(n_p)
    kf_e = moment_fermi(n_e)
    
    mu_n = np.sqrt(kf_n**2 + M_N**2)
    mu_p = np.sqrt(kf_p**2 + M_P**2)
    mu_e = np.sqrt(kf_e**2 + M_E**2)
    
    return mu_n - (mu_p + mu_e)

def solve_npe_gas(n_b_array):
    epsilon_list, pressure_list = [], []
    proton_fractions = []
    
    for n_B in n_b_array:
        # Encontra n_p que zera a equação de equilíbrio beta
        # Os limites para n_p são um pouco maiores que 0 até n_B/2
        sol = root_scalar(beta_equilibrium_residual, args=(n_B,), bracket=[1e-10, n_B/2])
        n_p = sol.root
        n_n = n_B - n_p
        n_e = n_p
        proton_fractions.append(n_p / n_B)
        
        # Momentos de Fermi
        kf_n = moment_fermi(n_n)
        kf_p = moment_fermi(n_p)
        kf_e = moment_fermi(n_e)
        
        # Somatório das contribuições de densidade de energia e pressão
        eps = calc_energy_density(kf_n, M_N) + calc_energy_density(kf_p, M_P) + calc_energy_density(kf_e, M_E)
        P = calc_pressure(kf_n, M_N) + calc_pressure(kf_p, M_P) + calc_pressure(kf_e, M_E)
        
        epsilon_list.append(eps)
        pressure_list.append(P)
        
    return np.array(epsilon_list), np.array(pressure_list), np.array(proton_fractions)

def solve_white_dwarf(ne_array, A_Z=2.0):
    """
    Calcula a EoS para uma Anã Branca (Carbono/Oxigênio).
    ne_array: array de densidades numéricas de elétrons (em fm^-3)
    A_Z: Razão Massa/Carga dos íons (2.0 para C ou O)
    """
    epsilon_list, pressure_list = [], []
    
    for ne in ne_array:
        # O momento de Fermi é ditado pelos elétrons
        kf_e = moment_fermi(ne)
        
        # 1. A Pressão vem inteiramente do gás de elétrons degenerados
        P = calc_pressure(kf_e, M_E)
        
        # 2. A Densidade de Energia (Massa) vem da energia cinética dos elétrons 
        # MAIS a massa de repouso gigante dos núcleos (prótons e nêutrons).
        eps_e = calc_energy_density(kf_e, M_E)
        
        # Densidade de energia dos íons (n_ions * Massa do Núcleon)
        # Como A_Z = 2, temos 2 núcleons para cada elétron
        eps_ions = ne * A_Z * M_N 
        
        # Densidade de energia total
        eps = eps_e + eps_ions
        
        epsilon_list.append(eps)
        pressure_list.append(P)
        
    return np.array(epsilon_list), np.array(pressure_list)

def validate_eos(eps_array, P_array):
    """Filtra a EoS removendo pontos que violam a física."""
    # Derivada dP/deps (Velocidade do som ao quadrado)
    vs_squared = np.gradient(P_array, eps_array)
    
    # Criamos uma máscara booleana: Maior que 0 (Estável) e Menor/Igual a 1 (Causal)
    valid_mask = (vs_squared > 0.0) & (vs_squared <= 1.0)
    
    # Adicional: Pressão e energia devem ser positivas
    valid_mask &= (P_array >= 0) & (eps_array >= 0)
    
    # Retorna apenas os dados que passaram no teste
    return eps_array[valid_mask], P_array[valid_mask]

def validate_mr_curve(masses, radii):
    """
    Corta a curva Massa-Raio descartando o ramo instável (após a massa máxima)
    e pontos onde o integrador falhou (M < 0 ou R < 0).
    """
    # 1. Filtra lixo numérico básico
    valid_basic = (masses > 0) & (radii > 0)
    M_clean = masses[valid_basic]
    R_clean = radii[valid_basic]
    
    if len(M_clean) == 0:
        return np.array([]), np.array([])
        
    # 2. Encontra o índice da massa máxima
    max_idx = np.argmax(M_clean)
    
    # 3. Retorna apenas a curva do início até a massa máxima
    # (Descarta tudo o que vem depois do max_idx)
    M_stable = M_clean[:max_idx + 1]
    R_stable = R_clean[:max_idx + 1]
    
    return M_stable, R_stable