import numpy as np
import matplotlib.pyplot as plt
from src.tov import generate_mr_curve

# Constants
HBAR_C = 197.3269804  # MeV*fm 

def load_eos(filepath):
    """
    Loads EoS from file.
    Assumes columns: n_B [fm^-3], Energy Density [fm^-4], Pressure [fm^-4]
    Returns eps [MeV/fm^3], P [MeV/fm^3]
    """
    data = np.loadtxt(filepath)
    
    # Identifying columns based on typical magnitude orders
    # Col 0: roughly 0.17 -> Number Density n_B
    # Col 1: roughly 7.0 -> Energy Density (dominated by mass density)
    # Col 2: roughly 2.0 -> Pressure
    
    # User said input is in fm^-3 (likely meaning natural units where everything is powers of fm)
    # Energy Density [fm^-4] -> [MeV/fm^3] by multiplying by hbar*c
    # Pressure [fm^-4] -> [MeV/fm^3] by multiplying by hbar*c
    
    # Note: If the file is strictly Number Density [fm^-3], then Col 1 and 2 must be fm^-4 to be densities.
    
    eps_fm4 = data[:, 1]
    P_fm4 = data[:, 2]
    
    # Convert to MeV/fm^3 for the TOV solver
    eps_mev = eps_fm4 * HBAR_C
    P_mev = P_fm4 * HBAR_C
    
    # Ensure they are sorted by pressure (required for interpolation usually)
    # The file seems to be in descending order? '0.17116' then '0.17072' in row 2.
    # Let's check if we need to flip.
    if P_mev[0] > P_mev[-1]:
        print("Reversing EoS data to be in ascending order...")
        eps_mev = eps_mev[::-1]
        P_mev = P_mev[::-1]
        
    return eps_mev, P_mev

def main():
    print("Loading EoS...")
    try:
        eps, P = load_eos("eos.dat")
    except Exception as e:
        print(f"Error loading EoS: {e}")
        return

    print(f"Loaded {len(eps)} points.")
    print(f"Range P: {P.min():.2e} to {P.max():.2e} MeV/fm^3")
    print(f"Range eps: {eps.min():.2e} to {eps.max():.2e} MeV/fm^3")

    print("Solving TOV equations...")
    # Use larger step for speed if needed, or keeping default
    M, R = generate_mr_curve(eps, P)
    
    print(f"Generated {len(M)} stellar configurations.")
    if len(M) > 0:
        print(f"Max Mass: {np.max(M):.2f} M_sun")
        print(f"Radius at Max Mass: {R[np.argmax(M)]:.2f} km")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(R, M, label='EoS Model', linewidth=2, color='blue')
    plt.xlabel('Radius [km]')
    plt.ylabel('Mass [M$_{\odot}$]')
    plt.title('Mass-Radius Relation')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('mr_curve.png')
    print("Plot saved to mr_curve.png")
    # plt.show() # Uncomment if running in an environment with display

if __name__ == "__main__":
    main()
