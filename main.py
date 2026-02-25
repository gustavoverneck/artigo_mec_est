from src.constants import *
from src.eos import *
from src.tov import generate_mr_curve
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Vetor de densidades bariônicas de 0.05 a 1.0 fm^-3 (A densidade de saturação nuclear é ~0.16 fm^-3)
    n_b_array = np.linspace(0.01, 15, 150)

    # 1. Calcula a EoS
    print("Calculando a Equação de Estado (EoS)...")
    eps_pure, P_pure = solve_pure_neutrons(n_b_array)
    eps_npe, P_npe, y_p = solve_npe_gas(n_b_array)
    
    # 2. Aplica os Filtros da Física (Causalidade e Estabilidade)
    eps_pure, P_pure = validate_eos(eps_pure, P_pure)
    eps_npe, P_npe = validate_eos(eps_npe, P_npe)

    
    # 3. Integra a TOV
    print("Integrando as equações TOV para o Gás n,p,e em Equilíbrio Beta...")
    M_pure, R_pure = generate_mr_curve(eps_pure, P_pure)
    M_npe, R_npe = generate_mr_curve(eps_npe, P_npe)
    
    # 4. Aplica o Filtro de Estabilidade Gravitacional
    # M_pure, R_pure = validate_mr_curve(M_pure, R_pure)
    # M_npe, R_npe = validate_mr_curve(eps_npe, P_npe)


    # Densidades de elétrons em fm^-3
    ne_array = np.logspace(-12, -4, 150)
    print("Calculando EoS da Anã Branca...")
    eps_wd, P_wd = solve_white_dwarf(ne_array, A_Z=2.0)

    # 1. Verificação Física da EoS
    eps_wd, P_wd = validate_eos(eps_wd, P_wd)

    print("Integrando a estrela (Isso pode levar alguns segundos)...")
    M_wd, R_wd = generate_mr_curve(eps_wd, P_wd, r_span=[1e-5, 20000.0], max_step=30)

    # 2. Verificação Física da Curva Massa-Raio
    # M_wd, R_wd = validate_mr_curve(M_wd, R_wd)

    
# ==========================================
    # Plotagem - Estilo Revista (Nature/Science)
    # ==========================================
    # Configurações para renderização LaTeX e fontes serifadas
    # [Ajuste] "text.usetex": False para evitar erro se não tiver LaTeX instalado
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Fonte matemática estilo Computer Modern (LaTeX)
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (7.0, 2.5),  # Largura para coluna dupla (~180mm) ou ajustado
        "lines.linewidth": 1.5,
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
    })

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)

    # Plot 1: EoS
    ax1 = axes[0]
    ax1.plot(eps_pure, P_pure, label=r'Nêutrons Puros', color='#1f77b4')  # Azul padrão
    ax1.plot(eps_npe, P_npe, label=r'$n, p, e$', color='#d62728', linestyle='--')  # Vermelho padrão
    ax1.set_xlabel(r'Dens. de Energia $\varepsilon$ (MeV/fm$^3$)')
    ax1.set_ylabel(r'Pressão $P$ (MeV/fm$^3$)')
    ax1.set_title(r'a) Equação de Estado', fontweight='bold')
    ax1.legend(loc='best', frameon=False)
    ax1.grid(True)

    # Plot 2: Fração de Prótons
    ax2 = axes[1]
    ax2.plot(n_b_array, y_p * 100, color='#2ca02c')  # Verde padrão
    ax2.set_xlabel(r'Densidade Bariônica $n_B$ (fm$^{-3}$)')
    ax2.set_ylabel(r'Fração de Prótons $y_p$ (%)')
    ax2.set_title(r'b) Equilíbrio Químico', fontweight='bold')
    ax2.grid(True)

    # Plot 3: Curva Massa-Raio
    ax3 = axes[2]
    # Nêutrons Puros
    ax3.plot(R_pure, M_pure, label=r'Nêutrons Puros', color='#1f77b4')
    # n,p,e
    ax3.plot(R_npe, M_npe, label=r'$n, p, e$', color='#d62728', linestyle='--')

    # Marcar a Massa Máxima
    max_idx = np.argmax(M_npe)
    M_max = M_npe[max_idx]
    R_max = R_npe[max_idx]
    
    ax3.scatter(R_max, M_max, color='black', s=20, zorder=5)
    # Anotação com setinha ou texto próximo
    ax3.annotate(f'$M_{{\\rm max}} = {M_max:.2f} M_\\odot$\n$R = {R_max:.1f}$ km',
                 xy=(R_max, M_max), xytext=(R_max + 1.1, M_max),
                 arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
                 fontsize=10)

    ax3.set_xlabel(r'Raio $R$ (km)')
    ax3.set_ylabel(r'Massa $M$ ($M_\odot$)')
    ax3.set_title(r'c) Diagrama Massa-Raio', fontweight='bold')
    ax3.legend(loc='lower right', frameon=False)
    ax3.grid(True)

    plt.savefig('result_plot.pdf', dpi=300, bbox_inches='tight')
    # plt.show()


# -------------------------
    # Plot 1: EoS WD
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    ax1 = axes[0]
    ax1.plot(eps_wd, P_wd, label=r'Elétrons Puros', color='#1f77b4')  # Azul padrãoVermelho padrão
    ax1.set_xlabel(r'Dens. de Energia $\varepsilon$ (MeV/fm$^3$)')
    ax1.set_ylabel(r'Pressão $P$ (MeV/fm$^3$)')
    ax1.set_title(r'a) Equação de Estado', fontweight='bold')
    ax1.legend(loc='best', frameon=False)
    ax1.grid(True)

    # Plot 2: Massa-Raio
    ax2 = axes[1]
    max_idx = np.argmax(M_wd)
    ax2.plot(R_wd, M_wd, label=r'Elétrons Puros', color='#1f77b4')
    ax2.set_xlabel('Raio $R$ (km)')
    ax2.set_ylabel('Massa $M$ ($M_\\odot$)')
    ax2.set_title(r'b) Diagrama Massa-Raio', fontweight='bold')
    ax2.grid(True)

    # Destacando a massa máxima
    ax2.scatter(R_wd[max_idx], M_wd[max_idx], color='black', zorder=5)
    ax2.axhline(y=1.44, color='gray', linestyle='--', label='Limite de Chandrasekhar (1.44 $M_\odot$)')
    
    ax2.text(0.95, 0.95, 
             f'$M_{{\\rm max}}$ = {M_wd[max_idx]:.2f} $M_\odot$\n$R \\approx$ {R_wd[max_idx]:.0f} km', 
             transform=ax2.transAxes,
             ha='right', va='top', multialignment='left',
             fontsize=10)


    plt.savefig('result__wd_plot.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

