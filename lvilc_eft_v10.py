# --------------------------------------------------------------------------------
# 5. HMF (ハロー質量関数) の精密計算 (Sheth-Tormen with Jacobian)
# --------------------------------------------------------------------------------
# 質量範囲
M_bins = np.logspace(8, 13, 100) # 範囲を矮小銀河メインに調整

def get_sigma_and_derivative(k, Pk, M_vector):
    """
    sigma(R) と d(ln sigma)/d(ln M) を計算する
    """
    # Mass -> Radius
    R_vector = (3 * M_vector / (4 * np.pi * rho_m))**(1.0/3.0)

    # 1. Sigmaの計算
    sigmas = []
    for R in R_vector:
        sigmas.append(calc_sigma_R(k, Pk, R))
    sigmas = np.array(sigmas)

    # 2. 微分項 |d ln(sigma) / d ln M| の計算
    # M ~ R^3 なので dlnM = 3 dlnR
    # ここでは数値微分(np.gradient)で d(ln sigma) / d(ln M) を直接求める
    d_ln_sigma = np.gradient(np.log(sigmas))
    d_ln_M = np.gradient(np.log(M_vector))
    # ゼロ除算回避
    d_ln_M[d_ln_M==0] = 1e-10

    abs_dlns_dlnM = np.abs(d_ln_sigma / d_ln_M)

    return sigmas, abs_dlns_dlnM

def compute_hmf_full(M_vector, sigma, dlns_dlnM):
    """
    Sheth-Tormen HMF: dn/dlnM = (rho_m / M) * f(nu) * |dlns/dlnM|
    """
    A, a, p = 0.3222, 0.707, 0.3
    delta_c = 1.686
    nu = delta_c / sigma

    # f(nu)
    f_nu = A * np.sqrt(2*a/np.pi) * (1 + (1/(a*nu**2))**p) * nu * np.exp(-a * nu**2 / 2)

    # Full formula
    return (rho_m / M_vector) * f_nu * dlns_dlnM

# --- 計算実行 ---
# LCDM
sig_lcdm, deriv_lcdm = get_sigma_and_derivative(kh, P_lcdm, M_bins)
dn_lcdm = compute_hmf_full(M_bins, sig_lcdm, deriv_lcdm)

# LVILC
sig_lvilc, deriv_lvilc = get_sigma_and_derivative(kh, P_lvilc, M_bins)
dn_lvilc = compute_hmf_full(M_bins, sig_lvilc, deriv_lvilc)

# 比率
ratio_hmf = dn_lvilc / dn_lcdm

# --------------------------------------------------------------------------------
# 6. プロット作成 (修正版)
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Transfer Function ---
ax = axes[0]
ax.plot(kh, S_k, color='crimson', lw=3, label=f'Transfer Function\n$k_c={kc_fid}, \lambda={lam_fid}$')
ax.axvspan(0.1, 3.0, color='gray', alpha=0.1, label='Lyman-$\\alpha$ Safe Zone')
ax.axvspan(10.0, 100.0, color='orange', alpha=0.1, label='Dwarf Galaxy Scale')
ax.set_xscale('log')
ax.set_xlabel(r'Wavenumber $k$ $[h/\mathrm{Mpc}]$', fontsize=12)
ax.set_ylabel(r'Ratio $P_{\mathrm{LVILC}} / P_{\Lambda\mathrm{CDM}}$', fontsize=12)
ax.set_title('Power Spectrum Suppression', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# --- Plot 2: Sigma(R) ---
ax = axes[1]
R_range = np.linspace(1, 20, 50)
sig8_lcdm_plt = [calc_sigma_R(kh, P_lcdm, r) for r in R_range]
sig8_lvilc_plt = [calc_sigma_R(kh, P_lvilc, r) for r in R_range]

ax.plot(R_range, sig8_lcdm_plt, 'k--', label='$\Lambda$CDM')
ax.plot(R_range, sig8_lvilc_plt, 'r-', lw=2, label='LVILC-EFT')
ax.axvline(8.0, color='blue', ls=':', label='$R=8 \mathrm{Mpc}/h$')
ax.set_xlabel(r'Scale $R$ $[\mathrm{Mpc}/h]$', fontsize=12)
ax.set_ylabel(r'$\sigma(R)$', fontsize=12)
ax.set_title(f'$\sigma_8$ Consistent\nDiff: {sigma8_lvilc_calc - sigma8_lcdm_calc:.5f}', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# --- Plot 3: HMF Suppression Ratio (CORRECTED) ---
ax = axes[2]
ax.plot(M_bins, ratio_hmf, color='purple', lw=3, label='Survival Rate')
ax.axhline(1.0, color='k', ls='--', label='No Suppression')

# 重要な数値を表示
val_at_1e8 = ratio_hmf[np.argmin(np.abs(M_bins - 1e8))]
ax.plot(1e8, val_at_1e8, 'ro', markersize=8)
ax.text(1e8, val_at_1e8 + 0.05, f'{val_at_1e8*100:.1f}% Survival\n@ $10^8 M_\odot$',
        color='red', fontweight='bold')

ax.set_xscale('log')
ax.set_xlabel(r'Halo Mass $M$ $[M_\odot/h]$', fontsize=12)
ax.set_ylabel(r'Abundance Ratio $n_{\mathrm{LVILC}} / n_{\Lambda\mathrm{CDM}}$', fontsize=12)
ax.set_title('Missing Satellites Solution', fontsize=14)
ax.fill_between(M_bins, 0, ratio_hmf, color='purple', alpha=0.1)
ax.grid(alpha=0.3, which='both')
ax.set_ylim(0, 1.1)
ax.legend()

plt.tight_layout()
plt.show()