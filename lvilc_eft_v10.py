import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# --------------------------------------------------------------------------------
# 1. 環境構築 & CAMB準備
# --------------------------------------------------------------------------------
try:
    import camb
except ImportError:
    print("Installing CAMB...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "camb"])
    import camb

# --------------------------------------------------------------------------------
# 2. パラメータ設定 (ここがないとエラーになります)
# --------------------------------------------------------------------------------
h = 0.6736
params = {
    'H0': h * 100,
    'ombh2': 0.02237,
    'omch2': 0.1200,
    'As': 2.0968e-9,
    'ns': 0.9649,
    'tau': 0.0544
}
kc_fid = 4.5    # Cutoff scale
lam_fid = 4.0   # Sharpness

# --------------------------------------------------------------------------------
# 3. CAMB計算 & LVILC適用
# --------------------------------------------------------------------------------
print("Running CAMB computation...")
pars = camb.CAMBparams()
pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'], tau=params['tau'])
pars.InitPower.set_params(As=params['As'], ns=params['ns'])
pars.set_matter_power(redshifts=[0.], kmax=100.0)
pars.set_dark_energy()

results = camb.get_results(pars)
kh, z, pk_lin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=100.0, npoints=2000)
P_lcdm = pk_lin[0]

# LVILC Transfer Function
S_k = 1.0 / (1.0 + (kh / kc_fid)**lam_fid)
P_lvilc = P_lcdm * S_k

# --------------------------------------------------------------------------------
# 4. Sigma8 計算 (変数を定義)
# --------------------------------------------------------------------------------
def calc_sigma_R(k, Pk, R_Mpc_h):
    x = k * R_Mpc_h
    W = np.ones_like(x)
    mask = x > 1e-5
    W[mask] = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / (x[mask]**3)
    integrand = k**2 * Pk * W**2
    return np.sqrt(trapezoid(integrand, k) / (2 * np.pi**2))

# あなたのコードに合わせて変数名を「_calc」付きで定義
sigma8_lcdm_calc = calc_sigma_R(kh, P_lcdm, 8.0)
sigma8_lvilc_calc = calc_sigma_R(kh, P_lvilc, 8.0)

print(f"Sigma8 LCDM : {sigma8_lcdm_calc:.5f}")
print(f"Sigma8 LVILC: {sigma8_lvilc_calc:.5f}")

# --------------------------------------------------------------------------------
# 5. HMF (ハロー質量関数) の精密計算 (あなたが持ってきた成功ロジック)
# --------------------------------------------------------------------------------
# 質量範囲
M_bins = np.logspace(8, 13, 100)
rho_crit = 2.775e11
Om_m = (params['omch2'] + params['ombh2']) / h**2
rho_m = rho_crit * Om_m

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
    d_ln_sigma = np.gradient(np.log(sigmas))
    d_ln_M = np.gradient(np.log(M_vector))
    # ゼロ除算回避
    d_ln_M[d_ln_M==0] = 1e-10

    abs_dlns_dlnM = np.abs(d_ln_sigma / d_ln_M)

    return sigmas, abs_dlns_dlnM

def compute_hmf_full(M_vector, sigma, dlns_dlnM):
    """
    Sheth-Tormen HMF
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
# 6. プロット作成 (SyntaxWarning修正版)
# --------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# --- Plot 1: Transfer Function ---
ax = axes[0]
# 修正: label=f'...' -> label=rf'...' にして警告を消去
ax.plot(kh, S_k, color='crimson', lw=3, label=rf'Transfer Function ($k_c={kc_fid}, \lambda={lam_fid}$)')
ax.axvspan(0.1, 3.0, color='gray', alpha=0.1, label=r'Lyman-$\alpha$ Safe Zone')
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

ax.plot(R_range, sig8_lcdm_plt, 'k--', label=r'$\Lambda$CDM')
ax.plot(R_range, sig8_lvilc_plt, 'r-', lw=2, label='LVILC-EFT')
ax.axvline(8.0, color='blue', ls=':', label=r'$R=8 \mathrm{Mpc}/h$')
ax.set_xlabel(r'Scale $R$ $[\mathrm{Mpc}/h]$', fontsize=12)
ax.set_ylabel(r'$\sigma(R)$', fontsize=12)
# 修正: タイトルに raw string (r) を使用
ax.set_title(rf'$\sigma_8$ Consistent' + '\n' + f'Diff: {sigma8_lvilc_calc - sigma8_lcdm_calc:.5f}', fontsize=14)
ax.legend()
ax.grid(alpha=0.3)

# --- Plot 3: HMF Suppression Ratio ---
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