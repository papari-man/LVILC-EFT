import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# --------------------------------------------------------------------------------
# 1. 環境構築 (CAMBのインストール)
# --------------------------------------------------------------------------------
try:
    import camb
except ImportError:
    print("Installing CAMB...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "camb"])
    import camb

# --------------------------------------------------------------------------------
# 2. パラメータ設定
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
# 4. Sigma8 計算 (ここで変数を確実に定義します)
# --------------------------------------------------------------------------------
def calc_sigma_R(k, Pk, R_Mpc_h):
    x = k * R_Mpc_h
    W = np.ones_like(x)
    mask = x > 1e-5
    W[mask] = 3.0 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask])) / (x[mask]**3)
    integrand = k**2 * Pk * W**2
    return np.sqrt(trapezoid(integrand, k) / (2 * np.pi**2))

# 変数名を統一して計算
sigma8_lcdm = calc_sigma_R(kh, P_lcdm, 8.0)
sigma8_lvilc = calc_sigma_R(kh, P_lvilc, 8.0)
print(f"Sigma8 LCDM : {sigma8_lcdm:.5f}")
print(f"Sigma8 LVILC: {sigma8_lvilc:.5f}")

# --------------------------------------------------------------------------------
# 5. HMF (ハロー質量関数) 計算
# --------------------------------------------------------------------------------
M_bins = np.logspace(8, 13, 100)
rho_crit = 2.775e11
Om_m = (params['omch2'] + params['ombh2']) / h**2
rho_m = rho_crit * Om_m

def get_sigma_vector(k, Pk, M):
    R = (3 * M / (4 * np.pi * rho_m))**(1.0/3.0)
    return np.array([calc_sigma_R(k, Pk, r) for r in R])

def get_hmf_proxy(M, sig):
    # d(ln sigma)/d(ln M)
    dlns_dlnM = np.abs(np.gradient(np.log(sig), np.gradient(np.log(M))))
    # Sheth-Tormen
    A, a, p, dc = 0.3222, 0.707, 0.3, 1.686
    nu = dc / sig
    f_nu = A * np.sqrt(2*a/np.pi) * (1+(1/(a*nu**2))**p) * nu * np.exp(-a*nu**2/2)
    return (rho_m/M) * f_nu * dlns_dlnM

sig_lcdm_M = get_sigma_vector(kh, P_lcdm, M_bins)
sig_lvilc_M = get_sigma_vector(kh, P_lvilc, M_bins)

hmf_lcdm = get_hmf_proxy(M_bins, sig_lcdm_M)
hmf_lvilc = get_hmf_proxy(M_bins, sig_lvilc_M)
ratio = hmf_lvilc / hmf_lcdm

# --------------------------------------------------------------------------------
# 6. プロット (警告回避のため rf文字列を使用)
# --------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1
ax[0].plot(kh, S_k, color='crimson', lw=3, label=rf'Transfer Function ($k_c={kc_fid}, \lambda={lam_fid}$)')
ax[0].set_xscale('log')
ax[0].set_title('Power Spectrum Suppression')
ax[0].set_ylabel(r'Ratio $P_{\mathrm{LVILC}} / P_{\Lambda\mathrm{CDM}}$')
ax[0].set_xlabel(r'Wavenumber $k$ [h/Mpc]')
ax[0].axvspan(10, 100, color='orange', alpha=0.1, label='Dwarf Scale')
ax[0].legend()
ax[0].grid(alpha=0.3)

# Panel 2
# ここで計算済みの変数 sigma8_lcdm, sigma8_lvilc を使います
ax[1].text(0.5, 0.6, "Large Scale Structure Check", fontsize=14, ha='center', fontweight='bold')
ax[1].text(0.5, 0.4, rf"$\sigma_8$ ($\Lambda$CDM): {sigma8_lcdm:.5f}" + "\n" +
                     rf"$\sigma_8$ (LVILC): {sigma8_lvilc:.5f}" + "\n\n" +
                     f"Diff: {sigma8_lvilc - sigma8_lcdm:.5f}", 
           fontsize=13, ha='center', bbox=dict(facecolor='white', alpha=0.8))
ax[1].axis('off')
ax[1].set_title(r'$\sigma_8$ Consistency')

# Panel 3
ax[2].plot(M_bins, ratio, 'purple', lw=3)
ax[2].set_xscale('log')
ax[2].set_ylim(0, 1.1)
ax[2].axhline(1, color='k', ls='--')
ax[2].set_xlabel(r'Halo Mass $M_{\odot}/h$')
ax[2].set_title('Missing Satellites Solution')
ax[2].fill_between(M_bins, 0, ratio, color='purple', alpha=0.1)

# Highlight
idx = np.argmin(np.abs(M_bins - 1e8))
val = ratio[idx]
ax[2].plot(M_bins[idx], val, 'ro', markersize=8)
ax[2].text(M_bins[idx], val+0.1, f"{val*100:.1f}% Survival", color='red', fontweight='bold')

plt.tight_layout()
plt.show()
