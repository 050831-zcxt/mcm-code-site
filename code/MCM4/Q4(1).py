import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource

# ==========================================
# 0. 全局画图设置 (期刊级风格)
# ==========================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "figure.dpi": 150
})

COLOR_BLUE   = "#4E79A7"
COLOR_RED    = "#E15759"
COLOR_ORANGE = "#F28E2B"
COLOR_GREEN  = "#59A14F"
COLOR_GRAY   = "#BAB0AC"

# ==========================================
# 1. 数据生成
# ==========================================
years = np.arange(2025, 2126)
n_years = len(years)

# (A) 策略数据
mu_t = 1.0 / (1.0 + np.exp(-0.15 * (years - 2045)))       # S 型减排
P_adapt_t = 0.005 * np.exp(0.03 * (years - 2025))         # 指数适应投资

# (B) 温度数据
T_mean = 1.2 + 0.5 * np.exp(-((years - 2060)**2) / (2 * 20**2))
np.random.seed(42)
noise = np.random.normal(0, 0.1, size=(100, n_years))
T_ensemble = T_mean + noise
T_upper = np.percentile(T_ensemble, 95, axis=0)
T_lower = np.percentile(T_ensemble, 5, axis=0)

# (C) 临界点状态
X_data = np.zeros((n_years, 4))
X_data[:, 0] = -1.0 + 0.8 * np.exp(-((years - 2055)**2) / (2 * 15**2)) + 0.05 * np.random.normal(size=n_years)
X_data[:, 1] = -1.0 + 0.6 * np.exp(-((years - 2065)**2) / (2 * 20**2)) + 0.05 * np.random.normal(size=n_years)
X_data[:, 2] = -1.0 + 0.4 * np.exp(-((years - 2060)**2) / (2 * 25**2))
X_data[:, 3] = -1.0 + 0.3 * np.exp(-((years - 2060)**2) / (2 * 30**2))

# ==========================================
# 2. 三张图分别输出
# ==========================================

# --- 图 1: 最优策略路径 ---
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111)
ax1_r = ax1.twinx()

ax1.fill_between(years, 0, mu_t, color=COLOR_BLUE, alpha=0.1)
l1 = ax1.plot(years, mu_t, color=COLOR_BLUE, lw=3, label="Mitigation Rate $\mu(t)$")
l2 = ax1_r.plot(years, P_adapt_t * 100, color=COLOR_ORANGE, lw=3, ls='--',
               label="Adaptation Investment")

ax1.axvline(2045, color='gray', linestyle=':', alpha=0.6)
ax1.text(2046, 0.5, "Inflection Point\n(Year 2045)", color='gray', fontsize=10)

ax1.set_title("Optimal Strategic Pathway", loc='left')
ax1.set_ylabel("Mitigation Rate (0–1)", color=COLOR_BLUE, fontweight='bold')
ax1_r.set_ylabel("Adaptation Cost (% of GDP)", color=COLOR_ORANGE, fontweight='bold')
ax1.set_ylim(0, 1.1)
ax1_r.set_ylim(0, 10)
ax1.grid(True, alpha=0.2)

lns = l1 + l2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', frameon=True, framealpha=0.9)

plt.tight_layout()
plt.show()

# --- 图 2: 温度超调动态 ---
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)

ax2.fill_between(years, T_lower, T_upper, color=COLOR_GREEN, alpha=0.2,
                 label="95% Confidence Interval")
ax2.fill_between(years, T_mean - 0.05, T_mean + 0.05, color=COLOR_GREEN, alpha=0.4,
                 label="50% Confidence Interval")
ax2.plot(years, T_mean, color=COLOR_GREEN, lw=3, label="Projected Temperature")

ax2.axhline(1.5, color='gray', ls=':', lw=1.5)
ax2.text(2026, 1.52, "1.5°C Target", color='gray', fontsize=10)
ax2.axhline(2.0, color=COLOR_RED, ls='--', lw=1.5)
ax2.text(2026, 2.02, "2.0°C Danger Line", color=COLOR_RED, fontsize=10)

ax2.fill_between(years, 2.0, 2.5, color=COLOR_RED, alpha=0.05)

ax2.set_title("Temperature Overshoot Dynamics", loc='left')
ax2.set_ylabel("Temp. Anomaly (°C)", fontweight='bold')
ax2.set_ylim(1.0, 2.3)
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# --- 图 3: 各临界元稳定性轨迹 ---
fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111)

ax3.axhline(0, color=COLOR_RED, lw=2, alpha=0.8)
ax3.text(2026, 0.05, "TIPPING THRESHOLD (IRREVERSIBLE COLLAPSE)",
         color=COLOR_RED, fontweight='bold')
ax3.axhline(-1, color=COLOR_BLUE, lw=1.5, ls='--')
ax3.text(2026, -0.95, "HEALTHY BASIN ATTRACTOR",
         color=COLOR_BLUE, fontweight='bold')

labels = ['Greenland Ice Sheet (GIS)', 'AMOC Circulation',
          'Amazon Rainforest', 'Permafrost']
colors_tip = [COLOR_RED, "#9D7660", COLOR_GREEN, "#384259"]

for i in range(4):
    ax3.plot(years, X_data[:, i], label=labels[i],
             color=colors_tip[i], lw=2.5)

ax3.fill_between(years, -0.3, 0, color=COLOR_RED, alpha=0.05,
                 label="High Risk Zone")

ax3.set_title("Stability Trajectories of Tipping Elements", loc='left')
ax3.set_ylabel("System State Index $x_i(t)$", fontweight='bold')
ax3.set_xlabel("Year", fontweight='bold')
ax3.set_ylim(-1.2, 0.2)
ax3.legend(loc='lower right', ncol=2, frameon=True)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

# ==========================================
# 3. 3D 敏感性曲面（原来那张图，保持不变）
# ==========================================
fig4 = plt.figure(figsize=(10, 8))
ax = fig4.add_subplot(111, projection='3d')

X = np.linspace(-0.5, 0.5, 30)
Y = np.linspace(0.5, 2.0, 30)
X, Y = np.meshgrid(X, Y)
Z = 1 / (1 + np.exp(8 * (Y - 1.2) - 4 * X))

ls = LightSource(270, 45)
rgb = ls.shade(Z, cmap=cm.RdYlGn, vert_exag=0.1, blend_mode='soft')

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       facecolors=rgb, linewidth=0,
                       antialiased=False, shade=False)
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.RdYlGn, alpha=0.5)

ax.set_xlabel('\nThreshold Uncertainty ($\delta_T$)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel('\nInteraction Strength ($\gamma$)', fontsize=11, fontweight='bold', labelpad=10)
ax.set_zlabel('\nSurvival Probability', fontsize=11, fontweight='bold', labelpad=10)
ax.set_title("Robustness Landscape: The Cliff of Risk", fontsize=18, fontweight='bold', pad=10)

ax.set_zlim(0, 1.0)
ax.view_init(elev=25, azim=-130)

m = cm.ScalarMappable(cmap=cm.RdYlGn)
m.set_array(Z)
cbar = plt.colorbar(m, shrink=0.5, pad=0.1)
cbar.set_label('Safety Level (0=Collapse, 1=Safe)', fontweight='bold')

plt.tight_layout()
plt.show()
