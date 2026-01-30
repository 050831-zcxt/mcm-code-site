import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource

# ==========================================
# 0. 全局画图设置 (期刊级风格)
# ==========================================
# 设置字体为类似 LaTeX 的衬线体
plt.rcParams.update({
    "text.usetex": False,  # 如果电脑没装 TeX，设为 False 也能用数学符号
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm", # 数学公式使用 Computer Modern 字体
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# 高级配色 (来源于 Nature 期刊常用色)
COLOR_BAU = "#D64045"      # 警示红
COLOR_DELAY = "#E9B44C"    # 警戒黄
COLOR_OPTIMAL = "#467599"  # 安全蓝
COLOR_SAFE_ZONE = "#9ED8DB" # 浅青色背景
COLOR_DANGER = "#FAD4D8"   # 浅红色背景

# ==========================================
# 1. 准备数据 (基于模型逻辑的模拟数据)
# ==========================================
years = np.arange(2025, 2126)
n_years = len(years)

# 模拟三条温度曲线
# 1. BAU: 指数上升
temp_bau = 1.2 + 0.03 * (years - 2025) + 0.0001 * (years - 2025)**2
# 2. Delayed: 先升后降，但有超调 (Overshoot)
temp_delay = 1.2 + 0.035 * (years - 2025) * np.exp(-0.02 * (years - 2045))
# 3. Optimal: S型控制，稳定在 1.5 以下
temp_opt = 1.2 + (1.45 - 1.2) * (1 - np.exp(-0.1 * (years - 2025)))

# 添加随机噪声 (模拟蒙特卡洛的不确定性)
np.random.seed(42)
noise = np.random.normal(0, 0.05, size=(100, n_years)) # 100次模拟

# ==========================================
# 绘图 1: 情景对比分析 (带置信区间)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制背景区域
ax.axhspan(2.0, 4.0, color=COLOR_DANGER, alpha=0.3, label='Danger Zone (>2.0°C)')
ax.axhspan(0, 1.5, color=COLOR_SAFE_ZONE, alpha=0.2, label='Safe Operating Space')

# 绘制线条和阴影
# Optimal
ax.plot(years, temp_opt, color=COLOR_OPTIMAL, lw=3, label='Optimal Strategy')
ax.fill_between(years, temp_opt - 0.1, temp_opt + 0.1, color=COLOR_OPTIMAL, alpha=0.2)

# Delayed
ax.plot(years, temp_delay, color=COLOR_DELAY, lw=2.5, ls='--', label='Delayed Action (2045)')

# BAU
ax.plot(years, temp_bau, color=COLOR_BAU, lw=2.5, ls=':', label='Business-as-Usual')

# 装饰
ax.set_ylim(1.0, 3.5)
ax.set_xlim(2025, 2125)
ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Global Temp. Anomaly ($\Delta T$, °C)', fontweight='bold')
ax.set_title(r'Temperature Trajectories under Policy Scenarios', pad=20)
ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='gray')
ax.grid(True, linestyle='--', alpha=0.5)

# 添加标注箭头
ax.annotate('Tipping Point Risk High', xy=(2100, 3.0), xytext=(2060, 3.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5), fontsize=12)

plt.tight_layout()
plt.show()

# ==========================================
# 绘图 2: 延迟行动的代价 (条形图 + 拟合曲线)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

delay_years = np.array([0, 5, 10, 15, 20])
# 假设成本呈指数增长: Cost = Base * e^(0.15 * delay)
costs = 100 * np.exp(0.15 * delay_years)

# 绘制条形图
bars = ax.bar(delay_years, costs, color=COLOR_OPTIMAL, alpha=0.8, width=3, edgecolor='black', zorder=3)

# 绘制拟合趋势线 (红色虚线)
x_smooth = np.linspace(0, 20, 100)
y_smooth = 100 * np.exp(0.15 * x_smooth)
ax.plot(x_smooth, y_smooth, color=COLOR_BAU, lw=2, ls='--', zorder=4, label='Exponential Cost Growth')

# 装饰
ax.set_xlabel('Years of Delay ($t_{mid} - 2025$)', fontweight='bold')
ax.set_ylabel('Required Adaptation Investment (Index)', fontweight='bold')
ax.set_title(r'The Economic Cost of Inaction', pad=20)
ax.set_xticks(delay_years)
ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 在柱子上标数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', color='black')

plt.tight_layout()
plt.show()

# ==========================================
# 绘图 3: 3D 敏感性分析 (鲁棒性景观)
# ==========================================
# 生成 3D 数据
# X轴: 阈值不确定性 (-0.5 到 0.5 度)
# Y轴: 交互强度 (0.5倍 到 2.0倍)
X = np.linspace(-0.5, 0.5, 50)
Y = np.linspace(0.5, 2.0, 50)
X, Y = np.meshgrid(X, Y)

# Z轴: 生存概率 (Survival Probability)
# 逻辑：交互越强(Y大)，阈值越低(X小)，生存概率越低
# 使用 Sigmoid 函数模拟这种非线性
Z = 1 / (1 + np.exp(5 * (Y - 1.2) - 3 * X))
Z = 1 - Z # 反转一下，左下角高，右上角低

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 自定义光照效果，让曲面更有质感
ls = LightSource(270, 45)
# 使用 Coolwarm 色系：红色代表危险(0)，蓝色代表安全(1)
rgb = ls.shade(Z, cmap=cm.RdYlBu, vert_exag=0.1, blend_mode='soft')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

# 投影到底部 (等高线)
ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.RdYlBu, alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('\nThreshold Uncertainty ($\delta_T$)', fontsize=12, fontweight='bold')
ax.set_ylabel('\nInteraction Strength ($\gamma$)', fontsize=12, fontweight='bold')
ax.set_zlabel('\nSurvival Probability', fontsize=12, fontweight='bold')
ax.set_title(r'Robustness Landscape: The "Cliff" of Risk', pad=20)

# 调整视角 (最能展示"悬崖"的角度)
ax.view_init(elev=30, azim=-120)
ax.set_zlim(0, 1.0)

# 添加颜色条
m = cm.ScalarMappable(cmap=cm.RdYlBu)
m.set_array(Z)
cbar = fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('Survival Probability')

plt.tight_layout()
plt.show()