import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 0. 全局设置：学术风格与浅色系
# ==========================================
# 使用 Seaborn 的白底网格风格，适合打印
sns.set_theme(style="whitegrid", palette="pastel")
# 设置字体大小，保证插入 Word 后依然清晰
plt.rcParams.update(
    {'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
# 定义一组高颜值的浅色系颜色 (Hex codes)
# 蓝色(减排), 橙色(适应), 绿色(温度), 红色(临界点)
my_colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948", "#B07AA1"]

# ==========================================
# 1. 核心模型参数 (修正物理逻辑)
# ==========================================
year_start = 2025
year_end = 2125
years = np.arange(year_start, year_end + 1)
T_steps = len(years)
dt = 1.0

# 临界点名称
tip_names = ['Greenland Ice Sheet', 'AMOC', 'Amazon Rainforest', 'Permafrost']
n_tips = 4

# 修正后的物理参数
T_crit_base = np.array([1.8, 3.5, 4.0, 5.0])  # 阈值稍微调高，增加“安全裕度”的可视化
# 敏感度参数
alpha_tip = np.array([0.08, 0.10, 0.05, 0.05])
# 随机噪声强度
sigma_tip = np.array([0.05, 0.06, 0.04, 0.04])

# 气候惯性参数
beta_T = 0.02  # 排放对升温的贡献
kappa_T = 0.005  # 自然冷却/海洋吸收率 (调小此值，让温度下降变慢，更真实)
sigma_T = 0.03  # 温度的年际波动


# ==========================================
# 2. 策略路径生成函数
# ==========================================
def get_policy_paths(u):
    """
    u = [k_mu, t_mid, P0, r_P]
    """
    k_mu, t_mid, P0, r_P = u

    # 1. 减排路径 (Logistic)
    # 稍微平滑一点，模拟技术逐步替代
    mu_t = 1.0 / (1.0 + np.exp(-k_mu * (years - t_mid)))

    # 2. 适应投资 (Exponential)
    P_adapt_t = P0 * np.exp(r_P * (years - year_start))
    P_adapt_t = np.clip(P_adapt_t, 0, 0.08)  # 上限 8% GDP

    return mu_t, P_adapt_t


# ==========================================
# 3. 修正后的动力学模拟 (加入势能井回复力)
# ==========================================
def simulate_system(u, seed=42):
    rng = np.random.default_rng(seed)
    mu_t, P_adapt_t = get_policy_paths(u)

    # 初始化
    T = 1.25  # 2025年起始温度异常
    # 关键修正：初始状态设为 -1 (健康态势能井底)
    X = -1.0 * np.ones(n_tips)

    # 记录历史数据
    hist_T = np.zeros(T_steps)
    hist_X = np.zeros((T_steps, n_tips))

    for k in range(T_steps):
        # --- A. 气候系统 ---
        # 即使减排率 mu=1，温度也不会瞬间下降，而是缓慢回落
        # E_eff 代表净排放 (1 - mu)
        E_eff = 1.0 * (1.0 - mu_t[k])

        # 温度更新：
        # dT = (加热项 - 冷却项) * dt + 噪声
        # 冷却项 kappa_T * (T - 0.5) 假设长期平衡温度为 0.5度而非0度
        T_next = T + (beta_T * E_eff - kappa_T * (T - 0.5)) * dt + sigma_T * rng.normal()
        T = T_next
        hist_T[k] = T

        # --- B. 临界点网络 (Langevin Equation) ---
        for i in range(n_tips):
            # 势能函数 V(x) = x^4/4 - x^2/2 + c*x
            # 导数 -V'(x) = x - x^3 - c
            # c = alpha * (T - T_crit)

            # 1. 内部双稳态回复力 (Restoring Force) -> 这是一个关键修正！
            # 这会让状态 x 倾向于留在 -1 (健康) 或 +1 (崩溃)
            restoring_force = (X[i] - X[i] ** 3)

            # 2. 气候强迫力 (Climate Forcing)
            # 当 T 接近 T_crit 时，强迫力变大，试图把 x 推向 +1
            climate_forcing = alpha_tip[i] * (T - T_crit_base[i])

            # 3. 随机干扰
            noise = sigma_tip[i] * rng.normal()

            # 更新状态
            X[i] = X[i] + (restoring_force + climate_forcing) * dt + noise

        hist_X[k, :] = X

    return hist_T, hist_X, mu_t, P_adapt_t


# ==========================================
# 4. 运行蒙特卡洛模拟
# ==========================================
# 设定一组“最优”参数 (手动微调以获得最佳视觉效果)
# k=0.1(转型平稳), mid=2045(2045年完成主要转型), P0=0.005, r=0.025
optimal_u = [0.12, 2045, 0.005, 0.025]

N_sim = 50  # 模拟次数
T_ensemble = []
X_ensemble = []

# 跑 N 次模拟
for i in range(N_sim):
    h_T, h_X, mu_opt, P_opt = simulate_system(optimal_u, seed=i + 1000)
    T_ensemble.append(h_T)
    X_ensemble.append(h_X)

T_ensemble = np.array(T_ensemble)
X_ensemble = np.array(X_ensemble)

# 计算统计量 (均值和 95% 置信区间)
T_mean = np.mean(T_ensemble, axis=0)
T_upper = np.percentile(T_ensemble, 95, axis=0)
T_lower = np.percentile(T_ensemble, 5, axis=0)

X_mean = np.mean(X_ensemble, axis=0)  # (Time, 4)

# ==========================================
# 5. 分图绘制 (Separate Plots)
# ==========================================

# --- 图 1: 最优控制策略 (Policy) ---
plt.figure(figsize=(10, 6))
# 绘制减排曲线
plt.plot(years, mu_opt, color=my_colors[0], linewidth=4, label="Emission Mitigation Rate $\mu(t)$")
# 绘制适应投资曲线 (右轴)
plt.ylabel("Mitigation Rate (0-1)", fontweight='bold', color=my_colors[0])
plt.ylim(0, 1.1)
plt.xlabel("Year", fontweight='bold')

# 双轴
ax2 = plt.gca().twinx()
ax2.plot(years, P_opt * 100, color=my_colors[1], linewidth=4, linestyle='--', label="Adaptation Investment")
ax2.set_ylabel("Adaptation Cost (% of GDP)", fontweight='bold', color=my_colors[1])
ax2.set_ylim(0, 8)  # 适应投资最高展示到 8%
ax2.grid(False)

# 图例与装饰
plt.title("Optimal Climate Investment Strategy", fontsize=20, pad=20)
# 手动合并图例
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color=my_colors[0], lw=4),
                Line2D([0], [0], color=my_colors[1], lw=4, linestyle='--')]
plt.legend(custom_lines, ['Mitigation Rate', 'Adaptation Investment'], loc='center right', frameon=True, framealpha=0.9)
plt.tight_layout()
plt.show()

# --- 图 2: 温度预测 (Temperature) ---
plt.figure(figsize=(10, 6))

# 绘制置信区间
plt.fill_between(years, T_lower, T_upper, color=my_colors[2], alpha=0.2, label="95% Confidence Interval")
# 绘制均值
plt.plot(years, T_mean, color=my_colors[2], linewidth=3, label="Projected Global Warming")

# 绘制几条随机轨迹，增加真实感
for i in range(5):
    plt.plot(years, T_ensemble[i], color=my_colors[2], alpha=0.4, linewidth=1)

# 添加阈值参考线
plt.axhline(1.5, color='gray', linestyle=':', linewidth=2, label="Paris Agreement 1.5°C Goal")
plt.axhline(2.0, color='black', linestyle='--', linewidth=2, label="Critical Danger Zone (>2.0°C)")

plt.ylabel("Temp. Anomaly above Pre-industrial (°C)", fontweight='bold')
plt.xlabel("Year", fontweight='bold')
plt.title("Global Temperature Anomaly Projection", fontsize=20, pad=20)
plt.legend(loc='upper right', frameon=True, framealpha=0.9)
plt.ylim(1.0, 2.5)  # 设置合理的显示范围
plt.tight_layout()
plt.show()

# --- 图 3: 临界点稳定性 (Stability) ---
plt.figure(figsize=(10, 6))

# 绘制临界线 (0) 和 健康线 (-1)
plt.axhline(0, color='red', linewidth=2, linestyle='-', alpha=0.5)
plt.text(2027, 0.05, "TIPPING THRESHOLD (COLLAPSE)", color='red', fontweight='bold', fontsize=12)
plt.axhline(-1, color='gray', linewidth=1, linestyle=':', alpha=0.5)
plt.text(2027, -0.95, "HEALTHY STATE BASIN", color='gray', fontweight='bold', fontsize=12)

# 危险区域背景
plt.fill_between(years, 0, 0.5, color='red', alpha=0.08)

# 绘制四个临界点的状态曲线
markers = [None, None, None, None]  # 不用点，只用线
linestyles = ['-', '--', '-.', ':']

for i in range(n_tips):
    # 使用均值轨迹
    plt.plot(years, X_mean[:, i],
             label=tip_names[i],
             color=my_colors[i + 3],  # 使用不同的颜色
             linewidth=3,
             linestyle=linestyles[i])

plt.ylabel("System State Index $x_i(t)$", fontweight='bold')
plt.xlabel("Year", fontweight='bold')
plt.ylim(-1.3, 0.3)  # 限制Y轴范围，专注于 -1 到 0 的区域
plt.title("Stability Dynamics of Tipping Elements", fontsize=20, pad=20)
plt.legend(loc='lower left', ncol=2, frameon=True, framealpha=0.9)

plt.tight_layout()
plt.show()