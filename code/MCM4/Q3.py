import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ==========================================
# 0. 全局高级绘图设置
# ==========================================
# 使用 Seaborn 的高级样式
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'figure.facecolor': 'white'})
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'SimHei'],  # 兼容中文系统，防止方块
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})

# 定义高级自定义色盘
colors = ["#3B8EA5", "#F49E4C", "#AB3428", "#6BBA6E", "#2E5EAA"]

# ==========================================
# 1. 核心模型参数定义
# ==========================================
year_start = 2025
year_end = 2125
years = np.arange(year_start, year_end + 1)
T_steps = len(years)
dt = 1.0

# 临界点网络 (GIS, AMOC, AMZ, PF)
n_tips = 4
# 阈值设置：格陵兰(1.8), AMOC(3.5), 亚马逊(4.0), 冻土(5.0)
T_crit_base = np.array([1.8, 3.5, 4.0, 5.0])
alpha_tip = np.array([0.08, 0.10, 0.05, 0.05])  # 敏感度
sigma_tip = np.array([0.05, 0.06, 0.04, 0.04])  # 噪声强度

# 气候惯性参数
beta_T = 0.02  # 排放对升温的贡献
kappa_T = 0.005  # 自然冷却/海洋吸收率
sigma_T = 0.03  # 温度的年际波动


# ==========================================
# 2. 策略与动力学函数
# ==========================================

def get_policy_paths(u):
    """根据决策变量 u 生成减排和适应投资路径"""
    k_mu, t_mid, P0, r_P = u
    # 1. 减排路径 (S型曲线)
    mu_t = 1.0 / (1.0 + np.exp(-k_mu * (years - t_mid)))
    # 2. 适应投资 (指数增长)
    P_adapt_t = P0 * np.exp(r_P * (years - year_start))
    return mu_t, P_adapt_t


def simulate_system(u, seed=42):
    """
    核心动力学模拟引擎
    返回: 温度历史, 临界点状态历史, 是否崩溃(bool)
    """
    rng = np.random.default_rng(seed)
    mu_t, _ = get_policy_paths(u)

    # 初始化状态
    T = 1.25  # 2025年起始温度异常
    X = -1.0 * np.ones(n_tips)  # 初始状态为 -1 (健康态)

    hist_T = np.zeros(T_steps)
    hist_X = np.zeros((T_steps, n_tips))
    tipped = False

    for k in range(T_steps):
        # --- A. 气候系统更新 ---
        # 净排放 = 基准 * (1 - 减排率)
        E_eff = 1.0 * (1.0 - mu_t[k])
        # 温度微分方程: dT = (加热 - 冷却)dt + 噪声
        T = T + (beta_T * E_eff - kappa_T * (T - 0.5)) * dt + sigma_T * rng.normal()
        hist_T[k] = T

        # --- B. 临界点网络更新 (Langevin Equation) ---
        for i in range(n_tips):
            # 内部回复力 (双稳态势能井导数): -(x^3 - x)
            restoring = (X[i] - X[i] ** 3)
            # 气候强迫力: 当 T > T_crit 时，推向崩溃
            forcing = alpha_tip[i] * (T - T_crit_base[i])
            # 随机干扰
            noise = sigma_tip[i] * rng.normal()

            # 更新状态
            X[i] = X[i] + (restoring + forcing) * dt + noise

            # 判断是否崩溃 (越过 0 点)
            if X[i] > 0:
                tipped = True

        hist_X[k, :] = X

    return hist_T, hist_X, tipped


# ==========================================
# 3. 绘图部分 (含报错修复)
# ==========================================

# 设定一组“最优”策略参数用于展示
# [减排速度k, 关键年份t_mid, 初始适应P0, 适应增长率r]
optimal_u = [0.15, 2045, 0.005, 0.025]


def plot_all_figures():
    print("开始生成图表，请稍候...")

    # -------------------------------------------------------
    # 图 1: 政策敏感性热力图 (Policy Sensitivity Heatmap)
    # -------------------------------------------------------
    print("正在计算热力图数据 (Step 1/3)...")
    N_grid = 20
    k_mu_range = np.linspace(0.05, 0.25, N_grid)
    t_mid_range = np.linspace(2030, 2060, N_grid)
    prob_matrix = np.zeros((N_grid, N_grid))

    # 计算网格数据
    for i in range(N_grid):
        for j in range(N_grid):
            u_test = [k_mu_range[i], t_mid_range[j], 0.005, 0.025]
            tipped_count = 0
            # 模拟次数减少到 20 次以加快速度 (正式跑建议 50-100)
            N_sim_fast = 20
            for seed in range(N_sim_fast):
                _, _, tipped = simulate_system(u_test, seed=seed)
                if tipped: tipped_count += 1
            prob_matrix[i, j] = tipped_count / N_sim_fast

    # 绘图
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(prob_matrix, ax=ax1, cmap="rocket_r",
                cbar_kws={'label': 'System Collapse Probability'},
                linewidths=0.5, linecolor='white',
                square=True, vmin=0, vmax=1)

    # *** 修复报错的核心代码 ***
    # 我们手动设置刻度，不依赖 Seaborn 的自动推断
    xticks_idx = np.arange(0, N_grid, 3)
    yticks_idx = np.arange(0, N_grid, 3)

    ax1.set_xticks(xticks_idx + 0.5)  # 居中
    ax1.set_xticklabels(np.round(t_mid_range[xticks_idx]).astype(int))

    ax1.set_yticks(yticks_idx + 0.5)  # 居中
    ax1.set_yticklabels(np.round(k_mu_range[yticks_idx], 2))

    ax1.invert_yaxis()  # 让 y 轴从小到大向上

    ax1.set_xlabel("Action Timing ($t_{mid}$ Year)", fontweight='bold')
    ax1.set_ylabel("Mitigation Aggressiveness ($k_\mu$)", fontweight='bold')
    ax1.set_title("Policy Sensitivity: Risk Landscape", fontsize=18, pad=20)

    # 标记最优策略点
    best_k_idx = np.argmin(np.abs(k_mu_range - optimal_u[0]))
    best_t_idx = np.argmin(np.abs(t_mid_range - optimal_u[1]))
    ax1.scatter(best_t_idx + 0.5, best_k_idx + 0.5, marker='*', s=400, c='yellow', edgecolors='black', linewidth=1.5,
                label='Optimal Strategy')
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 图 2: 系统相平面轨迹图 (Phase Space Trajectory)
    # -------------------------------------------------------
    print("正在生成相平面图 (Step 2/3)...")
    h_T, h_X, _ = simulate_system(optimal_u, seed=101)
    x_gis = h_X[:, 0]
    x_amoc = h_X[:, 1]

    fig2, ax2 = plt.subplots(figsize=(10, 9))

    # 绘制背景区域
    safe_patch = mpatches.Rectangle((-1.5, -1.5), 1.5, 1.5, color='green', alpha=0.1, label='Safe Basin')
    danger_patch = mpatches.Rectangle((0, 0), 0.5, 0.5, color='red', alpha=0.1, label='Tipping Zone')
    ax2.add_patch(safe_patch)
    ax2.add_patch(danger_patch)

    # 绘制散点轨迹
    sc = ax2.scatter(x_gis, x_amoc, c=years, cmap='viridis', s=50, alpha=0.8, edgecolor='none')
    ax2.plot(x_gis, x_amoc, c='gray', alpha=0.3, linewidth=1)

    # 标记起点终点
    ax2.scatter(x_gis[0], x_amoc[0], c='green', s=200, marker='o', edgecolors='black', linewidth=2,
                label='Start (2025)')
    ax2.scatter(x_gis[-1], x_amoc[-1], c='blue', s=200, marker='X', edgecolors='black', linewidth=2, label='End (2125)')

    # 辅助线
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5)

    ax2.set_xlim(-1.4, 0.4)
    ax2.set_ylim(-1.4, 0.4)
    ax2.set_xlabel("Greenland Ice Sheet State ($x_1$)", fontweight='bold')
    ax2.set_ylabel("AMOC Circulation State ($x_2$)", fontweight='bold')
    ax2.set_title("Phase Space Trajectory: The Path to Safety", fontsize=18, pad=20)

    cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
    cbar.set_label('Year')

    # 合并图例
    handles, labels = ax2.get_legend_handles_labels()
    handles.extend([safe_patch, danger_patch])
    labels.extend(['Safe Basin', 'Tipping Zone'])
    ax2.legend(handles=handles, labels=labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 图 3: 电影级温度“超调”图 (Cinematic Temperature)
    # -------------------------------------------------------
    print("正在生成温度预测图 (Step 3/3)...")
    N_sim_temp = 100
    T_ensemble = []
    for i in range(N_sim_temp):
        t_run, _, _ = simulate_system(optimal_u, seed=i + 200)
        T_ensemble.append(t_run)
    T_ensemble = np.array(T_ensemble)

    T_mean = np.mean(T_ensemble, axis=0)
    T_p95 = np.percentile(T_ensemble, 97.5, axis=0)
    T_p05 = np.percentile(T_ensemble, 2.5, axis=0)
    T_p75 = np.percentile(T_ensemble, 75, axis=0)
    T_p25 = np.percentile(T_ensemble, 25, axis=0)

    fig3, ax3 = plt.subplots(figsize=(12, 7))

    # 绘制多层置信区间
    ax3.fill_between(years, T_p05, T_p95, color=colors[0], alpha=0.15, label="95% Confidence Interval")
    ax3.fill_between(years, T_p25, T_p75, color=colors[0], alpha=0.3, label="50% Confidence Interval")

    # 绘制均值线
    ax3.plot(years, T_mean, color='white', linewidth=4, alpha=0.7)  # 光晕效果
    ax3.plot(years, T_mean, color=colors[4], linewidth=2.5, label="Mean Projection")

    # 阈值参考线
    ax3.axhline(1.5, color='gray', linestyle=':', linewidth=1.5)
    ax3.axhline(2.0, color='black', linestyle='--', linewidth=1.5)

    # 危险区域背景
    ax3.fill_between(years, 2.0, 3.0, color='#AB3428', alpha=0.08)
    ax3.text(2110, 2.1, "DANGER ZONE (>2.0°C)", color='#AB3428', fontweight='bold', ha='center')
    ax3.text(2110, 1.55, "Paris Goal (1.5°C)", color='gray', ha='center')

    # 标记峰值 (Overshoot)
    peak_idx = np.argmax(T_mean)
    peak_year = years[peak_idx]
    peak_temp = T_mean[peak_idx]
    ax3.scatter(peak_year, peak_temp, color='#F49E4C', s=150, zorder=5, edgecolors='white', linewidth=2)
    ax3.annotate(f'Peak Overshoot\n~{peak_year}: {peak_temp:.2f}°C',
                 xy=(peak_year, peak_temp), xytext=(peak_year + 10, peak_temp + 0.3),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'),
                 fontsize=12, fontweight='bold')

    ax3.set_xlim(year_start, year_end)
    ax3.set_ylim(1.0, 2.8)
    ax3.set_xlabel("Year", fontweight='bold')
    ax3.set_ylabel("Global Temperature Anomaly (°C)", fontweight='bold')
    ax3.set_title("The \"Overshoot\" Trajectory: Stabilizing the Climate", fontsize=18, pad=20)

    ax3.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='none')
    sns.despine()  # 去除边框
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    print("绘图完成！")


# 运行绘图主程序
if __name__ == "__main__":
    plot_all_figures()