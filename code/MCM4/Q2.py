import numpy as np
import matplotlib.pyplot as plt


def main():
    # =========================
    # 1. 时间与基础设置
    # =========================
    t0, t1, dt = 2025.0, 2125.0, 0.1      # 年份起止与步长（0.1 年）
    times = np.arange(t0, t1 + dt, dt)
    n_steps = len(times)

    # 四个临界要素：GIS, AMOC, AMZ, PF
    labels = ["GIS (Greenland Ice Sheet)",
              "AMOC",
              "Amazon Rainforest",
              "Permafrost"]

    # 势能函数参数：V_i = a/4 x^4 - b/2 x^2 + c_i(T) x
    a = 1.0
    b = 1.0

    # 临界温度阈值 Tcrit（单位：摄氏度高于前工业）
    # 取值基于文献区间的中值近似
    Tcrit = np.array([
        1.8,   # GIS
        3.0,   # AMOC
        2.5,   # Amazon
        1.5    # Permafrost
    ])

    # 温度对势能“倾斜”的敏感系数 kappa
    kappa = np.array([
        0.8,   # GIS
        0.7,   # AMOC
        0.6,   # Amazon
        0.9    # Permafrost
    ])

    # 噪声强度 sigma（高斯白噪声）
    sigma = np.array([
        0.20,  # GIS
        0.15,  # AMOC
        0.20,  # Amazon
        0.25   # Permafrost
    ])

    # =========================
    # 2. 网络耦合矩阵 W
    # =========================
    # w_ji 表示 j 对 i 的影响（与你 PDF 中的矩阵一致）
    W = np.array([
        [0.0,  -0.10,  0.01,  0.02],
        [0.40,  0.0,   0.02,  0.03],
        [0.05,  0.35,  0.0,   0.01],
        [0.02, -0.20,  0.01,  0.0]
    ])

    # =========================
    # 3. 设定全球升温路径 T(t)
    # =========================
    # 这里给一个简单的“当前政策 / 中等排放”轨迹：
    # 2025 年 1.2°C -> 2100 年 2.7°C -> 2125 年 3.0°C
    def T_of_year(year: float) -> float:
        if year <= 2100:
            return 1.2 + (2.7 - 1.2) * (year - 2025.0) / (2100.0 - 2025.0)
        else:
            return 2.7 + (3.0 - 2.7) * (year - 2100.0) / (2125.0 - 2100.0)

    T_series = np.array([T_of_year(y) for y in times])

    # =========================
    # 4. SDE 模拟（Euler–Maruyama）
    # =========================
    n_sims = 200
    n_vars = 4

    xsims = np.zeros((n_sims, n_steps, n_vars))

    # 初始状态：全部接近健康态 x ≈ -1
    x0 = np.array([-0.9, -0.9, -0.9, -0.9])

    for s in range(n_sims):
        x = x0.copy()
        xsims[s, 0, :] = x
        for k in range(n_steps - 1):
            T = T_series[k]
            # 控制参数 c_i(T) = kappa_i * (T - Tcrit_i)
            c = kappa * (T - Tcrit)

            # 势能导数 dV/dx = a x^3 - b x + c
            dVdx = a * x ** 3 - b * x + c

            # 网络耦合：sum_j w_ji * (x_j + 1) / 2
            # 这里用 W^T @ (...)，得到每个 i 的 Coupling_i
            coupling = W.T @ ((x + 1.0) / 2.0)

            # 漂移项：-dV/dx + coupling
            drift = -dVdx + coupling

            # 随机项：sigma_i * dW_i
            dW = np.sqrt(dt) * np.random.randn(n_vars)

            x = x + drift * dt + sigma * dW

            # 为防止数值发散，将状态限制在 [-1.5, 1.5]
            x = np.clip(x, -1.5, 1.5)

            xsims[s, k + 1, :] = x

    # =========================
    # 5. 统计量计算
    # =========================
    mean_x = xsims.mean(axis=0)
    std_x = xsims.std(axis=0)

    # 定义“已触发 tipped”：x > 0.5
    tipped = xsims > 0.5
    prob_tipped = tipped.mean(axis=0)   # shape: (n_steps, 4)

    final_probs = prob_tipped[-1, :]

    print("=== Tipping probabilities at year 2125 ===")
    for i, name in enumerate(labels):
        print(f"{name}: {final_probs[i]:.3f}")

    # =========================
    # 6. 可视化
    # =========================

    # 图 1：一条样本路径
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sample_idx = 0
    for i in range(n_vars):
        ax1.plot(times, xsims[sample_idx, :, i], label=labels[i])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("State variable $x_i$")
    ax1.set_title("Sample Path of Tipping Element States")
    ax1.axhline(0.0, linestyle="--")
    ax1.legend()
    fig1.tight_layout()

    # 图 2：均值 ± 1 标准差
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for i in range(n_vars):
        ax2.plot(times, mean_x[:, i], label=labels[i])
        ax2.fill_between(times,
                         mean_x[:, i] - std_x[:, i],
                         mean_x[:, i] + std_x[:, i],
                         alpha=0.2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mean state $\\bar{x}_i$")
    ax2.set_title("Ensemble Mean and Variability of Tipping Element States")
    ax2.axhline(0.0, linestyle="--")
    ax2.legend()
    fig2.tight_layout()

    # 图 3：随时间的触发概率
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for i in range(n_vars):
        ax3.plot(times, prob_tipped[:, i], label=labels[i])
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Probability of being tipped (x_i > 0.5)")
    ax3.set_title("Tipping Probabilities Over Time")
    ax3.set_ylim(0.0, 1.05)
    ax3.legend()
    fig3.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
