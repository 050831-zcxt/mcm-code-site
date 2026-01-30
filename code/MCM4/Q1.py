import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 时间与基本设置
# =========================
start_year = 2025
end_year = 2125
dt = 5  # 年为单位
years = np.arange(start_year, end_year + 1, dt)
T = len(years)

# =========================
# 2. 经济参数（近似 DICE）
# =========================
gamma = 0.3  # 资本份额
delta_K = 0.1  # 资本折旧（每 5 年）
g_A_annual = 0.015  # TFP 年增长率
g_L_annual = 0.005  # 人口年增长率
g_A = (1 + g_A_annual) ** dt - 1
g_L = (1 + g_L_annual) ** dt - 1

# 初始值
Y0 = 105.5  # 初始全球产出（万亿 2010 美元）
L0 = 7.3  # 初始人口（十亿人）
A0 = 1.0  # 初始 TFP

# 由 Cobb–Douglas 反推初始资本存量
B0 = (Y0 / (A0 * (L0 ** (1 - gamma)))) ** (1 / gamma)

# 投资占比
s_I = 0.25

# 减缓成本参数 Lambda = Y * theta1 * mu^theta2
theta1 = 0.2
theta2 = 2.0

# 适应参数 D_net = a2*T^2 / (1 + psi1 * P_adapt^psi2)
a2 = 0.00236
psi1 = 50.0
psi2 = 1.0

# =========================
# 3. 排放与碳循环参数
# =========================
E0 = 35.85  # 初始工业 CO2 排放（GtCO2/5 年）
mu0 = 0.03  # 初始减排率
sigma0 = E0 / (Y0 * (1 - mu0))  # 初始碳强度

g_sigma_annual = 0.012  # 年脱碳率
g_sigma = (1 + g_sigma_annual) ** dt - 1

# 三箱碳循环矩阵（5 年步长）
phiC = np.array([
    [0.88, 0.196, 0.0],
    [0.12, 0.797, 0.001465],
    [0.0, 0.007, 0.998535]
])

# 初始碳库（GtC）
MAT0 = 851.0  # 大气
MUP0 = 460.0  # 上层海洋+陆地
MLO0 = 1740.0  # 深海

M_pre = 588.0  # 工业化前大气碳库（GtC）

# CO2 → C 转换
CO2_to_C = 12.0 / 44.0

# =========================
# 4. 气候响应参数
# =========================
eta_RF = 5.35  # W/m^2 per ln(C/C0)
F_ext = 0.5  # 非 CO2 额外强迫

PhiT = np.array([
    [0.872, 0.088],
    [0.025, 0.975]
])
kappa_F = 0.100

TAT0 = 1.1  # 初始全球均温升温（℃）
TLO0 = 0.8  # 深海温度异常（℃）

# =========================
# 5. 减排与适应路径（基准情景）
# =========================
mu_path = np.linspace(mu0, 0.5, T)  # 减排率从 3% 线性升至 50%
Padapt_path = np.full(T, 0.02)  # 适应支出 2% GWP

# =========================
# 6. 预分配数组
# =========================
Y = np.zeros(T)
A = np.zeros(T)
L = np.zeros(T)
B = np.zeros(T)
sigma = np.zeros(T)

E_CO2 = np.zeros(T)  # GtCO2
E_C = np.zeros(T)  # GtC

MAT = np.zeros(T)
MUP = np.zeros(T)
MLO = np.zeros(T)

F = np.zeros(T)
TAT = np.zeros(T)
TLO = np.zeros(T)

Lambda = np.zeros(T)  # 减排成本
Dnet = np.zeros(T)  # 净气候损失（占比）
Q = np.zeros(T)  # 净产出
I = np.zeros(T)  # 投资

# =========================
# 7. 初始化
# =========================
A[0] = A0
L[0] = L0
B[0] = B0
sigma[0] = sigma0

MAT[0] = MAT0
MUP[0] = MUP0
MLO[0] = MLO0

TAT[0] = TAT0
TLO[0] = TLO0

# =========================
# 8. 模拟主循环
# =========================
for t in range(T):
    # 经济产出
    Y[t] = A[t] * (B[t] ** gamma) * (L[t] ** (1 - gamma))

    # 排放
    mu_t = mu_path[t]
    E_CO2[t] = sigma[t] * (1 - mu_t) * Y[t]  # GtCO2
    E_C[t] = E_CO2[t] * CO2_to_C  # GtC

    # 碳循环
    M_vec = np.array([MAT[t], MUP[t], MLO[t]])
    M_next = phiC @ M_vec + np.array([E_C[t], 0.0, 0.0])
    if t < T - 1:
        MAT[t + 1], MUP[t + 1], MLO[t + 1] = M_next

    # 辐射强迫
    F[t] = eta_RF * np.log(MAT[t] / M_pre) + F_ext

    # 温度
    T_vec = np.array([TAT[t], TLO[t]])
    T_next = PhiT @ T_vec + np.array([kappa_F * F[t], 0.0])
    if t < T - 1:
        TAT[t + 1], TLO[t + 1] = T_next

    # 减排成本
    Lambda[t] = Y[t] * theta1 * (mu_t ** theta2)

    # 气候损失（考虑适应）
    Padapt_t = Padapt_path[t]
    Dnet[t] = a2 * (TAT[t] ** 2) / (1.0 + psi1 * (Padapt_t ** psi2))

    # 净产出
    Q[t] = (1.0 - Lambda[t] / Y[t]) / (1.0 + Dnet[t]) * Y[t]

    # 投资
    I[t] = s_I * Q[t]

    # 资本、TFP、人口、碳强度更新
    if t < T - 1:
        B[t + 1] = (1.0 - delta_K) * B[t] + I[t]
        A[t + 1] = A[t] * (1.0 + g_A)
        L[t + 1] = L[t] * (1.0 + g_L)
        sigma[t + 1] = sigma[t] * (1.0 - g_sigma)

# =========================
# 9. 汇总结果数据（表格）
# =========================
results = pd.DataFrame({
    "Year": years,
    "Y_GWP": Y,  # 总产出（万亿 2010 美元）
    "Q_net": Q,  # 净产出
    "Population_billion": L,  # 人口（十亿）
    "T_AT_degC": TAT,  # 全球均温升温（℃）
    "E_CO2_Gt": E_CO2,  # 工业 CO2 排放（GtCO2/5 年）
    "MAT_GtC": MAT,  # 大气碳库（GtC）
    "Damage_ratio": Dnet,  # 损失占比
    "Mitigation_cost_share": Lambda / Y,  # 减排成本占比
    "Adapt_share": Padapt_path,  # 适应支出占比
    "mu_reduction": mu_path  # 减排率
})

print("===== 结果数据（前 5 行） =====")
print(results.head())
print("\n===== 结果数据（后 5 行） =====")
print(results.tail())

print("\n===== 最终年份关键指标 =====")
print(results.iloc[-1][[
    "Year", "Y_GWP", "Q_net", "Population_billion",
    "T_AT_degC", "E_CO2_Gt", "MAT_GtC",
    "Damage_ratio", "Mitigation_cost_share",
    "Adapt_share", "mu_reduction"
]])

# =========================
# 10. 美观可视化
# =========================

# 图 1：温度 + 损失占比（双轴）
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(years, TAT, marker='o', linestyle='-')
ax1.set_xlabel("Year")
ax1.set_ylabel("Global mean temperature anomaly (°C)")
ax1.set_title("Global Temperature and Climate Damages")

# 在温度线上标注数据（每隔一个点）
for i in range(0, T, 2):
    ax1.annotate(f"{TAT[i]:.2f}",
                 (years[i], TAT[i]),
                 textcoords="offset points",
                 xytext=(0, 6), ha='center', fontsize=8)

ax2 = ax1.twinx()
ax2.plot(years, Dnet * 100, marker='s', linestyle='--')
ax2.set_ylabel("Climate damages (% of GWP)")

for i in range(0, T, 2):
    ax2.annotate(f"{Dnet[i] * 100:.2f}",
                 (years[i], Dnet[i] * 100),
                 textcoords="offset points",
                 xytext=(0, -10), ha='center', fontsize=8)

fig.tight_layout()
plt.grid(True, axis="both", linestyle=":")
plt.show()

# 图 2：GWP + 净产出 + 减排成本占比
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, Y, marker='o', linestyle='-')
ax.plot(years, Q, marker='s', linestyle='--')
ax.set_xlabel("Year")
ax.set_ylabel("Output (trillion 2010 USD)")
ax.set_title("Gross and Net Output with Mitigation Costs")

for i in range(0, T, 3):
    ax.annotate(f"{Y[i]:.1f}", (years[i], Y[i]),
                textcoords="offset points",
                xytext=(0, 6), ha='center', fontsize=8)
    ax.annotate(f"{Q[i]:.1f}", (years[i], Q[i]),
                textcoords="offset points",
                xytext=(0, -12), ha='center', fontsize=8)

ax3 = ax.twinx()
ax3.plot(years, (Lambda / Y) * 100, marker='^', linestyle=':')
ax3.set_ylabel("Mitigation cost (% of GWP)")
for i in range(0, T, 3):
    ax3.annotate(f"{(Lambda[i] / Y[i]) * 100:.2f}",
                 (years[i], (Lambda[i] / Y[i]) * 100),
                 textcoords="offset points",
                 xytext=(0, -10), ha='center', fontsize=8)

fig.tight_layout()
plt.grid(True, axis="both", linestyle=":")
plt.show()

# 图 3：排放 + 大气碳库
fig, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(years, E_CO2, marker='o', linestyle='-')
ax4.set_xlabel("Year")
ax4.set_ylabel("Industrial CO$_2$ emissions (GtCO$_2$/5y)")
ax4.set_title("CO$_2$ Emissions and Atmospheric Carbon Stock")

for i in range(0, T, 2):
    ax4.annotate(f"{E_CO2[i]:.1f}",
                 (years[i], E_CO2[i]),
                 textcoords="offset points",
                 xytext=(0, 6), ha='center', fontsize=8)

ax5 = ax4.twinx()
ax5.plot(years, MAT, marker='s', linestyle='--')
ax5.set_ylabel("Atmospheric carbon (GtC)")
for i in range(0, T, 2):
    ax5.annotate(f"{MAT[i]:.1f}",
                 (years[i], MAT[i]),
                 textcoords="offset points",
                 xytext=(0, -10), ha='center', fontsize=8)

fig.tight_layout()
plt.grid(True, axis="both", linestyle=":")
plt.show()
