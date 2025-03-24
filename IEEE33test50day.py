import random
import scipy.io as sio
import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
import numpy
import pandas as pd
import copy as cp
import matplotlib.pyplot as plt
from typing import Optional, Tuple
# 测试环境(50天)
class IEEE33_ENV:
    def __init__(self):
        numpy.set_printoptions(threshold=sys.maxsize)  # 控制小数精度为最大值
        pd.set_option("display.max_rows", None, "display.max_columns", None)  # 控制输出全部行列
        self.current_row_index = 0  # 初始化当前行索引
        self.current_day = 0  # 初始化当前天数

    def step(self, action, obse, tt) -> Tuple[np.ndarray, float, bool, float]:
        # action转换,action在-1~1之间，需要转换为实际值
        data_DER = sio.loadmat('test_set.mat')  # 载入mat文件
        LOAD = sio.loadmat('load_date')

        WTP_1 = data_DER['fengji1']
        WTP_2 = data_DER['fengji2']
        WTP_3 = data_DER['fengji3']

        PVP_1 = data_DER['solar1']
        PVP_2 = data_DER['solar2']
        PVP_3 = data_DER['solar3']

        load_24_1h = LOAD['load_date']

        # 获取数据集的行数
        num_rows = min(50, PVP_1.shape[0])  # 限制在前100行

        # 使用当前行索引
        selected_row = self.current_row_index

        # 如果当前行索引超出范围，重置为0
        if self.current_row_index >= num_rows:
            self.current_row_index = 0
        else:
            self.current_row_index += 1

        # 从数据集中提取 PVP 值
        PVP = [0 for i in range(3)]
        PVP[0] = max(0, PVP_1[selected_row, tt])
        PVP[1] = max(0, PVP_2[selected_row, tt])
        PVP[2] = max(0, PVP_3[selected_row, tt])

        PVQ = [0 for i in range(3)]
        PVQ[0] = action[0][0] * np.sqrt(540 ** 2 - min(PVP[0], 540) ** 2)
        PVQ[1] = action[1][0] * np.sqrt(320 ** 2 - min(PVP[1], 320) ** 2)
        PVQ[2] = action[2][0] * np.sqrt(430 ** 2 - min(PVP[2], 430) ** 2)

        # 获取数据集的行数
        # num_rows = min(100, WTP_1.shape[0])  # 限制在前100行

        # 使用相同的行索引
        WTP = [0 for i in range(3)]
        WTP[0] = max(0, WTP_1[selected_row, tt])
        WTP[1] = max(0, WTP_2[selected_row, tt])
        WTP[2] = max(0, WTP_3[selected_row, tt])

        WTQ = [0 for _ in range(3)]
        WTQ[0] = action[0][1] * np.sqrt(600 ** 2 - min(WTP[0], 600) ** 2)
        WTQ[1] = action[1][1] * np.sqrt(500 ** 2 - min(WTP[1], 500) ** 2)
        WTQ[2] = action[2][1] * np.sqrt(550 ** 2 - min(WTP[2], 550) ** 2)

        ES_P = [0 for _ in range(3)]
        SOC_1 = obse[0][-1] * 400
        SOC_2 = obse[1][-1] * 300
        SOC_3 = obse[2][-1] * 250

        ES_P[0] = action[0][2] * np.sqrt(150 ** 2) if 0 <= action[0][2] * np.sqrt(150 ** 2) + SOC_1 <= 400 else (
            -SOC_1 if action[0][2] * np.sqrt(150 ** 2) + SOC_1 < 0 else 400 - SOC_1)
        ES_P[1] = action[1][2] * np.sqrt(120 ** 2) if 0 <= action[1][2] * np.sqrt(120 ** 2) + SOC_2 <= 300 else (
            -SOC_2 if action[1][2] * np.sqrt(120 ** 2) + SOC_2 < 0 else 300 - SOC_2)
        ES_P[2] = action[2][2] * np.sqrt(100 ** 2) if 0 <= action[2][2] * np.sqrt(100 ** 2) + SOC_3 <= 250 else (
            -SOC_3 if action[2][2] * np.sqrt(100 ** 2) + SOC_3 < 0 else 250 - SOC_3)

        SOC_1 = (SOC_1 + ES_P[0]) / 400
        SOC_2 = (SOC_2 + ES_P[1]) / 300
        SOC_3 = (SOC_3 + ES_P[2]) / 250

        # Define the constants
        n = 33
        n1 = 32
        isb = 1
        H = 31
        pr = 0.0001
        v_amp = 0

        B1 = np.array([
            [1, 2, 0.0922, 0.047j, 1, 0],
            [2, 3, 0.0493, 0.2511j, 1, 0],
            [3, 4, 0.366, 0.1864j, 1, 0],
            [4, 5, 0.3811, 0.1941j, 1, 0],
            [5, 6, 0.819, 0.707j, 1, 0],
            [6, 7, 0.1872, 0.6188j, 1, 0],
            [7, 8, 0.7114, 0.2351j, 1, 0],
            [8, 9, 1.0299, 0.74j, 1, 0],
            [9, 10, 1.044, 0.74j, 1, 0],
            [10, 11, 0.1966, 0.065j, 1, 0],
            [11, 12, 0.3744, 0.1238j, 1, 0],
            [12, 13, 1.468, 1.155j, 1, 0],
            [13, 14, 0.5416, 0.7129j, 1, 0],
            [14, 15, 0.591, 0.526j, 1, 0],
            [15, 16, 0.7463, 0.545j, 1, 0],
            [16, 17, 1.289, 1.721j, 1, 0],
            [17, 18, 0.732, 0.574j, 1, 0],
            [2, 19, 0.164, 0.1565j, 1, 0],
            [19, 20, 1.5042, 1.3554j, 1, 0],
            [20, 21, 0.4095, 0.4784j, 1, 0],
            [21, 22, 0.7089, 0.9373j, 1, 0],
            [3, 23, 0.4512, 0.3083j, 1, 0],
            [23, 24, 0.898, 0.7091j, 1, 0],
            [24, 25, 0.896, 0.7011j, 1, 0],
            [6, 26, 0.203, 0.1034j, 1, 0],
            [26, 27, 0.2842, 0.1447j, 1, 0],
            [27, 28, 1.059, 0.9337j, 1, 0],
            [28, 29, 0.8042, 0.7006j, 1, 0],
            [29, 30, 0.5075, 0.2585j, 1, 0],
            [30, 31, 0.9744, 0.963j, 1, 0],
            [31, 32, 0.3105, 0.3619j, 1, 0],
            [32, 33, 0.341, 0.5302j, 1, 0],
            [12, 22, 0.1, 0.1j, 1, 0],
            [8, 21, 0.1, 0.1j, 1, 0],
            [15, 9, 0.1, 0.1j, 1, 0],
            [25, 29, 0.1, 0.1j, 1, 0],
            [18, 33, 0.1, 0.1j, 1, 0]
        ], dtype=complex)

        # Node data matrix
        B2 = np.array([
            [1, 0, 0, 0, 1, 0],
            [2, 1, -100, -60, 1, 0],
            [3, 1, -90, -40, 1, 0],
            [4, 1, -120, -80, 1, 0],
            [5, 1, -60, -30, 1, 0],
            [6, 1, -60, -20, 1, 0],
            [7, 1, -200, -100, 1, 0],
            [8, 1, -200, -100, 1, 0],
            [9, 1, -60, -20, 1, 0],
            [10, 1, -60, -35, 1, 0],
            [11, 1, -45, -30, 1, 0],
            [12, 1, -60, -35, 1, 0],
            [13, 1, -60, -35, 1, 0],
            [14, 1, -120, -80, 1, 0],
            [15, 1, -60, -10, 1, 0],
            [16, 1, -60, -20, 1, 0],
            [17, 1, -60, -20, 1, 0],
            [18, 1, -90, -40, 1, 0],
            [19, 1, -90, -40, 1, 0],
            [20, 1, -90, -40, 1, 0],
            [21, 1, -90, -40, 1, 0],
            [22, 1, -90, -40, 1, 0],
            [23, 1, -90, -50, 1, 0],
            [24, 1, -420, -200, 1, 0],
            [25, 1, -420, -200, 1, 0],
            [26, 1, -60, -25, 1, 0],
            [27, 1, -60, -25, 1, 0],
            [28, 1, -60, -20, 1, 0],
            [29, 1, -120, -70, 1, 0],
            [30, 1, -200, -600, 1, 0],
            [31, 1, -150, -70, 1, 0],
            [32, 1, -210, -100, 1, 0],
            [33, 1, -60, -40, 1, 0]
        ], dtype=float)

        # Base values
        SBase = 10  # 10 MVA
        VBase = 12.66  # 12.66 kV
        ZBase = VBase ** 2 / SBase  # Ohm

        # switch_96 = sio.loadmat('E:/第一篇重构/python版本/MATD3_no_reconfigure/switch_96_15min.mat')
        # switch_status_96 = switch_96['expanded_sw_set']
        # indices = np.where(switch_status_96[:, tt] == 1)[0]
        # # 使用这些索引从矩阵 A 中提取行
        # B1 = B1[indices, :]
        B1 = B1[0:32, :]

        B1[:, 2] = B1[:, 2] / ZBase  # 将实际值换算成标幺值
        B1[:, 3] = B1[:, 3] / ZBase

        Y = np.zeros((n, n), dtype=complex)  # 生成全0复数矩阵

        for i in range(B1.shape[0]):  # 计算导纳矩阵
            p = int(B1[i, 0].real) - 1
            q = int(B1[i, 1].real) - 1
            Y[p, q] -= 1 / ((B1[i, 2] + B1[i, 3]) * B1[i, 4])
            Y[q, p] = Y[p, q]
            Y[p, p] += 1 / (B1[i, 2] + B1[i, 3]) + 0.5 * B1[i, 5]
            Y[q, q] += 1 / ((B1[i, 2] + B1[i, 3]) * B1[i, 4] ** 2) + 0.5 * B1[i, 5]

        G = Y.real
        B = Y.imag

        # Convert the actual values to per unit values

        B2[:, 2] = B2[:, 2] / 1000 / SBase  # Active power
        B2[:, 3] = B2[:, 3] / 1000 / SBase  # Reactive power
        # 计算有功和无功功率比例
        P_Level = B2[:, 2] / np.sum(B2[:, 2])
        Q_Level = B2[:, 3] / np.sum(B2[:, 3])

        P_Level = P_Level.reshape((33, 1))
        Q_Level = Q_Level.reshape((33, 1))

        Pd_total = (load_24_1h - 87.6667) / 64 * 500 / 1000 / SBase
        Qd_total = Pd_total * np.tan(np.arccos(0.85))

        # 计算24小时系统各节点负荷
        Pd = P_Level * Pd_total
        Qd = Q_Level * Qd_total

        B2[:, 2] = -Pd[:, tt]
        B2[:, 2] = -Qd[:, tt]

        PV1 = 6;
        PV2 = 13;
        PV3 = 27
        WT1 = 10;
        WT2 = 16;
        WT3 = 30
        ES1 = 4;
        ES2 = 15;
        ES3 = 29

        # Adjust for grid injections
        B2[PV1 - 1, 2] += PVP[0] / 10000
        B2[PV2 - 1, 2] += PVP[1] / 10000
        B2[PV3 - 1, 2] += PVP[2] / 10000
        B2[PV1 - 1, 3] += PVQ[0] / 10000
        B2[PV2 - 1, 3] += PVQ[1] / 10000
        B2[PV3 - 1, 3] += PVQ[2] / 10000

        B2[WT1 - 1, 2] += WTP[0] / 10000
        B2[WT2 - 1, 2] += WTP[1] / 10000
        B2[WT3 - 1, 2] += WTP[2] / 10000
        B2[WT1 - 1, 3] += WTQ[0] / 10000
        B2[WT2 - 1, 3] += WTQ[1] / 10000
        B2[WT3 - 1, 3] += WTQ[2] / 10000

        B2[ES1 - 1, 2] += ES_P[0] / 10000
        B2[ES2 - 1, 2] += ES_P[1] / 10000
        B2[ES3 - 1, 2] += ES_P[2] / 10000

        # 初始化迭代次数
        Times = 1

        # 其他相关变量初始化
        Q = 0
        PQV = 0
        x = 1.655  # x1 + x2；x1为定子漏抗，x2为转子漏抗
        xp = 18.8  # xc * xm / (xc - xm)；xc为机端并联电容器电抗，xm为激磁电抗
        h = 0
        Ig = 0.01
        PVU = [0, 0]
        PI = 0

        OrgS = np.zeros((64, 1))  # Initialize OrgS matrix

        for i in range(n):  # Adjust the index for 0-based Python indexing
            if i != 0 and B2[i, 1] == 1:  # Adjusted index for 0-based, PQ node handling
                h += 1
                for j in range(n):
                    OrgS[2 * h - 2, 0] += B2[i, 4] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) + B2[i, 5] * (
                            G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])
                    OrgS[2 * h - 1, 0] += B2[i, 5] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) - B2[i, 4] * (
                            G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])
        for i in range(n):  # PV node handling
            if i != 0 and B2[i, 1] == 2:  # Adjusted index for 0-based
                h += 1
                for j in range(n):
                    OrgS[2 * h - 2, 0] += B2[i, 4] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) + B2[i, 5] * (
                            G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])
                    OrgS[2 * h - 1, 0] += B2[i, 5] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) - B2[i, 4] * (
                            G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])

        DetaS = np.zeros((2 * n - 2, 1))
        h = 0
        for i in range(n):  # Adjust the index for 0-based Python indexing, handling PQ nodes
            if i != 0 and B2[i, 1] == 1:  # Adjusted index for 0-based
                h += 1
                DetaS[2 * h - 2, 0] = B2[i, 2] - OrgS[2 * h - 2, 0]
                DetaS[2 * h - 1, 0] = B2[i, 3] - OrgS[2 * h - 1, 0]

        t = 0
        for i in range(n):  # Handling PV nodes
            if i != 0 and B2[i, 1] == 2:  # Adjusted index for 0-based
                h += 1
                t += 1
                DetaS[2 * h - 2, 0] = B2[i, 2] - OrgS[2 * h - 2, 0]
                DetaS[2 * h - 1, 0] = PVU[t - 1, 0] ** 2 + PVU[t - 1, 1] ** 2 - B2[i, 4] ** 2 - B2[i, 5] ** 2

        I = np.zeros((n - 1, 1), dtype=complex)
        h = 0

        for i in range(n):
            if i != 0:
                h += 1
                # Python's complex number manipulation and the complex sqrt(-1) which is '1j'
                numerator = OrgS[2 * h - 2, 0] - OrgS[2 * h - 1, 0] * 1j
                denominator = np.conj(B2[i, 4] + B2[i, 5] * 1j)
                I[h - 1, 0] = numerator / denominator

        Jacbi = np.zeros((2 * (n - 1), 2 * (n - 1)))

        h = 0
        k = 0

        # Handling PQ nodes
        for i in range(n):
            if B2[i, 1] == 1:  # Adjusted index for 0-based, check if node is PQ
                h += 1
                for j in range(n):
                    if j != 0:  # Skipping the slack bus
                        k += 1
                        if i == j:  # Diagonal elements
                            Jacbi[2 * h - 2, 2 * k - 2] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5] + np.imag(
                                I[h - 1, 0])
                            Jacbi[2 * h - 2, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5] + np.real(I[h - 1, 0])
                            Jacbi[2 * h - 1, 2 * k - 2] = -Jacbi[2 * h - 2, 2 * k - 1] + 2 * np.real(I[h - 1, 0])
                            Jacbi[2 * h - 1, 2 * k - 1] = Jacbi[2 * h - 2, 2 * k - 2] - 2 * np.imag(I[h - 1, 0])
                        else:  # Off-diagonal elements
                            Jacbi[2 * h - 2, 2 * k - 2] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5]
                            Jacbi[2 * h - 2, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5]
                            Jacbi[2 * h - 1, 2 * k - 2] = -Jacbi[2 * h - 2, 2 * k - 1]
                            Jacbi[2 * h - 1, 2 * k - 1] = Jacbi[2 * h - 2, 2 * k - 2]
                        if k == (n - 1):  # Reset k at the end of the row
                            k = 0

        # Handling PV nodes
        h = 0  # Reset h if necessary, depending on whether it should continue from previous value
        for i in range(n):
            if B2[i, 1] == 2:  # Check if node is PV
                h += 1
                for j in range(n):
                    if j != 0:
                        k += 1
                        if i == j:  # Diagonal elements
                            Jacbi[2 * h - 2, 2 * k - 2] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5] + np.imag(
                                I[h - 1, 0])
                            Jacbi[2 * h - 2, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5] + np.real(I[h - 1, 0])
                            Jacbi[2 * h - 1, 2 * k - 2] = 2 * B2[i, 5]
                            Jacbi[2 * h - 1, 2 * k - 1] = 2 * B2[i, 4]
                        else:  # Off-diagonal elements
                            Jacbi[2 * h - 2, 2 * k - 2] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5]
                            Jacbi[2 * h - 2, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5]
                            Jacbi[2 * h - 1, 2 * k - 2] = 0
                            Jacbi[2 * h - 1, 2 * k - 1] = 0
                        if k == (n - 1):  # Reset k at the end of the row
                            k = 0

        DetaU = np.zeros((2 * (n - 1), 1))
        DetaU = np.linalg.solve(Jacbi, DetaS)

        while np.max(np.abs(DetaU)) > pr:
            OrgS = np.zeros((2 * n - 2, 1))
            h = 0
            j = 0
            for i in range(n):  # 对PQ节点的处理
                if i != isb - 1 and B2[i, 1] == 1:
                    h += 1
                    for j in range(n):
                        OrgS[2 * h - 1 - 1, 0] += B2[i, 4] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) + B2[i, 5] * (
                                G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])  # Pi
                        OrgS[2 * h - 1, 0] += B2[i, 5] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) - B2[i, 4] * (
                                G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])  # Qi

            for i in range(n):  # 对PV节点的处理
                if i != isb - 1 and B2[i, 1] == 2:
                    h += 1
                    for j in range(n):
                        OrgS[2 * h - 1 - 1, 0] += B2[i, 4] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) + B2[i, 5] * (
                                G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])
                        OrgS[2 * h - 1, 0] += B2[i, 5] * (G[i, j] * B2[j, 4] - B[i, j] * B2[j, 5]) - B2[i, 4] * (
                                G[i, j] * B2[j, 5] + B[i, j] * B2[j, 4])

            # print("修正后的迭代计算PQ、PV节点参数：\n", OrgS)

            DetaS = np.zeros((2 * n - 2, 1))
            h = 0
            for i in range(n):  # 对PQ节点的处理
                if i != isb - 1 and B2[i, 1] == 1:
                    h += 1
                    DetaS[2 * h - 1 - 1, 0] = B2[i - 1, 2] - OrgS[2 * h - 1 - 1, 0]
                    DetaS[2 * h - 1, 0] = B2[i - 1, 3] - OrgS[2 * h - 1, 0]

            t = 0
            for i in range(n):  # 对PV节点的处理
                if i != isb - 1 and B2[i, 1] == 2:
                    h += 1
                    t += 1
                    DetaS[2 * h - 1 - 1, 0] = B2[i, 2] - OrgS[2 * h - 1 - 1, 0]
                    DetaS[2 * h - 1, 0] = PVU[t - 1, 0] ** 2 + PVU[t - 1, 1] ** 2 - B2[i, 4] ** 2 - B2[i, 5] ** 2

            # print("修正后的迭代计算PQ、PV节点不平衡量：\n", DetaS)

            I = np.zeros((n - 1, 1), dtype=complex)
            h = 0
            for i in range(n):
                if i != isb - 1:
                    h += 1
                    I[h - 1, 0] = (OrgS[2 * h - 1 - 1, 0] - OrgS[2 * h - 1, 0] * 1j) / np.conj(B2[i, 4] + B2[i, 5] * 1j)

            # print("I：\n", I)

            Jacbi = np.zeros((2 * n - 2, 2 * n - 2))
            h = 0
            k = 0
            for i in range(n):  # 对PQ节点的处理
                if B2[i, 1] == 1:
                    h += 1
                    for j in range(n):
                        if j != isb - 1:
                            k += 1
                            if i == j:  # 对角元素的处理
                                Jacbi[2 * h - 1 - 1, 2 * k - 1 - 1] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5] + I[
                                    h - 1, 0].imag
                                Jacbi[2 * h - 1 - 1, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5] + I[
                                    h - 1, 0].real
                                Jacbi[2 * h - 1, 2 * k - 1 - 1] = -Jacbi[2 * h - 1 - 1, 2 * k - 1] + 2 * I[
                                    h - 1, 0].real
                                Jacbi[2 * h - 1, 2 * k - 1] = Jacbi[2 * h - 1 - 1, 2 * k - 1 - 1] - 2 * I[h - 1, 0].imag
                            else:  # 非对角元素的处理
                                Jacbi[2 * h - 1 - 1, 2 * k - 1 - 1] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5]
                                Jacbi[2 * h - 1 - 1, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5]
                                Jacbi[2 * h - 1, 2 * k - 1 - 1] = -Jacbi[2 * h - 1 - 1, 2 * k - 1]
                                Jacbi[2 * h - 1, 2 * k - 1] = Jacbi[2 * h - 1 - 1, 2 * k - 1 - 1]

                            if k == (n - 1):  # 将用于内循环的指针置于初始值，以确保雅可比矩阵换行
                                k = 0

            k = 0
            for i in range(n):  # 对PV节点的处理
                if B2[i, 1] == 2:
                    h += 1
                    for j in range(n):
                        if j != isb - 1:
                            k += 1
                            if i == j:  # 对角元素的处理
                                Jacbi[2 * h - 1 - 1, 2 * k - 1 - 1] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5] + I[
                                    h - 1, 0].imag
                                Jacbi[2 * h - 1 - 1, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5] + I[
                                    h - 1, 0].real
                                Jacbi[2 * h - 1, 2 * k - 1 - 1] = 2 * B2[i, 5]
                                Jacbi[2 * h - 1, 2 * k - 1] = 2 * B2[i, 4]
                            else:  # 非对角元素的处理
                                Jacbi[2 * h - 1 - 1, 2 * k - 1 - 1] = -B[i, j] * B2[i, 4] + G[i, j] * B2[i, 5]
                                Jacbi[2 * h - 1 - 1, 2 * k - 1] = G[i, j] * B2[i, 4] + B[i, j] * B2[i, 5]
                                Jacbi[2 * h - 1, 2 * k - 1 - 1] = 0
                                Jacbi[2 * h - 1, 2 * k - 1] = 0

                            if k == (n - 1):  # 将用于内循环的指针置于初始值，以确保雅可比矩阵换行
                                k = 0

            # print("修正后的雅克比矩阵：\n", Jacbi)
            DetaU = np.zeros((2 * n - 2, 1))
            DetaU = np.linalg.lstsq(Jacbi, DetaS, rcond=None)[0]

            j = 0
            for i in range(n):  # 对PQ节点处理
                if B2[i, 1] == 1:
                    j += 1
                    B2[i, 4] += DetaU[2 * j - 1, 0]
                    B2[i, 5] += DetaU[2 * j - 2, 0]

            for i in range(n):  # 对PV节点的处理
                if B2[i, 1] == 2:
                    j += 1
                    B2[i, 3] += DetaU[2 * j - 1, 0]
                    B2[i, 4] += DetaU[2 * j - 2, 0]

            for i in range(n):  # 对PQ(V)节点的处理
                if i + 1 == PQV:
                    B2[i, 1] = 3

            for i in range(n):  # 对PI节点的处理
                if i + 1 == PI:
                    B2[i, 1] = 4

            Times += 1  # 迭代次数加1

            # print("修正后的节点电压：\n", B2[:, 4])

            # 创建Sb，用于存储平衡节点功率
        Sb = 0
        for i in range(n):
            if i == isb - 1:
                for j in range(n):
                    Sb += (B2[i, 4] + 1j * B2[i, 5]) * np.conj(Y[i, j]) * np.conj(B2[j, 4] + 1j * B2[j, 5])

        # print("初始平衡节点功率：", Sb)

        Real_Sb = 10000 * np.real(Sb)
        Ploss = 0
        for i in range(n):
            for j in range(n):
                Ploss += B2[i, 4] * B2[j, 4] * (
                        G[i, j] * np.cos(B2[i, 5] - B2[j, 5]) + B[i, j] * np.sin(B2[i, 5] - B2[j, 5]))

        V_bus = np.sqrt(B2[:, 4] ** 2 + B2[:, 5] ** 2)  # 33列
        V_busmax = np.max(V_bus)
        V_busmin = np.min(V_bus)
        gridD = np.real(Sb) * 10
        dd_v = 0
        for i in range(n):
            if V_bus[i] > 1.04:
                dd_v += np.abs(V_bus[i] - 1.04) + 0.3
            if V_bus[i] < 0.96:
                dd_v += np.abs(V_bus[i] - 0.96) + 0.3
        # print(V_bus)
        penalty1 = -5 * dd_v
        penalty2 = 300 * Ploss
        # penalty2=0
        R = penalty1 - penalty2

        obs1 = np.append(np.append(V_bus[0:11], np.array(PVP[0]) / 100), np.array(WTP[0]) / 100)
        obs1 = np.append(obs1, np.array(SOC_1))
        obs2 = np.append(np.append(V_bus[11:22], np.array(PVP[1]) / 100), np.array(WTP[1]) / 100)
        obs2 = np.append(obs2, np.array(SOC_2))
        obs3 = np.append(np.append(V_bus[22:33], np.array(PVP[2]) / 100), np.array(WTP[2]) / 100)
        obs3 = np.append(obs3, np.array(SOC_3))

        obs = [obs1, obs2, obs3]

        SOC_ES = np.append(np.append(SOC_1, SOC_2), SOC_3)

        if tt >= 24 - 1:
            IsDone = True
        else:
            IsDone = False

        return obs, R, IsDone, PVQ, WTQ, ES_P, SOC_ES, V_bus, penalty2

    def reset(self, episode: int, day: int):
        self.current_row_index = 0  # 每天重置行索引
        self.current_day = day  # 更新当前天数
        self.t = 0
        # 更新数据集索引
        if episode < 1000:
            self.dataset_index = 1
        elif episode < 2000:
            self.dataset_index = 2
        else:
            self.dataset_index = 3

        obs1 = np.array([1 for _ in range(11 + 3)])
        obs2 = np.array([1 for _ in range(11 + 3)])
        obs3 = np.array([1 for _ in range(11 + 3)])
        obs = [obs1, obs2, obs3]

        return obs
