from IEEE33test50day import IEEE33_ENV
# from matd3 import MATD3
from matd3 import MATD3
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from networks import Actor
import time
import os
# 测试程序(50天)
class Arguments():
    def __init__(self):
        # Environment setting
        self.max_train_steps = 10000  # Maximum training steps
        self.max_episodes = 3000
        self.episode_limit = 100  # Maximum steps for each episode
        self.evaluate_freq = 5000  # The number of evaluating episodes
        self.evaluate_times = 3  # Evaluate times
        self.max_action = 1.0  # Max action
        self.algorithm = "MATD3"  # MADDPG
        self.buffer_size = 72000  # The capacity of the replay buffer
        self.batch_size = 256  # Batch size
        self.hidden_dim = 128  # Hidden dimension of the network
        self.noise_std_init = 0.2  # The std of Gaussian noise for exploration
        self.noise_std_min = 0.05  # The minimum std of Gaussian noise for exploration
        self.noise_decay_steps = 72000*2  # How many steps before the noise_std decays to the minimum
        self.use_noise_decay = True  # Whether to decay the noise_std
        self.lr_a = 0.0001  # Learning rate of the actor
        self.lr_c = 0.0001  # Learning rate of the critic
        self.lr_tem = 0.0003
        self.gamma = 0.98  # Discount factor
        self.tau = 0.001  # Softly update the target network
        self.use_orthogonal_init = True  # Whether to use orthogonal initialization
        self.use_grad_clip = True  # Whether to use gradient clip
        self.autotune = True

        # TD3MORE(暂未使用)
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates


# 修改后的sim函数，用于保存50天的结果
def sim():
    all_days_results = []

    args = Arguments()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.N = 3  # 环境中智能体的数量
    args.obs_dim_n = [11 + 3, 11 + 3, 11 + 3]  # 每个智能体的观测维度
    args.action_dim_n = [3, 3, 3]  # 每个智能体的动作维度
    noise_std = args.noise_std_min  # 记录噪声标准差

    agent_good_n = [MATD3(args, i, torch.device('cuda:0')) for i in range(args.N)]
    for s_agent_id in range(args.N):
        agent_good_n[s_agent_id].load_model('BFS', 'MATD3', 2, 3000, s_agent_id)

    s_env = IEEE33_ENV()

    for day in range(50):  # 遍历50天
        Voltage = []
        all_PVQ = []
        all_WTQ = []
        all_ESP = []
        all_ES_SOC = []
        all_Ploss = []  # 添加一个列表保存每一天的网损结果

        for episode in range(args.max_episodes):
             if episode % 1000 == 0:
                 print(f"加载新数据集：{episode}到{episode+1000}集")

        obs_n = s_env.reset(episode, day)
        rewards = 0
        LOSS_24step = 0

        for step in range(24):
                a_n = [agent.choose_action(obs, noise_std) for agent, obs in zip(agent_good_n, obs_n)]
                obs_next_n, _r_n, _done_n, s_PVQ, s_WTQ, s_ES_P, s_SOC_ES, s_V_bus, s_Ploss = s_env.step(a_n, obs_n, step)  # 步进
                r_n = np.array([_r_n for _ in range(args.N)])
                obs_n = obs_next_n
                rewards += r_n[0]
                LOSS_24step += s_Ploss
                all_PVQ.append(s_PVQ)
                all_WTQ.append(s_WTQ)
                all_ESP.append(s_ES_P)
                all_ES_SOC.append(s_SOC_ES)
                Voltage.append(s_V_bus)
                all_Ploss.append(s_Ploss)  # 保存每一步的网损

        # 保存当前天的结果
        day_result = {
            'day': day + 1,
            'all_PVQ': all_PVQ,
            'all_WTQ': all_WTQ,
            'Voltage': Voltage,
            'ESS_24_P': all_ESP,
            'ESS_24_SOC': all_ES_SOC,
            'Loss_24step': all_Ploss  # 将网损结果添加到字典中
        }
        all_days_results.append(day_result)

    # 将所有天数的结果保存为.mat文件
    savemat('50_days_simulation_results.mat', {'results': all_days_results})

    print("50天的仿真结果已保存到'50_days_simulation_results.mat'文件中")

    # 可选：打印每一天的结果维度信息
    # for day_result in all_days_results:
    #     print(f"\n第{day_result['day']}天的结果：")
    #     print("all_PVQ 维度：")
    #     print("外部列表长度：", len(day_result['all_PVQ']))
    #     if day_result['all_PVQ'] and isinstance(day_result['all_PVQ'][0], list):
    #         print("内部列表长度：", [len(inner_list) for inner_list in day_result['all_PVQ']])
    #
    #     print("all_WTQ 维度：")
    #     print("外部列表长度：", len(day_result['all_WTQ']))
    #     if day_result['all_WTQ'] and isinstance(day_result['all_WTQ'][0], list):
    #         print("内部列表长度：", [len(inner_list) for inner_list in day_result['all_WTQ']])
    #
    #     print("all_ESP 维度：")
    #     print("外部列表长度：", len(day_result['all_ESP']))
    #     if day_result['all_ESP'] and isinstance(day_result['all_ESP'][0], list):
    #         print("内部列表长度：", [len(inner_list) for inner_list in day_result['all_ESP']])
    #
    #     print("all_SOC 维度：")
    #     print("外部列表长度：", len(day_result['ESS_24_SOC']))
    #     if day_result['ESS_24_SOC'] and isinstance(day_result['ESS_24_SOC'][0], list):
    #         print("内部列表长度：", [len(inner_list) for inner_list in day_result['ESS_24_SOC']])
    #
    #     print("\nVoltage_a 维度：")
    #     print("外部列表长度：", len(day_result['Voltage']))
    #     if day_result['Voltage'] and isinstance(day_result['Voltage'][0], list):
    #         print("内部列表长度：", [len(inner_list) for inner_list in day_result['Voltage']])
    #
    #
    #
    # # 准备数据
    # data = list(zip(*all_PVQ))  # 转换数据以便于绘制
    #
    # # 绘制图表
    # plt.figure(figsize=(10, 6))
    # for i in range(3):
    #     plt.plot(data[i], label=f'Line {i + 1}')
    #
    # # 添加图例
    # #plt.legend()
    #
    # # 添加标题和轴标签
    # plt.title("Plot of all_PVQ")
    # plt.xlabel("Step")
    # plt.ylabel("Value")
    #
    # # 显示图表
    # plt.show()
    #
    # # 准备数据
    # data_ESP = list(zip(*all_ESP))  # 转换数据以便于绘制
    #
    # # 绘制图表
    # plt.figure(figsize=(10, 6))
    # for i in range(3):
    #     plt.plot(data_ESP[i], label=f'Line {i + 1}')
    #
    # # 添加图例
    # # plt.legend()
    #
    # # 添加标题和轴标签
    # plt.title("Plot of all_ESP")
    # plt.xlabel("Step")
    # plt.ylabel("Value")
    #
    # # 显示图表
    # plt.show()
    #
    # # 准备数据
    # data_SOC = list(zip(*all_ES_SOC))  # 转换数据以便于绘制
    #
    # # 绘制图表
    # plt.figure(figsize=(10, 6))
    # for i in range(3):
    #     plt.plot(data_SOC[i], label=f'Line {i + 1}')
    #
    # # 添加图例
    # # plt.legend()
    #
    # # 添加标题和轴标签
    # plt.title("Plot of all_SOC")
    # plt.xlabel("Step")
    # plt.ylabel("Value")
    #
    # # 显示图表
    # plt.show()
    #
    #
    # data_va = list(zip(*Voltage))# 转换数据以便于绘制
    #
    # # 创建一个包含三个子图的图形，子图纵向排列
    # fig, axs = plt.subplots(3, 1, figsize=(7, 18))  # 调整figsize以适应纵向排列
    #
    # # 绘制第一个子图
    # axs[0].plot(data_va)
    # #axs[0].set_title("Plot of data_va")
    #
    #
    # # 显示整个图形
    # plt.tight_layout()
    # plt.show()

sim()
