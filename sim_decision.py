from IEEE33new import IEEE33_ENV
# from matd3 import MATD3
from matd3 import MATD3
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from networks import Actor
import time
import os

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


# 模拟运行训练好的模型
def sim():
    Voltage=[]
    all_PVQ=[]
    all_WTQ=[]
    all_ESP=[]
    all_ES_SOC=[]

    args = Arguments()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.N = 3  # 环境中智能体的数量
    args.obs_dim_n = [11 + 3, 11 + 3, 11 + 3]  # observation dimensions of N agents
    args.action_dim_n = [3, 3, 3]  # action dimensions of N agents
    noise_std = args.noise_std_min  # 记录噪声标准差

    agent_good_n = [MATD3(args, i, torch.device('cuda:0')) for i in range(args.N)]
    for s_agent_id in range(args.N):
        agent_good_n[s_agent_id].load_model('BFS', 'MATD3', 2, 3000, s_agent_id)

    s_env = IEEE33_ENV()
    obs_n = s_env.reset()
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

    # print(all_PVQ)
    # print(Voltage_a)
    # print(Voltage_b)
    # print(Voltage_c)
    # print(VUF_123bus)

    # 将变量组织到一个字典中
    data_dict = {
        'all_PVQ': all_PVQ,
        'all_WTQ': all_WTQ,
        'Voltage': Voltage,
        'ESS_24_P': all_ESP,
        'ESS_24_SOC': all_ES_SOC
    }

    # 使用savemat保存为MAT文件
    savemat('have_agent_data_file.mat', data_dict)

    print("all_PVQ dimensions:")
    print("Outer list length:", len(all_PVQ))
    if all_PVQ and isinstance(all_PVQ[0], list):
        print("Inner lists lengths:", [len(inner_list) for inner_list in all_PVQ])

    print("all_WTQ dimensions:")
    print("Outer list length:", len(all_WTQ))
    if all_WTQ and isinstance(all_WTQ[0], list):
        print("Inner lists lengths:", [len(inner_list) for inner_list in all_WTQ])

    print("all_ESP dimensions:")
    print("Outer list length:", len(all_ESP))
    if all_ESP and isinstance(all_ESP[0], list):
        print("Inner lists lengths:", [len(inner_list) for inner_list in all_ESP])

    print("all_SOC dimensions:")
    print("Outer list length:", len(all_ES_SOC))
    if all_ES_SOC and isinstance(all_ES_SOC[0], list):
        print("Inner lists lengths:", [len(inner_list) for inner_list in all_ES_SOC])

    print("\nVoltage_a dimensions:")
    print("Outer list length:", len(Voltage))
    if Voltage and isinstance(Voltage[0], list):
        print("Inner lists lengths:", [len(inner_list) for inner_list in Voltage])



    # 准备数据
    data = list(zip(*all_PVQ))  # 转换数据以便于绘制

    # 绘制图表
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(data[i], label=f'Line {i + 1}')

    # 添加图例
    #plt.legend()

    # 添加标题和轴标签
    plt.title("Plot of all_PVQ")
    plt.xlabel("Step")
    plt.ylabel("Value")

    # 显示图表
    plt.show()

    # 准备数据
    data_ESP = list(zip(*all_ESP))  # 转换数据以便于绘制

    # 绘制图表
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(data_ESP[i], label=f'Line {i + 1}')

    # 添加图例
    # plt.legend()

    # 添加标题和轴标签
    plt.title("Plot of all_ESP")
    plt.xlabel("Step")
    plt.ylabel("Value")

    # 显示图表
    plt.show()

    # 准备数据
    data_SOC = list(zip(*all_ES_SOC))  # 转换数据以便于绘制

    # 绘制图表
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(data_SOC[i], label=f'Line {i + 1}')

    # 添加图例
    # plt.legend()

    # 添加标题和轴标签
    plt.title("Plot of all_SOC")
    plt.xlabel("Step")
    plt.ylabel("Value")

    # 显示图表
    plt.show()


    data_va = list(zip(*Voltage))# 转换数据以便于绘制

    # 创建一个包含三个子图的图形，子图纵向排列
    fig, axs = plt.subplots(3, 1, figsize=(7, 18))  # 调整figsize以适应纵向排列

    # 绘制第一个子图
    axs[0].plot(data_va)
    #axs[0].set_title("Plot of data_va")


    # 显示整个图形
    plt.tight_layout()
    plt.show()



def sim_generalization():
    generalize_reward = []

    args = Arguments()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.N = 4  # 环境中智能体的数量
    args.obs_dim_n = [3 * 26 + 3 + 2 + 2, 3 * 33 + 3 + 2 + 2, 3 * 36 + 3 + 2 + 2,3 * 18 + 3 + 2 + 2]  # observation dimensions of N agents
    args.action_dim_n = [5, 5, 5, 5]  # action dimensions of N agents
    noise_std = args.noise_std_min  # 记录噪声标准差

    agent_good_n = [MATD3(args, i, torch.device('cuda:0')) for i in range(args.N)]
    for s_agent_id in range(args.N):
        agent_good_n[s_agent_id].load_model('BFS', 'MATD3', 2, 1000, s_agent_id)
    #for agent_id in range(4):  # Only for the first and second agents
    #    # Reinitialize the Actor for each agent
    #    agent_good_n[agent_id].actor = Actor_MASAC(args, agent_id, torch.device('cuda:0')).to(torch.device('cuda:0'))

    s_env = three_phase_BFS_ENV()
    for episode in range(50):
        s_obs_n = s_env.reset()
        rrr=0
        for step in range(24):
            s_a_n = [agent.choose_action(s_obs, noise_std) for agent, s_obs in zip(agent_good_n, s_obs_n)]
            s_obs_next_n, s_r_n, s_done_n, s_PVQ, s_VUF_step, s_result_VUF, s_ES_P, s_SOC_ES = s_env.step(s_a_n, s_obs_n,step)  # 步进
            s_obs_n = s_obs_next_n
            rrr=rrr+s_r_n
        generalize_reward.append(rrr)
        print(rrr)

# 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(generalize_reward)

    # 添加标题和轴标签
    plt.title("Reward")
    plt.xlabel("Validate day")
    plt.ylabel("Value")
    # 显示图表
    plt.show()

sim()
#sim_generalization()

