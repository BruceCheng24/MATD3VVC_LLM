import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat  # 用于保存 .mat 文件

from IEEE33new import IEEE33_ENV
from matd3 import MATD3
from replay_buffer import ReplayBuffer
# 训练程序

# Hyperparameters Setting for MADDPG in MPE environment
class Arguments():
    def __init__(self):
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
        self.noise_decay_steps = 72000 * 2  # How many steps before the noise_std decays to the minimum
        self.use_noise_decay = True  # Whether to decay the noise_std
        self.lr_a = 0.0001  # Learning rate of the actor
        self.lr_c = 0.0001  # Learning rate of the critic
        self.gamma = 0.98  # Discount factor
        self.tau = 0.001  # Softly update the target network
        self.use_orthogonal_init = True  # Whether to use orthogonal initialization
        self.use_grad_clip = True  # Whether to use gradient clip
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates


# 程序入口
if __name__ == "__main__":
    # 指定GPU
    torch.cuda.set_device(0)

    # 创建参数对象
    args = Arguments()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps
    args.N = 3  # 环境中智能体的数量

    # 创建环境
    env = IEEE33_ENV()
    env_evaluate = IEEE33_ENV()  # 创建评估环境

    # 获取环境的状态空间和动作空间
    state_dim = 347
    action_dim = 1
    args.obs_dim_total = 347
    args.obs_dim_n = [11 + 3, 11 + 3, 11 + 3]  # observation dimensions of N agents
    args.action_dim_n = [3, 3, 3]  # action dimensions of N agents
    print("obs_dim_n:", args.obs_dim_n)
    print("action_dim_n:", args.action_dim_n)

    # 设置随机种子
    np.random.seed(random.randint(0, 1000))
    torch.manual_seed(random.randint(0, 1000))

    # 创建智能体
    agent_n = [MATD3(args, i, torch.device('cuda:0')) for i in range(args.N)]

    # 创建经验池
    replay_buffer = ReplayBuffer(args, torch.device('cuda:0'))

    evaluate_rewards = []  # 记录每次评估的奖励
    total_steps = 0  # 记录总步数
    noise_std = args.noise_std_init  # 记录噪声标准差

    # 训练
    # evaluate_policy()  # 先评估一次

    # 绘图要记录的数据
    rewards_history = []
    loss_record = []
    total_train_time = 0  # 总训练时间
    voltage_before = []  # 记录电压控制前的结果
    voltage_after = []  # 记录电压控制后的结果

    # 初始化一个变量来记录所有步骤的总时间
    total_decision_time = 0
    total_decision_steps = 0

    for episode in range(args.max_episodes):
        if episode % 1000 == 0:
            print(f"Loading new dataset for episodes {episode} to {episode+1000}")
            # 加载新数据集的逻辑
            env = IEEE33_ENV()  # 重新创建环境以模拟加载新数据集
        obs_n = env.reset(episode)
        rewards = 0
        LOSS_24step = 0
        start_time = time.time()  # 记录episode开始时间
        episode_voltage_before = []
        episode_voltage_after = []

        for step in range(24):
            step_start_time = time.time()  # 记录step开始时间

            a_n = [agent.choose_action(obs, noise_std) for agent, obs in zip(agent_n, obs_n)]
            obs_next_n, _r_n, _done_n, s_PVQ, s_WTQ, s_ES_P, s_SOC_ES, s_V_bus, s_Ploss = env.step(a_n, obs_n,step)  # 步进
            episode_voltage_before.append(s_V_bus)
            r_n = np.array([_r_n for _ in range(args.N)])
            done_n = np.array([_done_n for _ in range(args.N)])
            obs_n_repeat = np.array([obs_n[id] for id in range(args.N)], dtype=object)
            replay_buffer.store_transition(obs_n_repeat, a_n, r_n, obs_next_n, done_n)  # 存储经验
            obs_n = obs_next_n
            episode_voltage_after.append(obs_n[0][11:])  # 假设电压数据在obs_n[0]的索引11之后
            rewards += r_n[0]
            LOSS_24step += s_Ploss

            # 记录step结束时间并计算决策时间
            step_end_time = time.time()
            step_decision_time = step_end_time - step_start_time
            total_decision_time += step_decision_time
            total_decision_steps += 1

            # 这里是为了让噪声标准差逐渐减小，让智能体的动作逐渐收敛
            if args.use_noise_decay:
                noise_std = noise_std - args.noise_std_decay if noise_std - args.noise_std_decay > args.noise_std_min \
                    else args.noise_std_min

            if (replay_buffer.current_size > args.batch_size) and (step % 8 == 0):  # 当经验池中的经验数量大于batch_size时，开始训练
                for agent_id in range(args.N):
                    agent_n[agent_id].train(replay_buffer, agent_n)

        end_time = time.time()  # 记录episode结束时间
        episode_time = end_time - start_time
        total_train_time += episode_time  # 累加训练时间

        # 保存电压控制前后的结果
        voltage_before.append(episode_voltage_before)
        voltage_after.append(episode_voltage_after)

        # 输出训练过程中的步长、reward、噪声标准差等信息, 实时绘制reward变化曲线
        loss_record.append(LOSS_24step)
        rewards_history.append(rewards)
        if rewards_history.__len__() % 1 == 0:
            print("episode:{}, noise_std:{}, reward:{}, loss:{}, episode_time:{}".format(
                rewards_history.__len__(), noise_std, rewards, LOSS_24step, episode_time))

    # 计算平均决策时间
    avg_decision_time = total_decision_time / total_decision_steps
    print("Average decision time per step: {:.6f} seconds".format(avg_decision_time))

    # 平均计算时间
    avg_time_per_episode = total_train_time / args.max_episodes
    print("Average time per episode: {}".format(avg_time_per_episode))

    # 保存配电网数据
    data = {
        'rewards_history': rewards_history,
        'loss_record': loss_record,
        'avg_time_per_episode': avg_time_per_episode,
        'voltage_before': voltage_before,
        'voltage_after': voltage_after,
        'avg_decision_time': avg_decision_time  # 保存平均决策时间
    }
    savemat('training_data.mat', data)

    plt.figure()
    plt.plot(np.arange(len(rewards_history)), rewards_history)
    plt.xlabel('Total episodes')
    plt.ylabel('Episode reward')
    plt.show()

    plt.figure()
    plt.plot(np.arange(len(loss_record)), loss_record)
    plt.xlabel('Total episodes')
    plt.ylabel('VUF_24hours')
    plt.show()

    plt.figure()
    rewards_history_np = np.array(rewards_history)
    loss_record_np = np.array(loss_record)
    plt.plot(np.arange(len(loss_record)), -rewards_history_np - loss_record_np)
    plt.xlabel('Total episodes')
    plt.ylabel('VD')
    plt.show()

    # 保存模型
    for agent_id in range(args.N):
        agent_n[agent_id].save_model('BFS', 'MATD3', 2, 3000, agent_id)

    print("Training over.")

















