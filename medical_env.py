import numpy as np
import matplotlib.pyplot as plt
import os
from gymnasium.spaces import Box, Discrete
from xuance.environment import RawMultiAgentEnv

class MedicalMECEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(MedicalMECEnv, self).__init__()
        # 处理传入配置，兼容 Namespace 和 dict
        if not isinstance(env_config, dict):
            env_config = vars(env_config)
            
        self.env_id = env_config.get('env_id', 'MedicalMEC-v0')
        self.n_uavs = env_config.get("n_uavs", 4)
        self.n_users = env_config.get("n_users", 20)
        self.map_size = env_config.get("map_size", 1000.0)
        self.max_episode_steps = env_config.get("max_episode_steps", 200)
        self.fig_save_path = env_config.get("log_dir", "./results")
        # --- [新增] 论文 Eq. 24-30 的系统参数 ---
        self.alpha_H = 10.0      # Eq.24 高优先级效用系数
        self.Phi_H = 20.0        # Eq.24 高优先级失败惩罚
        self.Phi_L = 10.0        # Eq.25 低优先级完成奖励
        self.mu_decay = 0.5      # Eq.25 低优先级衰减系数
        
        # --- 1. 智能体定义 ---
        self.num_agents = self.n_users
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
        # --- 2. 动作与观测空间 ---
        # Action: 0(Local), 1~M(UAVs), M+1(GEC)
        self.n_actions = self.n_uavs + 2
        self.action_space = {agent: Discrete(self.n_actions) for agent in self.agents}
        
        # Observation: 维度 = Task(4) + UserPos(2) + UAVPos(2*M) + Dists(M+1)
        self.obs_dim = 4 + 2 + 2 * self.n_uavs + (self.n_uavs + 1)
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32) 
                                  for agent in self.agents}
        
        # State (Global): 拼接所有 obs
        self.state_space = Box(-np.inf, np.inf, shape=(self.num_agents * self.obs_dim,), dtype=np.float32)

        # --- 3. 物理模型参数 (完整补全) ---
        # 通信参数
        self.H_uav = 100.0       # UAV 高度 (m)
        self.delta_t = 1.0       # 时隙长度 (s)
        self.fc = 2e9            # 载波频率 (Hz)
        self.B_uav = 20e6        # 带宽 (Hz)
        # 噪声功率 (dBm -> W)
        self.noise_power = 10**((-174 + 10*np.log10(self.B_uav)) / 10) * 1e-3 
        self.P_tx_user = 0.1     # 用户发射功率 (W)
        
        # 计算能力参数
        self.F_uav_max = 10e9    # UAV 最大计算频率 (Hz)  <-- 之前缺失的
        self.F_gec = 50e9        # 地面边缘服务器频率 (Hz)
        self.F_local = 1e9       # 本地计算频率 (Hz)
        self.kappa = 1e-28       # 能耗系数
        
        # 移动参数
        self.V_max = 20.0        # UAV 最大速度 (m/s)
        
        # 飞行能耗参数 (旋翼机模型)
        self.P0 = 79.86          # 悬停功率参数 1
        self.P_ind = 88.63       # 诱导功率参数
        self.U_tip = 120.0       # 叶尖速度
        self.v0 = 4.03           # 平均旋翼诱导速度
        self.d0 = 0.6            # 机身阻力比
        self.rho = 1.225         # 空气密度
        self.s = 0.05            # 旋翼实度
        self.A = 0.503           # 旋翼盘面积
        
        # 奖励归一化因子 (用于缩放 Reward 到合适范围)
        self.norm_gain = self.n_users * 10.0
        self.norm_e_fly = self.n_uavs * 350.0 * self.delta_t
        self.norm_e_comp = self.n_users * 5.0
        self.w_gain = 1.0        # 效用权重
        self.w_energy = 0.5      # 能耗权重

        # --- 4. 内部状态变量 ---
        self.uav_pos = np.zeros((self.n_uavs, 2))
        self.user_pos = np.zeros((self.n_users, 2))
        self.user_tasks = []
        self._current_step = 0
        
        # 轨迹记录与统计
        self.uav_trace = [[] for _ in range(self.n_uavs)]
        # [新增] 用于存储上一个 Episode 的完整轨迹，防止 Auto-Reset 清空数据导致画不出图
        self.last_uav_trace = [[] for _ in range(self.n_uavs)]
        self.episode_metrics = {'gain': [], 'delay': [], 'energy': []}
        
        # 初始化任务，防止第一次 reset 前调用报错
        self._generate_tasks()

    def reset(self):
        # [新增] 在清空 self.uav_trace 之前，将其深拷贝备份到 last_uav_trace
        # 这样即使 VecEnv 自动重置了环境，我们依然能画出上一次的轨迹
        if len(self.uav_trace[0]) > 1:
            self.last_uav_trace = [trace.copy() for trace in self.uav_trace]
        self._current_step = 0
        # 随机初始化位置
        self.uav_pos = np.random.rand(self.n_uavs, 2) * self.map_size
        self.user_pos = np.random.rand(self.n_users, 2) * self.map_size
        self._generate_tasks()
        
        # 重置轨迹
        self.uav_trace = [[self.uav_pos[m].copy()] for m in range(self.n_uavs)]
        self.episode_metrics = {'gain': [], 'delay': [], 'energy': []}
        
        observation = self._get_obs_dict()
        
        # [必须] 返回 info 包含 state 和 avail_actions
        info = {
            'state': self.state(),
            'avail_actions': self.avail_actions()
        }
        return observation, info

    def step(self, action_dict):
        self._current_step += 1
        #print(f"Start step{self._current_step}")
        # 1. 动作解析 (兼容 Tensor 和 int)
        actions = []
        for agent in self.agents:
            act = action_dict[agent]
            if hasattr(act, 'item'): act = act.item()
            actions.append(act)
        
        # 2. 用户关联
        association = {m: [] for m in range(self.n_uavs)}
        local_users, gec_users = [], []
        
        for u_idx, act in enumerate(actions):
            if act == 0: 
                local_users.append(u_idx)
            elif 1 <= act <= self.n_uavs: 
                association[act-1].append(u_idx)
            else: 
                gec_users.append(u_idx)
            
        # 3. 资源分配与轨迹优化
        f_allocs = self._allocate_resources_logic(association) # 这里用到了 F_uav_max
        self._optimize_trajectory_gradient(association)        # 这里用到了 V_max
        
        # 记录轨迹
        for m in range(self.n_uavs):
            self.uav_trace[m].append(self.uav_pos[m].copy())
            
        # 4. 计算指标
        step_metrics = self._calculate_metrics(association, local_users, gec_users, f_allocs)
        
        total_utility = sum(m['utility'] for m in step_metrics)
        
        # 计算飞行能耗
        total_fly = 0
        for m in range(self.n_uavs):
            # 获取上一步位置
            prev = self.uav_trace[m][-2] if len(self.uav_trace[m]) > 1 else self.uav_pos[m]
            total_fly += self._calc_fly_energy(self.uav_pos[m], prev)
            
        #total_comp = sum(m['energy'] for m in step_metrics)
        total_comp = sum(m['comp_energy'] for m in step_metrics)
        # 5. 计算奖励 (归一化)
        norm_gain = (total_utility / self.norm_gain) - self.w_energy * ((total_fly / self.norm_e_fly) + (total_comp / self.norm_e_comp))
        
        observation = self._get_obs_dict()
        rewards = {agent: norm_gain for agent in self.agents} # 合作奖励
        terminated = {agent: False for agent in self.agents}
        truncated = False
        
        if self._current_step >= self.max_episode_steps:
            for agent in self.agents: terminated[agent] = True
            truncated = True
            
        # 记录 Episode 数据
        self.episode_metrics['gain'].append(norm_gain)
        self.episode_metrics['delay'].append(np.mean([m['delay'] for m in step_metrics]))
        self.episode_metrics['energy'].append(total_fly + total_comp)
        
        # [必须] Info 包含 State
        info = {
            'state': self.state(),
            'avail_actions': self.avail_actions()
        }
        
        if truncated:
            info['episode_avg_gain'] = np.mean(self.episode_metrics['gain'])
            info['episode_avg_delay'] = np.mean(self.episode_metrics['delay'])
            info['episode_avg_energy'] = np.mean(self.episode_metrics['energy'])
            
        if not truncated:
            self._generate_tasks()

        return observation, rewards, terminated, truncated, info

    # --- XuanCe 必需接口 ---
    def get_env_info(self):
        return {
            'state_space': self.state_space,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.agents,
            'num_agents': self.num_agents,
            'max_episode_steps': self.max_episode_steps
        }

    def avail_actions(self):
        # 返回全1掩码 (所有动作都可用)
        return {agent: np.ones(self.n_actions, dtype=np.int8) for agent in self.agents}

    def agent_mask(self):
        return {agent: True for agent in self.agents}

    def state(self):
        # 拼接所有 obs 作为 global state
        obs_dict = self._get_obs_dict()
        return np.concatenate([obs_dict[agent] for agent in self.agents])

    def close(self):
        pass

    def render(self, *args, **kwargs):
        # 占位渲染
        return np.ones((64, 64, 3), dtype=np.uint8)

    # --- 内部逻辑函数 ---
    def _get_obs_dict(self):
        obs_dict = {}
        for u in range(self.n_users):
            task = self.user_tasks[u]
            u_pos = self.user_pos[u] / self.map_size
            uav_pos = (self.uav_pos / self.map_size).flatten()
            
            # 计算距离特征
            dists = [np.linalg.norm(self.user_pos[u] - self.uav_pos[m])/self.map_size for m in range(self.n_uavs)]
            # 最后一个距离是到 GEC (假设在中心或某固定点)
            dists.append(np.linalg.norm(self.user_pos[u] - np.array([500,500]))/self.map_size)
            
            # 拼接特征: 任务(4) + 用户位置(2) + UAV位置(2*M) + 距离(M+1)
            feat = np.concatenate([[task['D']/1e6, task['C']/1e9, task['tau'], task['o']], 
                                   u_pos, uav_pos, dists])
            obs_dict[self.agents[u]] = feat.astype(np.float32)
        return obs_dict

    # def _generate_tasks(self):
    #     self.user_tasks = []
    #     for i in range(self.n_users):
    #         is_high = np.random.rand() < 0.3
    #         self.user_tasks.append({
    #             'D': np.random.uniform(0.5e6, 2.0e6), # 数据量
    #             'C': np.random.uniform(0.5e9, 2.0e9), # 计算量
    #             'tau': 1.0 if is_high else 3.0,       # 延迟约束
    #             'o': 1 if is_high else 0              # 优先级
    #         })
    def _generate_tasks(self):
        self.user_tasks = [None] * self.n_users
        
        # 1. 设定固定比例，例如 30%
        ratio_high = 0.3
        num_high = int(self.n_users * ratio_high)
        
        # 2. 创建一个优先级列表：前 num_high 个为 1，其余为 0
        priorities = [1] * num_high + [0] * (self.n_users - num_high)
        
        # 3. 随机打乱优先级分配，确保位置随机但总数固定
        np.random.shuffle(priorities)
        
        for i in range(self.n_users):
            is_high = (priorities[i] == 1)
            self.user_tasks[i] = {
                'D': np.random.uniform(0.5e6, 2.0e6), # 数据量
                'C': np.random.uniform(0.5e9, 2.0e9), # 计算量
                'tau': 1.0 if is_high else 3.0,       # 延迟约束
                'o': 1 if is_high else 0              # 优先级
            }
    
    def _allocate_resources_logic(self, association):
        f_allocs = {}
        beta = 2.0
        for m, users in association.items():
            if not users: continue
            # 根据任务量和优先级加权分配资源
            weights = {u: np.sqrt(self.user_tasks[u]['D']*self.user_tasks[u]['C']*(1+beta*self.user_tasks[u]['o'])) for u in users}
            denom = sum(weights.values())
            for u in users: 
                # 这里使用了 F_uav_max
                f_allocs[(m, u)] = self.F_uav_max * (weights[u]/(denom+1e-9))
        return f_allocs

    def _optimize_trajectory_gradient(self, association):
        step_size = 5.0
        for m in range(self.n_uavs):
            users = association[m]
            if not users: continue
            grad = np.zeros(2)
            for u in users:
                vec = self.user_pos[u] - self.uav_pos[m]
                dist = np.linalg.norm(vec) + 5.0
                w = (1 + 2.0*self.user_tasks[u]['o']) * np.log10(self.user_tasks[u]['D'])
                grad += w * (vec / dist**2)
            
            if np.linalg.norm(grad) > 0:
                move = step_size * (grad / np.linalg.norm(grad))
                # 限制最大速度
                if np.linalg.norm(move) > self.V_max * self.delta_t:
                    move = move / np.linalg.norm(move) * self.V_max * self.delta_t
                self.uav_pos[m] = np.clip(self.uav_pos[m] + move, 0, self.map_size)

    def _calculate_metrics0(self, association, local_users, gec_users, f_allocs):
        metrics = []
        gec_pos = np.array([self.map_size/2, self.map_size/2])
        for u in range(self.n_users):
            task = self.user_tasks[u]
            delay, energy, utility = 100.0, 0, -10.0
            
            if u in local_users:
                delay = task['C']/self.F_local
                energy = self.kappa * self.F_local**2 * task['C']
            elif u in gec_users:
                dist = np.linalg.norm(self.user_pos[u] - gec_pos)
                rate = self._calc_rate(dist, False)
                delay = task['D']/rate + task['C']/self.F_gec
                energy = self.P_tx_user * (task['D']/rate)
            else:
                target_m = -1
                for m, us in association.items(): 
                    if u in us: target_m = m; break
                if target_m != -1 and (target_m, u) in f_allocs:
                    dist = np.linalg.norm(self.user_pos[u] - self.uav_pos[target_m])
                    rate = self._calc_rate(dist, True)
                    delay = task['D']/rate + task['C']/f_allocs[(target_m,u)]
                    energy = self.P_tx_user*(task['D']/rate) + self.kappa*f_allocs[(target_m,u)]**2*task['C']
            
            # 计算效用
            if task['o'] == 1: 
                utility = 20.0 if delay <= task['tau'] else -20.0
            else: 
                utility = 10.0 * np.exp(-0.5 * max(0, delay - task['tau']))
                
            metrics.append({'utility': utility, 'energy': energy, 'delay': delay})
        return metrics

    def _calculate_metrics(self, association, local_users, gec_users, f_allocs):
        metrics = []
        gec_pos = np.array([self.map_size/2, self.map_size/2])
        
        for u in range(self.n_users):
            task = self.user_tasks[u]
            delay = 0.0
            energy_step = 0.0 # 用户侧的能耗 (传输能耗)
            
            # 1. 计算时延 (T_eff) 和 用户能耗
            if u in local_users:
                delay = task['C'] / self.F_local
                # 本地计算也消耗能量，但这通常不算在 System Gain 的 E_comp 里(看论文具体定义)
                # 这里假设 Eq.29 的 E_comp 包含所有计算实体的能耗
                energy_step = self.kappa * self.F_local**2 * task['C'] 
                
            elif u in gec_users:
                dist = np.linalg.norm(self.user_pos[u] - gec_pos)
                rate = self._calc_rate(dist, False) # 假设 GEC 链路是非视距
                delay = task['D']/rate + task['C']/self.F_gec
                energy_step = self.P_tx_user * (task['D']/rate) # 用户传输能耗
                
            else: # 卸载给 UAV
                target_m = -1
                for m, us in association.items(): 
                    if u in us: target_m = m; break
                
                if target_m != -1 and (target_m, u) in f_allocs:
                    f_u = f_allocs[(target_m,u)]
                    dist = np.linalg.norm(self.user_pos[u] - self.uav_pos[target_m])
                    rate = self._calc_rate(dist, True) # UAV 链路是视距
                    
                    trans_delay = task['D'] / rate
                    comp_delay = task['C'] / f_u
                    delay = trans_delay + comp_delay
                    
                    energy_step = self.P_tx_user * trans_delay # 用户传输能耗
                    # 注意: UAV 的计算能耗算在 UAV 头上，不加在这里
                else:
                    delay = 100.0 # 惩罚值: 未成功关联
        
            # 2. 计算效用 (Strictly Eq. 24 & 25)
            utility = 0.0
            tau_u = task['tau']
            
            # --- High Priority (Eq. 24) ---
            if task['o'] == 1:
                if delay <= tau_u:
                    # Logarithmic utility
                    # 防止 log(<=0)，加个 max
                    margin = max(1e-5, 1.0 + tau_u - delay)
                    utility = self.alpha_H * np.log2(margin)
                else:
                    # Failure penalty
                    utility = -self.Phi_H
                    
            # --- Low Priority (Eq. 25) ---
            else:
                if delay <= tau_u:
                    utility = self.Phi_L
                else:
                    # Exponential decay
                    utility = self.Phi_L * np.exp(-self.mu_decay * (delay - tau_u))

            metrics.append({
                'utility': utility, 
                'user_energy': energy_step, # 仅用户传输/本地计算能耗
                'delay': delay,
                'comp_energy': 0.0 # 稍后在 step 里统计 UAV 的计算能耗
            })
            
        return metrics
    
    def _calc_rate(self, d, los):
        # 视距(LoS)与非视距(NLoS)路径损耗
        pl = 128.1 + 37.6*np.log10(np.sqrt(d**2+self.H_uav**2)/1000) + (0 if los else 20)
        return self.B_uav * np.log2(1 + (self.P_tx_user * 10**(-pl/10)) / self.noise_power)
    
    def _calc_fly_energy(self, curr, prev):
        # 旋翼机飞行能耗模型
        v = np.linalg.norm(curr - prev) / self.delta_t
        term1 = self.P0 * (1 + 3*v**2/self.U_tip**2)
        term2 = self.P_ind * np.sqrt(np.sqrt(1 + v**4/(4*self.v0**4)) - v**2/(2*self.v0**2))
        term3 = 0.5 * self.d0 * self.rho * self.s * self.A * v**3
        return (term1 + term2 + term3) * self.delta_t

    def render_trajectory(self, episode_idx, save_path):
        plt.figure(figsize=(8, 8)) # 稍微调大画布
        
        # 1. 画用户
        for u in range(self.n_users):
            c = 'r' if self.user_tasks[u]['o'] == 1 else 'b' # 高优任务红色，普通蓝色
            # s=30: 点的大小
            plt.scatter(self.user_pos[u, 0], self.user_pos[u, 1], c=c, s=30, alpha=0.6, edgecolors='none')
            
        # 2. 画 GEC (地面基站)
        plt.scatter(self.map_size/2, self.map_size/2, c='g', marker='s', s=100, label='GEC', edgecolors='k')
        
        # 3. 画无人机轨迹 (使用备份的 last_uav_trace)
        # 优先使用 last_uav_trace (完整轨迹)，如果为空(刚开始)则使用 current trace
        traces_to_plot = self.last_uav_trace if len(self.last_uav_trace[0]) > 1 else self.uav_trace
        
        colors = ['orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown'] # 预定义一些颜色
        
        for m in range(self.n_uavs):
            trace = np.array(traces_to_plot[m])
            if len(trace) > 0:
                color = colors[m % len(colors)]
                
                # A. 画轨迹线 (虚线)                 
                plt.plot(trace[:, 0], trace[:, 1], lw=2, linestyle='--', color=color, alpha=0.8)
                
                # B. 画起点 (圆点 'o')
                plt.scatter(trace[0, 0], trace[0, 1], marker='o', color=color, s=50, label=f'Start {m}')
                
                # C. 画终点/当前位置 (三角形 '^' 代表无人机)
                plt.scatter(trace[-1, 0], trace[-1, 1], marker='^', color=color, s=120, edgecolors='k', zorder=10, label=f'UAV {m}')

        plt.title(f"Episode {episode_idx} Trajectory")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        
        # 把 Legend 放外面防止遮挡
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        try:
            plt.savefig(os.path.join(save_path, f"traj_{episode_idx}.png"), dpi=100)
            print(f"Saved trajectory image to {save_path}")
        except Exception as e:
            print(f"Save fig failed: {e}")
        plt.close()

    # [新增] 必须实现 render 方法以满足抽象基类要求
    def render(self, *args, **kwargs):
        """
        XuanCe 框架要求的标准渲染接口。
        如果不用于实时可视化，返回一个占位的 RGB 数组即可。
        """
        return np.ones((64, 64, 3), dtype=np.uint8)