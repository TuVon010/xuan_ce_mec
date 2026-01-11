import numpy as np
from gymnasium.spaces import Box
from xuance.environment import RawMultiAgentEnv

class UAV_MEC_Env(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(UAV_MEC_Env, self).__init__()
        self.env_id = env_config.env_id
        
        # --- 1. 系统配置 ---
        self.n_uavs = env_config.n_uavs
        self.n_users = env_config.n_users
        self.map_size = env_config.map_size
        self.h_uav = env_config.h_uav
        self.v_max = env_config.v_max
        self.dt = 1.0 
        
        # 通信参数
        self.fc = 2.4e9
        self.c_light = 3e8
        self.bandwidth_uav = 20e6  # 20 MHz (增加带宽，降低传输时延)
        self.p_tx_user = 0.2       # 23 dBm
        self.noise_power = 1e-13   
        
        # 概率 LoS 参数
        self.psi_a = 9.61
        self.psi_b = 0.16
        self.eta_los = 1.0
        self.eta_nlos = 20.0
        
        # 计算参数 (关键：参数校准)
        # 假设 1 Gcycles/s = 1e9 Hz
        self.f_uav_max = 10e9  # UAV总算力 10 GHz
        self.f_local = 1e9     # 本地算力 1 GHz
        self.kappa = 1e-28     # 能耗系数
        
        # 优先级与效用参数 (Paper Draft 35)
        self.priority_ratios = [0.8, 0.2] # 80% Low, 20% High
        self.alpha_h = 20.0  # High优先级 Log系数 (调大，强调重要性)
        self.phi_h = 100.0   # High优先级 失败惩罚
        self.phi_l = 10.0    # Low优先级 完成奖励
        self.mu_decay = 2.0  # Low优先级 衰减速率
        
        # 目标函数权重 (System Gain = w1*Util - w2*E_fly - w3*E_comp)
        self.w_util = 1
        self.w_fly = 0     # 飞行能耗权重
        self.w_comp =0    # 计算能耗权重
        
        # 归一化常数 (用于将不同单位的数值拉到同一水平)
        # 预估最大值：
        # Max Utility approx: N_users * 15
        # Max E_fly approx: N_uavs * 200W * 1s
        # Max E_comp approx: N_users * 0.2J
        self.norm_util = self.n_users * 10.0
        self.norm_e_fly = self.n_uavs * 100.0 
        self.norm_e_comp = self.n_uavs * 0.3 
        
        # RL 空间
        self.agents = [f"uav_{i}" for i in range(self.n_uavs)]
        self.num_agents = self.n_uavs
        self.action_space = {agent: Box(-1.0, 1.0, shape=[2, ], dtype=np.float32) for agent in self.agents}
        
        obs_dim = 3 + 2 * self.n_users + 2 * (self.n_uavs - 1)
        self.observation_space = {agent: Box(-np.inf, np.inf, shape=[obs_dim, ], dtype=np.float32) for agent in self.agents}
        state_dim = 3 * self.n_uavs + 2 * self.n_users
        self.state_space = Box(-np.inf, np.inf, shape=[state_dim, ], dtype=np.float32)
        
        self.max_episode_steps = env_config.max_episode_steps
        self._current_step = 0
        self.uav_trace = [[] for _ in range(self.n_uavs)]

    def get_env_info(self):
        return {'state_space': self.state_space, 'observation_space': self.observation_space, 
                'action_space': self.action_space, 'agents': self.agents, 
                'num_agents': self.num_agents, 'max_episode_steps': self.max_episode_steps}

    def reset(self):
        self._current_step = 0
        self.uav_pos = np.random.uniform(0, self.map_size, (self.n_uavs, 2))
        self.user_pos = np.random.uniform(0, self.map_size, (self.n_users, 2))
        self.uav_battery = np.ones(self.n_uavs) * 500000.0 
        
        # 生成任务 (校准数值)
        n_high = int(self.n_users * self.priority_ratios[1])
        w_list = np.concatenate([np.full(n_high, 1), np.full(self.n_users - n_high, 0)])
        np.random.shuffle(w_list)
        
        self.user_tasks = {
            # Data: 1~2 Mb
            'D': np.random.uniform(1e6, 2e6, self.n_users),     
            # Computation Intensity: 800~1200 cycles/bit (这才合理，之前1e9是总周期，太大了)
            'C': np.random.uniform(800, 1200, self.n_users), 
            'w': w_list,
            'tau': np.zeros(self.n_users)
        }
        
        # 设置 Deadline
        # 本地计算时间估算: 1.5Mb * 1000cyc/b / 1GHz = 1.5s
        # 所以 High 的 Deadline 设为 0.5~1.0s (逼迫卸载)，Low 设为 0.8~4.0s
        for i in range(self.n_users):
            if w_list[i] == 1: self.user_tasks['tau'][i] = np.random.uniform(0.5, 1.0)
            else: self.user_tasks['tau'][i] = np.random.uniform(0.8, 4.0)
            
        self.uav_trace = [[] for _ in range(self.n_uavs)]
        for i in range(self.n_uavs): self.uav_trace[i].append(self.uav_pos[i].copy())
        return self._get_obs(), self._get_info()

    def step(self, action_input):
        self._current_step += 1
        
        # 1. 动作解析
        if isinstance(action_input, dict) and 'actions' in action_input: raw_actions = action_input['actions']
        else: raw_actions = action_input
        if not isinstance(raw_actions, dict): action_dict = {agent: raw_actions[i] for i, agent in enumerate(self.agents)}
        else: action_dict = raw_actions

        # 2. UAV 移动与飞行能耗
        fly_energy_step = 0.0
        collision_penalty = 0
        boundary_penalty = 0
        
        for i, agent in enumerate(self.agents):
            act = action_dict[agent]
            vel = act * self.v_max
            new_pos = self.uav_pos[i] + vel * self.dt
            
            # 边界检查
            if not ((0 <= new_pos[0] <= self.map_size) and (0 <= new_pos[1] <= self.map_size)):
                boundary_penalty += 0.1
                new_pos = np.clip(new_pos, 0, self.map_size)
            
            self.uav_pos[i] = new_pos
            self.uav_trace[i].append(new_pos.copy())
            
            # 飞行能耗 (简化模型: 悬停100W + 动能)
            speed = np.linalg.norm(vel)
            p_fly = 100.0 + 0.2 * (speed ** 3) # 降低系数防止数值爆炸
            e_fly = p_fly * self.dt
            fly_energy_step += e_fly
            self.uav_battery[i] -= e_fly

        # 碰撞惩罚
        for i in range(self.n_uavs):
            for j in range(i + 1, self.n_uavs):
                if np.linalg.norm(self.uav_pos[i] - self.uav_pos[j]) < 10.0:
                    collision_penalty += 0.1

        # 3. 计算资源分配与系统增益 (协作优化)
        total_utility, total_comp_energy, total_latency_sum = self._compute_system_gain_collaborative()

        step_avg_latency = total_latency_sum / self.n_users
        
        # 4. 计算 Reward (System Gain)
        # Term 1: Normalized Utility (正收益)
        if self._current_step%50 == 0:
            print(f"总效用{total_utility:.4f},计算总能耗{total_comp_energy:.4f},飞行{fly_energy_step:.4f}")
        term_util = self.w_util * (total_utility / self.norm_util)
        # Term 2: Normalized Fly Energy (负收益)
        term_fly = self.w_fly * (fly_energy_step / self.norm_e_fly)
        # Term 3: Normalized Comp Energy (负收益)
        term_comp = self.w_comp * (total_comp_energy / self.norm_e_comp)
        
        system_gain = term_util - term_fly - term_comp
        
        # 加上越界和碰撞惩罚
        global_reward = system_gain - (collision_penalty + boundary_penalty) * 0.01
        
        rewards = {agent: global_reward for agent in self.agents}
        
        # 5. 更新 Info (加入 step_avg_latency)
        info = {
            agent: {
                "step_utility": total_utility,
                "step_fly_energy": fly_energy_step,
                "step_comp_energy": total_comp_energy,
                "system_gain": system_gain,
                "step_avg_latency": step_avg_latency  # <--- 放入 info
            } for agent in self.agents
        }
        
        terminated = {agent: False for agent in self.agents}
        truncated = (self._current_step >= self.max_episode_steps)
        return self._get_obs(), rewards, terminated, truncated, info

    def _compute_system_gain_collaborative(self):
        """
        [修改版]：引入“最优资源分配” (替代平均分配)
        """
        # 1. 预计算通信速率 (不变)
        rates = np.zeros((self.n_users, self.n_uavs))
        for u in range(self.n_users):
            for m in range(self.n_uavs):
                h = self._get_channel_gain(m, u)
                snr = (self.p_tx_user * h) / self.noise_power
                rates[u, m] = self.bandwidth_uav * np.log2(1 + snr)

        uav_users_list = [[] for _ in range(self.n_uavs)]
        user_indices = np.argsort(self.user_tasks['w'])[::-1]
        
        # 定义计算资源的分配函数 (Closed-form Optimal Allocation)
        def allocate_resources(uav_user_indices):
            if not uav_user_indices:
                return {}
            
            # 权重计算: sqrt(priority_weight * C)
            # High(1) -> 权重 10, Low(0) -> 权重 1
            weights = {}
            total_weight = 0
            for uid in uav_user_indices:
                w_val = 10.0 if self.user_tasks['w'][uid] == 1 else 1.0
                c_val = self.user_tasks['C'][uid]
                # 经典的凸优化结果：资源分配与 sqrt(w*C) 成正比
                weight = np.sqrt(w_val * c_val)
                weights[uid] = weight
                total_weight += weight
            
            # 分配频率
            f_allocation = {}
            for uid in uav_user_indices:
                ratio = weights[uid] / (total_weight + 1e-9)
                f_allocation[uid] = self.f_uav_max * ratio
            return f_allocation

        total_sys_utility = 0
        total_sys_comp_energy = 0
        
        # 3. 逐个用户决策
        for u in user_indices:
            # --- Option A: Local ---
            t_loc = (self.user_tasks['D'][u] * self.user_tasks['C'][u]) / self.f_local
            e_loc = self.kappa * (self.f_local**2) * (self.user_tasks['D'][u] * self.user_tasks['C'][u])
            util_loc = self._calculate_user_utility(u, t_loc)
            gain_local = self.w_util * (util_loc / self.norm_util) - self.w_comp * (e_loc / self.norm_e_comp)
            
            best_gain = gain_local
            best_choice = -1 
            
            # --- Option B: Offload to UAV m ---
            for m in range(self.n_uavs):
                if rates[u, m] < 1e4: continue 
                
                # 模拟加入
                current_users = uav_users_list[m]
                new_users = current_users + [u]
                
                # === 【关键修改点】 使用最优资源分配，而不是平均分配 ===
                f_map_new = allocate_resources(new_users)
                
                # 计算新状态下的总增益
                gain_uav_m_new = 0
                for u_idx in new_users:
                    f_assigned = f_map_new[u_idx]
                    t = self.user_tasks['D'][u_idx]/rates[u_idx, m] + (self.user_tasks['D'][u_idx]*self.user_tasks['C'][u_idx])/f_assigned
                    e = self.kappa * (f_assigned**2) * (self.user_tasks['D'][u_idx]*self.user_tasks['C'][u_idx])
                    u_val = self._calculate_user_utility(u_idx, t)
                    gain_uav_m_new += self.w_util * (u_val / self.norm_util) - self.w_comp * (e / self.norm_e_comp)
                
                # 计算旧状态增益
                gain_uav_m_old = 0
                if current_users:
                    f_map_old = allocate_resources(current_users)
                    for u_idx in current_users:
                        f_assigned = f_map_old[u_idx]
                        t = self.user_tasks['D'][u_idx]/rates[u_idx, m] + (self.user_tasks['D'][u_idx]*self.user_tasks['C'][u_idx])/f_assigned
                        e = self.kappa * (f_assigned**2) * (self.user_tasks['D'][u_idx]*self.user_tasks['C'][u_idx])
                        u_val = self._calculate_user_utility(u_idx, t)
                        gain_uav_m_old += self.w_util * (u_val / self.norm_util) - self.w_comp * (e / self.norm_e_comp)
                
                marginal = gain_uav_m_new - gain_uav_m_old
                if marginal > best_gain:
                    best_gain = marginal
                    best_choice = m
            
            if best_choice != -1:
                uav_users_list[best_choice].append(u)
            
        # 4. 最终汇总 (同样应用最优分配)
        # 4. 最终汇总 (修改这里)
        final_util_sum = 0
        final_e_comp_sum = 0
        total_latency_sum = 0.0  # <--- 新增：累加总时延
        
        # 汇总 UAV 用户
        for m in range(self.n_uavs):
            users = uav_users_list[m]
            if not users: continue
            
            f_map = allocate_resources(users)
            
            for u in users:
                f_final = f_map[u]
                # 计算时延
                t = self.user_tasks['D'][u]/rates[u, m] + (self.user_tasks['D'][u]*self.user_tasks['C'][u])/f_final
                
                # 累加指标
                e = self.kappa * (f_final**2) * (self.user_tasks['D'][u]*self.user_tasks['C'][u])
                final_util_sum += self._calculate_user_utility(u, t)
                final_e_comp_sum += e
                total_latency_sum += t  # <--- 累加
                
        # 汇总 Local 用户
        assigned_set = set([u for sublist in uav_users_list for u in sublist])
        for u in range(self.n_users):
            if u not in assigned_set:
                t = (self.user_tasks['D'][u]*self.user_tasks['C'][u]) / self.f_local
                e = self.kappa * (self.f_local**2) * (self.user_tasks['D'][u]*self.user_tasks['C'][u])
                
                final_util_sum += self._calculate_user_utility(u, t)
                final_e_comp_sum += e
                total_latency_sum += t  # <--- 累加
                
        # 返回三个值: 效用, 计算能耗, 总时延
        return final_util_sum, final_e_comp_sum, total_latency_sum
    def _calculate_user_utility(self, u_idx, delay):
        priority = self.user_tasks['w'][u_idx]
        deadline = self.user_tasks['tau'][u_idx]
        if priority == 1: # High
            if delay <= deadline: return self.alpha_h * np.log2(1.0 + deadline - delay + 1e-6)
            else: return -self.phi_h # 超时重罚
        else: # Low
            if delay <= deadline: return self.phi_l
            else: 
                overdue = delay - deadline
                return self.phi_l * np.exp(-self.mu_decay * overdue)

    def _get_channel_gain(self, uav_idx, user_idx):
        d_3d = np.linalg.norm(self.uav_pos[uav_idx] - self.user_pos[user_idx])
        val = np.clip(self.h_uav / (d_3d + 1e-6), -1.0, 1.0)
        elevation = (180/np.pi) * np.arcsin(val)
        p_los = 1.0 / (1.0 + self.psi_a * np.exp(-self.psi_b * (elevation - self.psi_a)))
        pl_los = 20*np.log10(d_3d+1) + 20*np.log10(self.fc) - 147.55 + self.eta_los
        pl_nlos = 20*np.log10(d_3d+1) + 20*np.log10(self.fc) - 147.55 + self.eta_nlos
        pl_avg = p_los * pl_los + (1 - p_los) * pl_nlos
        return 10 ** (-pl_avg / 10)

    def _get_obs(self):
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            own = np.concatenate([self.uav_pos[i]/self.map_size, [self.uav_battery[i]/500000.0]])
            u_rel = (self.user_pos - self.uav_pos[i]).flatten()/self.map_size
            n_rel = np.array([])
            for j in range(self.n_uavs):
                if i!=j: n_rel = np.concatenate([n_rel, (self.uav_pos[j]-self.uav_pos[i])/self.map_size])
            obs_dict[agent] = np.concatenate([own, u_rel, n_rel]).astype(np.float32)
        return obs_dict
        
    def _get_info(self): return {agent: {} for agent in self.agents}
    
    def render(self, mode='rgb_array', episode_id=0):
        import matplotlib.pyplot as plt
        import os
        if not os.path.exists("./results/images/"): os.makedirs("./results/images/")
        plt.figure(figsize=(6, 6))
        w = self.user_tasks['w']
        plt.scatter(self.user_pos[w==1, 0], self.user_pos[w==1, 1], c='r', marker='^', s=80, label='High')
        plt.scatter(self.user_pos[w==0, 0], self.user_pos[w==0, 1], c='b', marker='o', label='Low')
        cols = ['g','y','c','m']
        for i in range(self.n_uavs):
            trace = np.array(self.uav_trace[i])
            if len(trace)>0:
                plt.plot(trace[:,0], trace[:,1], c=cols[i%4], alpha=0.6)
                plt.scatter(trace[-1,0], trace[-1,1], c=cols[i%4], s=100, label=f'UAV{i}')
        plt.xlim(0, self.map_size); plt.ylim(0, self.map_size)
        plt.legend()
        path = f"./results/images/traj_ep_{episode_id}.png"
        plt.savefig(path); plt.close()
        return path
    
    def state(self):
        """
        返回全局状态 (Global State)，供 MADDPG 的 Critic 使用。
        形状必须与 self.state_space 一致。
        内容: [所有UAV位置(归一化), 所有UAV电量(归一化), 所有用户位置(归一化)]
        """
        # 1. 所有 UAV 位置 (N_uavs * 2)
        uav_pos_norm = self.uav_pos.flatten() / self.map_size
        
        # 2. 所有 UAV 电量 (N_uavs * 1)
        uav_batt_norm = self.uav_battery / 500000.0
        
        # 3. 所有用户位置 (N_users * 2)
        user_pos_norm = self.user_pos.flatten() / self.map_size
        
        # 拼接 (2N + N + 2M) -> 对应 __init__ 中的 state_dim = 3*n_uavs + 2*n_users
        s = np.concatenate([uav_pos_norm, uav_batt_norm, user_pos_norm])
        
        return s.astype(np.float32)
    def agent_mask(self):
        """
        返回布尔掩码，指示哪些智能体处于活动状态。
        对于你的场景，所有 UAV 始终是存活的，所以全返回 True。
        """
        return {agent: True for agent in self.agents}

    def avail_actions(self):
        """
        返回动作掩码 (Action Mask)。
        对于连续动作空间 (Box)，通常不需要 Mask，返回 None 即可。
        这样就能避开基类中访问 .n 属性导致的报错。
        """
        return None
    
    def close(self):
        """关闭环境，释放资源"""
        return