import numpy as np
from gymnasium.spaces import Box
# 严格按照教程导入 RawMultiAgentEnv
from xuance.environment import RawMultiAgentEnv

class UAVMECEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(UAVMECEnv, self).__init__()
        self.env_id = env_config.env_id
        
        # 1. 定义智能体列表 (必须是字符串列表)
        self.n_uavs = 4   # 确保这里是数字 4
        self.n_users = 20 # 确保这里是数字 20
        self.n_agents = self.n_uavs # 保持一致
        # --- 关键修改：必须定义 num_agents ---
        self.num_agents = self.n_uavs   # <--- XuanCe 认这个名字 
        
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        
        # 2. 物理与场景参数
        self.n_users = 20
        self.map_size = 500.0
        self.h_uav = 100.0
        self.v_max = 20.0
        self.dt = 1.0
        self.f_uav_max = 10e9
        self.f_user = 1e9
        self.max_episode_steps = 100
        self._current_step = 0

        # 3. 定义空间 (必须是字典格式 {agent_key: Space})
        # 动作空间: [vx, vy]
        self.action_space = {
            agent: Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) 
            for agent in self.agents
        }
        
        # 观测空间
        obs_dim = 2 + (self.n_users * 3) + (self.num_agents * 2)
        self.observation_space = {
            agent: Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        
        # 全局状态空间 (MADDPG 需要)
        state_dim = obs_dim * self.num_agents
        self.state_space = Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # 初始化内部数据
        self.uav_pos = np.zeros((self.num_agents, 2))
        self.user_pos = np.zeros((self.n_users, 2))
        self.user_tasks = []

    def get_env_info(self):
        """教程要求的必须实现的方法"""
        return {
            'state_space': self.state_space,
            'observation_space': self.observation_space,
            'action_space': self.action_space,
            'agents': self.agents,
            'num_agents': self.num_agents,
            'max_episode_steps': self.max_episode_steps
        }

    def avail_actions(self):
        """连续动作通常返回 None"""
        return None

    def agent_mask(self):
        """返回哪些智能体是活着的 (这里永远都活着)"""
        return {agent: True for agent in self.agents}

    def state(self):
        """返回全局状态 (所有观测的拼接)"""
        obs_dict = self._get_obs()
        # 拼接所有 agent 的观测向量
        return np.concatenate([obs_dict[a] for a in self.agents])

    def reset(self):
        """按照教程: 返回 (observation, info)"""
        self._current_step = 0
        
        # 场景初始化逻辑
        self.uav_pos = np.random.rand(self.num_agents, 2) * self.map_size
        self.user_pos = []
        for _ in range(3): # 3个簇
            center = np.random.rand(2) * self.map_size
            self.user_pos.extend(center + np.random.randn(6, 2) * 30)
        while len(self.user_pos) < self.n_users:
            self.user_pos.append(np.random.rand(2) * self.map_size)
        self.user_pos = np.array(self.user_pos).clip(0, self.map_size)
        
        self.user_tasks = []
        for _ in range(self.n_users):
            is_vip = np.random.rand() < 0.2
            self.user_tasks.append({
                'D': np.random.uniform(1e6, 2e6), 'C': 1000,
                'tau': 0.2 if is_vip else 0.5, 'priority': 1 if is_vip else 0
            })
            
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action_dict):
        """按照教程: 返回 (obs, rewards, terminated, truncated, info)"""
        self._current_step += 1
        
        # 1. 执行动作
        movements = np.zeros((self.num_agents, 2))
        for i, agent in enumerate(self.agents):
            # 这里的 action_dict 肯定是 { 'agent_0': [...], ... }
            if agent in action_dict:
                movements[i] = action_dict[agent] * self.v_max
        
        self.uav_pos += movements * self.dt
        self.uav_pos = np.clip(self.uav_pos, 0, self.map_size)
        
        # 2. 计算业务逻辑 (计算卸载、QoE等)
        system_qoe, avg_delay, energy_cost = self._solve_inner_optimization()
        
        # 3. 组装返回值 (必须全部是字典)
        observation = self._get_obs()
        
        # 计算奖励
        collision_penalty = 0
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if np.linalg.norm(self.uav_pos[i] - self.uav_pos[j]) < 10.0:
                    collision_penalty += 50
        
        global_reward = (system_qoe - energy_cost * 0.1 - collision_penalty) / 100.0
        
        rewards = {agent: global_reward for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = False if self._current_step < self.max_episode_steps else True
        
        # 将自定义指标放入 info
        info = {}
        for agent in self.agents:
            info[agent] = {
                "metric_QoE": system_qoe,
                "metric_Energy": energy_cost,
                "metric_Delay": avg_delay
            }
            
        return observation, rewards, terminated, truncated, info

    def render(self, *args, **kwargs):
        return np.ones([64, 64, 64])

    def close(self):
        pass

    def _get_obs(self):
        """内部辅助函数: 生成观测字典"""
        obs_dict = {}
        for i, agent in enumerate(self.agents):
            own = self.uav_pos[i] / self.map_size
            
            users_info = []
            for u_idx, u_pos in enumerate(self.user_pos):
                rel = (u_pos - self.uav_pos[i]) / self.map_size
                prio = self.user_tasks[u_idx]['priority']
                users_info.extend([rel[0], rel[1], prio])
            
            mates_info = []
            for j, m_pos in enumerate(self.uav_pos):
                rel = (m_pos - self.uav_pos[i]) / self.map_size
                mates_info.extend([rel[0], rel[1]])
            
            # 确保 Key 是字符串 'agent_x'
            obs_dict[agent] = np.concatenate([own, users_info, mates_info]).astype(np.float32)
        return obs_dict

    def _solve_inner_optimization(self):
        """传统贪婪算法解决卸载与资源分配"""
        total_qoe = 0
        delays = []
        served_count = 0
        
        # 简化的贪婪逻辑
        # 1. 计算增益并排序
        candidates = []
        for u_idx, u_pos in enumerate(self.user_pos):
            # 找最近UAV
            dists = np.linalg.norm(self.uav_pos - u_pos, axis=1)
            best_uav = np.argmin(dists)
            
            # 估算传输速率 (简化Log距离模型)
            dist = dists[best_uav]
            path_loss = 1e-3 * (dist + 1)**(-2) 
            rate = 1e6 * np.log2(1 + 0.1 * path_loss / 1e-13)
            
            # 本地 vs 卸载 估算
            t_local = (self.user_tasks[u_idx]['D'] * 1000) / self.f_user
            t_trans = self.user_tasks[u_idx]['D'] / rate
            t_comp_est = (self.user_tasks[u_idx]['D'] * 1000) / (self.f_uav_max / 5.0) # 假设分1/5
            
            gain = t_local - (t_trans + t_comp_est)
            candidates.append({'u': u_idx, 'm': best_uav, 'gain': gain, 'trans': t_trans})
            
            # 默认本地
            total_qoe += self._util(t_local, self.user_tasks[u_idx])
            delays.append(t_local)

        # 2. 排序 (Gain高的优先，通常是VIP)
        candidates.sort(key=lambda x: x['gain'], reverse=True)
        
        # 3. 准入
        uav_loads = np.zeros(self.n_agents)
        for c in candidates:
            if c['gain'] > 0 and uav_loads[c['m']] < 5: # 简单容量限制
                # 决定卸载：修正QoE
                task = self.user_tasks[c['u']]
                t_real = c['trans'] + (task['D']*1000)/(self.f_uav_max/ (uav_loads[c['m']]+1))
                
                # 扣除之前加的本地，加上新的卸载
                total_qoe -= self._util(delays[c['u']], task)
                total_qoe += self._util(t_real, task)
                
                delays[c['u']] = t_real
                uav_loads[c['m']] += 1

        energy = np.sum(np.linalg.norm(self.uav_pos, axis=1)**3) * 0.01 # 简易能耗
        return total_qoe, np.mean(delays), energy

    def _util(self, t, task):
        # 简单的Sigmoid效用
        beta = 5 if task['priority'] else 1
        if t > task['tau']: return -10
        return beta * 100 * np.exp(-t/task['tau'])
