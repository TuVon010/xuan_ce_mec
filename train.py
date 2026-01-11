import argparse
import numpy as np
from xuance.common import get_configs
from xuance.environment import make_envs, REGISTRY_MULTI_AGENT_ENV
from xuance.torch.agents import MADDPG_Agents
from uav_mec_env import UAV_MEC_Env 

def _to_dict_array(data_list):
    """
    将 list[dict] 转换为 dict[np.array]
    例如: [{'uav_0': 1}, {'uav_0': 2}] -> {'uav_0': np.array([1, 2])}
    """
    keys = data_list[0].keys()
    return {k: np.array([data[k] for data in data_list]) for k in keys}

def parse_args():
    parser = argparse.ArgumentParser("UAV MEC Training")
    parser.add_argument("--config", type=str, default="uav_mec_config.yaml")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    configs_dict = get_configs(file_dir=args.config)
    configs = argparse.Namespace(**configs_dict)

    # 注册与创建环境
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = UAV_MEC_Env
    envs = make_envs(configs) # 注意：这里返回的是 VectorizeEnv (并行环境包装器)
    
    # 创建智能体
    agents = MADDPG_Agents(config=configs, envs=envs)
    
    # === 开始自定义训练循环 ===
    print(f"Start Training... Total Steps: {configs.running_steps}")
    
    # 记录训练数据
    n_episodes = configs.running_steps // configs.max_episode_steps
    current_step = 0
    # 获取并行环境数量，用于生成 mask
    n_envs = envs.num_envs 
    
    # ... (前面的代码不变) ...
    
    # 全局步数计数器
    global_step = 0

    for i_episode in range(n_episodes):
        obs_list, _ = envs.reset()
        
        # === 1. 修改初始化统计变量 ===
        # 我们现在关心的是: 系统增益(目标), 总效用, 飞行能耗, 计算能耗
        ep_rewards = 0
        ep_system_gain = 0
        ep_utility = 0
        ep_fly_energy = 0
        ep_comp_energy = 0
        ep_avg_latency_accum = 0.0 # <--- 新增累加器
        
        for step in range(configs.max_episode_steps):
            global_step += n_envs 

            # === 1. 智能体决策 (修正点) ===
            # 错误写法: action_data = agents.action(obs_dict) 
            # 正确写法: 直接传 obs_list (List[Dict])
            action_data = agents.action(obs_list) 
            
            # 获取动作列表 [{'uav_0': act, ...}, ...]
            step_actions_list = action_data['actions'] 

            # === 2. 环境步进 ===
            # step_actions_list 已经是 List[Dict] 格式，直接传给 envs
            next_obs_list, rewards_list, terminated_list, truncated, info_list = envs.step(step_actions_list)

            # === 3. 数据转换 (仅用于 Buffer 存储) ===
            # 在这里进行转换，专门喂给 Buffer
            obs_dict = _to_dict_array(obs_list)          # <--- 转换 obs
            next_obs_dict = _to_dict_array(next_obs_list)# <--- 转换 next_obs
            actions_dict_stored = _to_dict_array(step_actions_list) # <--- 转换 action
            rewards_dict = _to_dict_array(rewards_list)
            terminals_dict = _to_dict_array(terminated_list)
            agent_mask_dict = {k: np.ones(n_envs, dtype=bool) for k in agents.agent_keys}

            # === 4. 存入 Buffer ===
            experience_data = {
                'obs': obs_dict,                 # 存的是 Dict[Array]
                'actions': actions_dict_stored,
                'obs_next': next_obs_dict,
                'rewards': rewards_dict,
                'terminals': terminals_dict,
                'agent_mask': agent_mask_dict
            }
            agents.memory.store(**experience_data)
            
            # --- 训练逻辑 (保持不变) ---
            start_training_step = getattr(configs, 'start_training', 1000)
            if global_step >= start_training_step:
                agents.train_epochs(n_epochs=1)
            
            obs_list = next_obs_list
            
            # === 2. 修改指标提取逻辑 (适配新环境) ===
            try:
                # info_list[0] 是第一个并行环境的信息
                first_env_info = info_list[0]
                agent_key = agents.agent_keys[0] # 获取第一个智能体名称 ('uav_0')
                
                if agent_key in first_env_info:
                    step_info = first_env_info[agent_key]
                    
                    # 累加 Reward (RL 优化目标)
                    ep_rewards += rewards_list[0][agent_key]
                    
                    # 累加 System Gain (物理优化目标)
                    ep_system_gain += step_info.get('system_gain', 0)
                    
                    # 累加 Utility (效用)
                    ep_utility += step_info.get('step_utility', 0)
                    
                    # 累加 Energy (能耗)
                    ep_fly_energy += step_info.get('step_fly_energy', 0)
                    ep_comp_energy += step_info.get('step_comp_energy', 0)
                    # 累加每个时隙的平均时延
                    ep_avg_latency_accum += step_info.get('step_avg_latency', 0)
                    
            except Exception as e:
                pass # 避免因 info 缺失导致训练中断
        
        # === 3. 修改打印格式 ===
        # 计算平均 System Gain (因为 Gain 是每个 Slot 的瞬时值，看平均更有意义)
        avg_sys_gain = ep_system_gain / configs.max_episode_steps
        
        # 计算平均 Utility (同上)
        avg_utility = ep_utility / configs.max_episode_steps
        
        # 能耗看总量 (Total)
        total_fly = ep_fly_energy
        total_comp = ep_comp_energy
        # 计算整个 Episode 的平均时延 (所有时隙的平均值的平均)
        avg_latency_per_slot = ep_avg_latency_accum / configs.max_episode_steps
        
        print(f"Episode {i_episode+1}/{n_episodes} (Step {global_step}) | "
              f"Reward: {ep_rewards:.2f} | "
              f"Avg SysGain: {avg_sys_gain:.4f} | "
              f"Avg Util: {avg_utility:.2f} | "
              f"Latency: {avg_latency_per_slot:.4f}s | " # <--- 打印在这里
              f"E_Fly: {total_fly:.1f} | "
              f"E_Comp: {total_comp:.1f}"
              )
        
        # --- 保存图片 (保持不变) ---
        if (i_episode + 1) % 10 == 0: 
            try:
                raw_env = envs.envs[0] 
                save_path = raw_env.render(episode_id=i_episode+1)
                print(f"  -> Trajectory saved to {save_path}")
            except Exception as e:
                print(f"  -> Render failed: {e}")


    # 结束
    agents.save_model("final_model.pth")
    envs.close()
    agents.finish()