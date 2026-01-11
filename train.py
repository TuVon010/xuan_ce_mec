import os
import argparse
import numpy as np
import torch
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, make_envs
from xuance.torch.agents import MAPPO_Agents 
from medical_env import MedicalMECEnv

def parse_args():
    parser = argparse.ArgumentParser("MAPPO for Medical MEC")
    parser.add_argument("--config", type=str, default="./configs/medical_mec.yaml")
    parser.add_argument("--log_interval", type=int, default=5)
    return parser.parse_args()

# --- 辅助函数：将 List[Dict] 转为 Dict[np.array] ---
def _to_dict_array(data_list):
    if not isinstance(data_list, list):
        return data_list
    if len(data_list) == 0:
        return {}
    keys = data_list[0].keys()
    # 将 list of dicts 堆叠为 dict of arrays
    return {k: np.stack([d[k] for d in data_list]) for k in keys}

def main():
    args = parse_args()
    
    # 1. 读取配置
    if not os.path.exists(args.config): 
        raise FileNotFoundError(f"Missing config: {args.config}")
    config_dict = get_configs(args.config)
    for k, v in config_dict.items(): setattr(args, k, v)
    
    # 类型转换 (防止 YAML 解析错误)
    try:
        args.learning_rate = float(args.learning_rate)
        args.weight_decay = float(args.weight_decay)
        args.gamma = float(args.gamma)
        args.running_steps = int(args.running_steps)
        args.n_epochs = int(args.n_epochs)
        args.n_minibatch = int(args.n_minibatch)
    except Exception as e:
        print(f"Warning: Type casting failed {e}")

    # 补充默认值
    if not hasattr(args, 'seed'): setattr(args, 'seed', 1)
    if not hasattr(args, 'env_seed'): setattr(args, 'env_seed', 1)
    if not hasattr(args, 'parallels'): setattr(args, 'parallels', 1)

    # 2. 设置路径
    log_dir = os.path.join("./logs", args.env_id)
    model_dir = os.path.join("./models", args.env_id)
    fig_dir = os.path.join("./figures", args.env_id)
    
    setattr(args, "log_dir", log_dir)
    setattr(args, "model_dir", model_dir)
    setattr(args, "fig_dir", fig_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 3. 注册环境
    REGISTRY_MULTI_AGENT_ENV[args.env_name] = MedicalMECEnv
    
    # 4. 创建环境与智能体
    envs = make_envs(args)
    agents = MAPPO_Agents(config=args, envs=envs)
    
    # 5. 训练循环
    print(f"Start Training MAPPO... Total Episodes: {args.n_episodes}")
    print(f"Logs: {log_dir} | Models: {model_dir}")
    
    best_gain = -np.inf
    
    for i_episode in range(args.n_episodes):
        # [关键初始化]
        obs_list, _ = envs.reset()
        state = np.array(envs.buf_state)
        
        # 初始化循环变量
        terminated = [False] * args.parallels
        episode_info = {}  # [本次报错修复点] 必须在此初始化
        step_counter = 0   # 防止死循环的计数器
        
        while True:
            # 1. 策略采样
            action_result = agents.action(obs_dict=obs_list, state=state)
            
            # 提取结果 (List[Dict] 或 Dict[Array])
            step_actions_list = action_result['actions'] 
            values_dict = action_result['values']        
            log_pi_dict = action_result['log_pi']        
            
            # 2. 环境步进
            next_obs_list, rewards_list, terminated_list, truncated, info = envs.step(step_actions_list)
            next_state = np.array(envs.buf_state)
            
            # 3. 数据转置 (List[Dict] -> Dict[Array])
            obs_dict = _to_dict_array(obs_list)
            actions_dict = _to_dict_array(step_actions_list)
            rewards_dict = _to_dict_array(rewards_list)
            terminals_dict = _to_dict_array(terminated_list)
            
            # 构造 Agent Mask (假设所有智能体都存活)
            agent_mask_dict = {k: np.ones(args.parallels, dtype=bool) for k in agents.agent_keys}

            # 4. 存储 Experience
            experience_data = {
                'obs': obs_dict,
                'actions': actions_dict,
                'log_pi_old': log_pi_dict,
                'values': values_dict,
                'rewards': rewards_dict,
                'terminals': terminals_dict,
                'agent_mask': agent_mask_dict
            }
            if args.use_global_state:
                experience_data['state'] = state
            
            agents.memory.store(**experience_data)
            
            # 更新状态
            obs_list = next_obs_list
            state = next_state
            step_counter += 1
            
            # [关键修复] 显式检查结束条件，防止死循环
            # 只要有一个并行环境结束 (terminated) 或超时 (truncated)，就结束本轮
            any_done = False
            for i in range(args.parallels):
                is_done = all(terminated_list[i].values()) or truncated[i]
                if is_done:
                    any_done = True
                    break
            
            # 提取 Info (用于日志)
            if isinstance(info, list):
                for i_info in info:
                    if 'episode_avg_gain' in i_info:
                        episode_info = i_info
                        break
            elif isinstance(info, dict):
                 if 'episode_avg_gain' in info: episode_info = info

            # 如果结束，跳出 while 循环
            if any_done:
                break
        
        # 训练 (Buffer 满时)
        if agents.memory.full:
            res = agents.action(obs_dict=next_obs_list, state=next_state)
            next_values = res['values']
            agents.train(args.n_epochs) 
            agents.memory.clear() 

        # --- 记录与绘图 ---
        if 'episode_avg_gain' in episode_info:
            gain = episode_info['episode_avg_gain']
            delay = episode_info['episode_avg_delay']
            energy = episode_info.get('episode_avg_energy', 0.0)
            #在这个 Episode 中，整个多无人机系统平均每秒（或每个时隙）消耗了多少能量
            # 记录到 Tensorboard
            agents.writer.add_scalar("Metrics/Gain", gain, i_episode)
            agents.writer.add_scalar("Metrics/Delay", delay, i_episode)
            agents.writer.add_scalar("Metrics/Energy", energy, i_episode)
            
            is_best = False
            if gain > best_gain:
                best_gain = gain
                agents.save_model("best_model.pth")
                is_best = True
                
            if i_episode % args.log_interval == 0 or is_best:
                print(f"Ep {i_episode:04d} | Gain: {gain:.4f} | Delay: {delay:.4f} | Energy: {energy:.2f} {'[BEST]' if is_best else ''}")
        else:
            # 这种情况通常只在第一轮或者info获取失败时出现
            pass

        if i_episode % 5 == 0:
            # 1. 获取第一个并行环境实例
            raw_env = envs.envs[0]

            # 2. [关键修复] 递归解包：如果被 Wrapper 包裹，一直往下找，直到找到包含 render_trajectory 的层
            while hasattr(raw_env, 'env') and not hasattr(raw_env, 'render_trajectory'):
                raw_env = raw_env.env

            # 3. 执行绘图并打印调试信息
            if hasattr(raw_env, 'render_trajectory'):
                print(f"--> Saving trajectory for Ep {i_episode} to {fig_dir} ...") 
                try:
                    raw_env.render_trajectory(i_episode, fig_dir)
                except Exception as e:
                    print(f"Error plotting: {e}")
            else:
                # 如果打印这句话，说明真的没找到，可能是环境类没正确加载
                print(f"Warning: 'render_trajectory' method not found! Current env type: {type(raw_env)}")

    agents.save_model("final_model.pth")
    agents.finish()
    envs.close()
#含义：best表示当前这一轮（Episode）的平均收益（Gain）超过了历史最高记录
#tensorboard --logdir=logs
if __name__ == "__main__":
    main()







