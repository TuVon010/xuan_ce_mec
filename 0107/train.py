import argparse
import os
import numpy as np
from xuance.common import get_configs
# 导入注册表
from xuance.environment import REGISTRY_MULTI_AGENT_ENV, make_envs
from xuance.torch.agents import MADDPG_Agents

# 导入你刚才写的新环境
from env0 import UAVMECEnv

# --- 步骤 3 (教程): 注册环境 ---
print("正在注册环境: UAVMECEnv...")
# 注意：这里的 Key 必须和 yaml 里的 env_name 一致
REGISTRY_MULTI_AGENT_ENV["UAVMECEnv"] = UAVMECEnv

def parse_args():
    parser = argparse.ArgumentParser("UAV MEC MADDPG")
    parser.add_argument("--test", type=int, default=0)
    return parser.parse_args()

def run():
    cmd_args = parse_args()
    
    # --- 步骤 2 (教程): 读取配置 ---
    # 请确保 yaml 文件名正确
    config_dict = get_configs(file_dir="uav_maddpg.yaml")
    configs = argparse.Namespace(**config_dict)
    configs.test_mode = True if cmd_args.test else False

    # --- 步骤 4 (教程): 创建环境与智能体 ---
    print("正在创建环境...")
    envs = make_envs(configs)
    
    print(f"环境创建成功。Agents: {envs.agents}")
    
    print("正在初始化 MADDPG Agents...")
    agents = MADDPG_Agents(config=configs, envs=envs)

    # --- 训练 ---
    if not configs.test_mode:
        print(f"开始训练... 总步数: {configs.running_steps}")
        agents.train(configs.running_steps // configs.parallels)
        print("训练结束，保存模型...")
        agents.save_model("final_model.pth")
        
    # --- 测试 ---
    else:
        print("开始测试...")
        # 定义测试环境生成函数
        def env_fn():
            configs.parallels = 1
            return make_envs(configs)

        model_path = os.path.join(configs.model_dir, configs.env_id)
        agents.load_model(model_path)
        
        # 使用 agents.test 接口进行标准化测试
        # 这会使用 env_fn 创建新环境并运行 test_episode 次
        scores = agents.test(env_fn, configs.test_episode)
        print(f"测试平均得分: {np.mean(scores)}")
        
    # 结束
    envs.close()
    if not configs.test_mode:
        agents.finish()

if __name__ == "__main__":
    run()