import scipy.io as sio
import numpy as np
import sys
import pandas as pd
from typing import Optional, Tuple
import openai

openai.api_key = "Your OpenAI API"  # 设置你的OpenAI API密钥
# GPT数据生成程序
class IEEE33_ENV:
    def __init__(self):
        np.set_printoptions(threshold=sys.maxsize)  # 控制小数精度为最大值
        pd.set_option("display.max_rows", None, "display.max_columns", None)  # 控制输出全部行列
        self.data_loaded = False
        self.generate_and_save_data()  # 初始化时生成并保存数据

    def gpt_generate(self, prompt: str) -> np.ndarray:
        chat_completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an AI assistant specialized in generating output data of wind turbines, photovoltaics, and load data for offline training of deep reinforcement learning."},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            response = chat_completion.choices[0].message['content'].strip()
            # 提取所有数字部分，并转换为浮点数列表
            data = np.array([float(x) for x in response.split(",")])

            # 检查数据长度并进行处理
            if len(data) > 24:
                data = data[:24]  # 取前24个数据
            elif len(data) < 24:
                # 补充缺失的数据，使用0填充
                data = np.pad(data, (0, 24 - len(data)), 'constant', constant_values=0)

            return data
        except ValueError:
            # 如果转换失败，返回一个默认值或其他处理逻辑
            print(f"Error converting response to float array: '{response}'")
            return np.zeros(24)

    def generate_and_save_data(self):
        if self.data_loaded:
            return

        data_DER = {
            "fengji1": [],
            "fengji2": [],
            "fengji3": [],
            "solar1": [],
            "solar2": [],
            "solar3": []
        }

        for i in range(1):  # 生成100个训练回合
            print(f"Generating data for round {i + 1}...")
            prompts = {
                "fengji1": f"Generate exactly 24 comma-separated floating-point numbers representing wind turbine power for fengji1[0, 0]. There are 3 wind turbines in the IEEE 33 node distribution network with capacities 600, 500, and 550 kW respectively. The data should be grouped in 24 units, each representing 1-hour intervals over 24 hours. Provide realistic output characteristics of the wind turbine with capacities 600kW in kW with an accuracy of three decimal places. Do not provide any additional explanation，only provide data.",
                "fengji2": f"Generate exactly 24 comma-separated floating-point numbers representing wind turbine power for fengji2[0, 0]. There are 3 wind turbines in the IEEE 33 node distribution network with capacities 600, 500, and 550 kW respectively. The data should be grouped in 24 units, each representing 1-hour intervals over 24 hours. Provide realistic output characteristics of the wind turbine with capacities 500kW in kW with an accuracy of three decimal places. Do not provide any additional explanation，only provide data.",
                "fengji3": f"Generate exactly 24 comma-separated floating-point numbers representing wind turbine power for fengji3[0, 0]. There are 3 wind turbines in the IEEE 33 node distribution network with capacities 600, 500, and 550 kW respectively. The data should be grouped in 24 units, each representing 1-hour intervals over 24 hours. Provide realistic output characteristics of the wind turbine with capacities 550kW in kW with an accuracy of three decimal places. Do not provide any additional explanation，only provide data.",
                "solar1": f"Generate exactly 24 comma-separated floating-point numbers representing PV power for solar1[0, 0]. There are 3 photovoltaics in the IEEE 33 node distribution network with capacities 540, 320, and 430 kW respectively. The data should be grouped in 24 units, each representing 1-hour intervals over 24 hours. Provide realistic output characteristics of the photovoltaic with capacities 540kW in kW with an accuracy of three decimal places. Do not provide any additional explanation，only provide data.",
                "solar2": f"Generate exactly 24 comma-separated floating-point numbers representing PV power for solar2[0, 0]. There are 3 photovoltaics in the IEEE 33 node distribution network with capacities 540, 320, and 430 kW respectively. The data should be grouped in 24 units, each representing 1-hour intervals over 24 hours. Provide realistic output characteristics of the photovoltaic with capacities 320kW in kW with an accuracy of three decimal places. Do not provide any additional explanation，only provide data.",
                "solar3": f"Generate exactly 24 comma-separated floating-point numbers representing PV power for solar3[0, 0]. There are 3 photovoltaics in the IEEE 33 node distribution network with capacities 540, 320, and 430 kW respectively. The data should be grouped in 24 units, each representing 1-hour intervals over 24 hours. Provide realistic output characteristics of the photovoltaic with capacities 430kW in kW with an accuracy of three decimal places. Do not provide any additional explanation，only provide data."
            }

            # 生成每个风机、光伏和负荷的数据
            data_DER["fengji1"].append(self.gpt_generate(prompts["fengji1"]))
            data_DER["fengji2"].append(self.gpt_generate(prompts["fengji2"]))
            data_DER["fengji3"].append(self.gpt_generate(prompts["fengji3"]))
            data_DER["solar1"].append(self.gpt_generate(prompts["solar1"]))
            data_DER["solar2"].append(self.gpt_generate(prompts["solar2"]))
            data_DER["solar3"].append(self.gpt_generate(prompts["solar3"]))

            # 打印生成的数据
            print(f"Round {i + 1} data:")
            print(f"Wind Turbine 1: {data_DER['fengji1'][-1]}")
            print(f"Wind Turbine 2: {data_DER['fengji2'][-1]}")
            print(f"Wind Turbine 3: {data_DER['fengji3'][-1]}")
            print(f"Solar 1: {data_DER['solar1'][-1]}")
            print(f"Solar 2: {data_DER['solar2'][-1]}")
            print(f"Solar 3: {data_DER['solar3'][-1]}")
            print("-" * 50)  # 分隔线

        # 转换为numpy数组
        for key in data_DER:
            data_DER[key] = np.array(data_DER[key])

        # 保存数据到MAT文件
        sio.savemat('data_24_WT_PV.mat', data_DER)

if __name__ == "__main__":
    env = IEEE33_ENV()
