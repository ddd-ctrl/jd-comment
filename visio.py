import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import PercentFormatter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def draw_health_dashboard():
    fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'polar': True})
    
    # 绘制三个指标仪表盘
    metrics = [
        ('商家投诉率', 0.12, '#F44336'), 
        ('用户复购率', 0.83, '#4CAF50'),
        ('纠纷解决时效', 0.68, '#2196F3')
    ]

    for i, (name, value, color) in enumerate(metrics):
        # 创建子图位置
        ax = fig.add_subplot(1,3,i+1, polar=True)
        
        # 绘制仪表盘
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0,1)
        
        # 刻度环
        ax.plot([0, 2*np.pi], [1,1], color='gray', lw=1)
        ax.fill_between(np.linspace(0,2*np.pi,100), 0,1, 
                       color='#EEEEEE', alpha=0.8)
        
        # 指针
        theta = value * 2 * np.pi
        ax.plot([theta, theta], [0.9,1.1], color=color, lw=3)
        
        # 数值标注
        ax.text(np.pi, 0.5, f'{value:.0%}', ha='center', 
               fontsize=20, color=color)
        ax.set_title(name, pad=20)
        ax.axis('off')

    plt.suptitle('平台生态健康度监控仪表盘', y=0.95)
    plt.tight_layout()
    plt.show()

draw_health_dashboard()