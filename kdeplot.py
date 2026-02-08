import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from scipy.stats import ks_2samp
import matplotlib.ticker as ticker
from shapely.geometry import LineString


# ==========================================
# 1. 核心计算工具函数
# ==========================================

def haversine(lat1, lon1, lat2, lon2):
    """计算两点间的哈拉维距离 (km)"""
    R = 6371  # 地球半径 km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def extract_properties(df, gps_col, time_col, is_raw=True):
    """从原始或合成数据中提取里程和时长"""
    distances = []
    times = []
    
    for _, row in df.iterrows():
        try:
            raw_gps_str = str(row[gps_col])
            raw_time_str = str(row[time_col])
            
            # --- 解析 GPS 坐标 ---
            if is_raw:
                # 真实数据格式: 'lon,lat;lon,lat'
                pts_str = raw_gps_str.split(';')
                pts = [[float(p.split(',')[1]), float(p.split(',')[0])] for p in pts_str]
            else:
                # 合成数据格式混乱，使用正则提取所有数字
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_gps_str)
                nums = [float(n) for n in nums]
                # 每两个数字成一对 [lat, lon]
                pts = [nums[i:i+2] for i in range(0, len(nums), 2)]
            
            if len(pts) < 2: continue

            # --- 解析时间戳 ---
            # 正则匹配 ISO 格式: 2025-08-10T06:51:56Z
            t_list = re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", raw_time_str)
            
            if len(t_list) < 2: continue

            # --- 计算指标 ---
            # 1. 总里程 (km)
            total_dist = 0
            for i in range(len(pts)-1):
                total_dist += haversine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
            
            # 2. 总时长 (min)
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            t_start = datetime.strptime(t_list[0], fmt)
            t_end = datetime.strptime(t_list[-1], fmt)
            duration = (t_end - t_start).total_seconds() / 60.0
            
            # 简单数据清洗：排除异常值（如负时长或超长距离）
            if duration > 0 and total_dist > 0:
                distances.append(total_dist)
                times.append(duration)
                
        except Exception:
            continue
        
    return np.array(distances), np.array(times)

def road_rate_to_gps(road_id, road_rate, geo):
    """根据路段ID和在该路段上的行走比例计算GPS坐标
    
    Args:
        road_id: 路段ID
        road_rate: 在路段上的相对位置 (0-1之间，0表示起点，1表示终点)
        geo: 包含路段信息的DataFrame
    
    Returns:
        (lat, lon): GPS坐标 (纬度, 经度)
    """
    # 获取路段的坐标列表（road_id是geo的索引）
    coordinates_str = geo.loc[road_id, 'coordinates']
    coordinates = eval(coordinates_str) if isinstance(coordinates_str, str) else coordinates_str
    
    # 创建LineString对象
    road_line = LineString(coordinates=coordinates)
    
    # 根据rate计算在路段上的距离（从起点开始）
    # road_rate为0-1之间，0表示起点，1表示终点
    distance_along_line = road_line.length * road_rate
    
    # 使用interpolate方法获取对应距离的点
    point = road_line.interpolate(distance_along_line)
    
    # 返回 (lat, lon) 格式
    return (point.y, point.x)

def extract_properties_road(df, road_id_col, time_col, rate_col, save_path=None):
    """从原始或合成数据中提取里程和时长"""
    distances = []
    times = []
    df_gps = []

    geo = pd.read_csv('data/Tongzhou/roadmap.geo')
    
    for _, row in df.iterrows():
        trace_road_id = eval(row[road_id_col])
        trace_datetime = eval(row[time_col])
        trace_rate = eval(row[rate_col])
        trace_gps = [road_rate_to_gps(road_id, rate, geo) for road_id, rate in zip(trace_road_id, trace_rate)]
        trace_datetime = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in trace_datetime]
        gps_str = ';'.join([f'{gps[1]},{gps[0]}' for gps in trace_gps])
        df_gps.append(gps_str)

        total_distance = 0
        total_time = 0
        for i in range(len(trace_road_id)-1):
            distance = haversine(trace_gps[i][0], trace_gps[i][1], trace_gps[i+1][0], trace_gps[i+1][1])
            time = (trace_datetime[i+1] - trace_datetime[i]).total_seconds() / 60
            total_distance += distance
            total_time += time
        distances.append(total_distance)
        times.append(total_time)
    # print(distances, times)
    if save_path is not None:
        df['gps_list'] = df_gps
        df.to_csv(save_path, index=False)

    return np.array(distances), np.array(times)

# ==========================================
# 2. 数据加载与处理
# ==========================================

print("正在读取数据并计算指标...")
path_test = 'data/Tongzhou/train.csv'
path_gene = 'gene/Tongzhou/seed0/2026-01-23_17-58-14.csv'

df_raw = pd.read_csv(path_test)
df_gene = pd.read_csv(path_gene)

dist_raw, time_raw = extract_properties(df_raw, 'gps_list', 'time_list', is_raw=True)
dist_gene, time_gene = extract_properties_road(df_gene, 'gene_trace_road_id', 'gene_trace_datetime', 'gene_trace_rate', save_path='gene/Tongzhou/seed0/gene_dist_time.csv')

# dist_raw, time_raw = dist_gene, time_gene
# dist_gene, time_gene = dist_raw, time_raw

# --- 【关键修改 1】数据清洗：过滤掉极端的异常值 ---
def remove_outliers(data_raw, data_gene, percentile=99):
    """
    根据原始数据的分布，截断两个数据集的尾部异常值。
    通常保留 99% 的数据能让分布图更美观且不失真。
    """
    # 以真实数据的分布为基准确定上限
    upper_bound = np.percentile(data_raw, percentile)
    # 也可以给一点余量，例如上限的 1.2 倍
    limit = upper_bound * 1.2
    
    # 过滤数据
    clean_raw = data_raw[data_raw <= limit]
    clean_gene = data_gene[data_gene <= limit]
    
    return clean_raw, clean_gene, limit

# 清洗 Distance (保留99%分位数)
c_dist_raw, c_dist_gene, dist_limit = remove_outliers(dist_raw, dist_gene, percentile=99)

# 清洗 Time (保留99%分位数)
c_time_raw, c_time_gene, time_limit = remove_outliers(time_raw, time_gene, percentile=99)

# 计算 K-S 相似度 (建议使用清洗后的数据计算，或者根据你的需求决定)
# 这里展示基于清洗后数据的相似度，更能反映主体分布的一致性
ks_dist = 1 - ks_2samp(c_dist_raw, c_dist_gene).statistic
ks_time = 1 - ks_2samp(c_time_raw, c_time_gene).statistic

# ==========================================
# 3. 绘图 (优化版)
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 配色方案
color_gene = '#8da0cb' 
color_raw = '#fc8d62'

# 通用绘图参数
kde_kws = {
    "fill": True, 
    "alpha": 0.5, 
    "linewidth": 2, 
    "edgecolor": 'black',
    "cut": 0  # 【关键修改 2】cut=0 确保曲线不会画到 0 以下
}

# --- 图(1): Travel Distance 分布 ---
sns.kdeplot(c_dist_gene, color=color_gene, label='Synthetic', ax=axes[0], **kde_kws)
sns.kdeplot(c_dist_raw, color=color_raw, label='Original', ax=axes[0], **kde_kws)

axes[0].set_xlabel('Travel distance (km)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
axes[0].set_xlim(0, dist_limit) # 限制X轴范围
axes[0].text(0.5, 0.8, f'Similarity: {ks_dist:.2f}', transform=axes[0].transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
axes[0].legend()

# --- 图(2): Travel Time 分布 ---
sns.kdeplot(c_time_gene, color=color_gene, label='Synthetic', ax=axes[1], **kde_kws)
sns.kdeplot(c_time_raw, color=color_raw, label='Original', ax=axes[1], **kde_kws)

axes[1].set_xlabel('Travel time (min)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Density', fontsize=12, fontweight='bold')
axes[1].set_xlim(0, time_limit) # 限制X轴范围
axes[1].text(0.5, 0.8, f'Similarity: {ks_time:.2f}', transform=axes[1].transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
axes[1].legend()

# 整体美化
plt.suptitle("(b) Trajectory properties", y=0.02, fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

plt.savefig("distance_duration_kde.png", dpi=300, bbox_inches='tight')
print("图片已优化并保存。")
plt.show()