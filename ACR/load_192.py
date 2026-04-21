import numpy as np

# 假设你已经用 np.load 加载了文件
data = np.load('/home/lucas/whole_body_tracking/artifacts/tmp/walk_paired.npz') # 或者对应的格式

# 获取特征块的名称和它们的切片范围
block_names = data['feature_block_names']
block_offsets = data['feature_block_offsets']

print("--- 192维特征向量构成揭秘 ---")
for i in range(len(block_names)):
    name = block_names[i]
    start, end = block_offsets[i]
    length = end - start
    print(f"块 {i+1}: {name} | 维度大小: {length}维 | 索引范围: [{start}:{end}]")