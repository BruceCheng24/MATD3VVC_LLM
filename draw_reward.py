import pandas as pd
from scipy.io import savemat

# 加载Excel文件
file_path = 'E:/第一篇重构/实验结果/IEEE33/reconfigure/PG-MATD3-IEEE33-Reconfigure.xlsx'
data = pd.read_excel(file_path, header=None)

# 数据似乎在一个单列中，我们将其分割成多个列
data = data[0].str.split(',', expand=True)

# 提取列名并设置
data.columns = data.iloc[0].apply(lambda x: x.split(':')[0])
data = data[0:]

# 重置索引
data.reset_index(drop=True, inplace=True)

# 清理列名中的空格
data.columns = data.columns.str.strip()

# 提取各列的数值部分
data['episode'] = data['episode'].str.split(':').str[1].astype(int)
data['noise_std'] = data['noise_std'].str.split(':').str[1].astype(float)
data['reward'] = data['reward'].str.split(':').str[1].astype(float)
data['loss'] = data['loss'].str.split(':').str[1].astype(float)

# 筛选出1到4000的episodes
filtered_data = data[data['episode'].between(1, 2000)]

# 准备要保存的数据
mat_data = {
    'reward': filtered_data['reward'].to_numpy(),
    'loss': filtered_data['loss'].to_numpy()
}

# 保存为.mat文件
savemat_file_path = 'E:/第一篇重构/实验结果/IEEE33/reconfigure/PG-MATD3-IEEE33-Reconfigure.mat'
savemat(savemat_file_path, mat_data)

print("Data has been saved to 'PG-MATD3-IEEE33-NO-Reconfigure.mat'")