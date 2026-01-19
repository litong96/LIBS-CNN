import pandas as pd
import numpy as np
import os
import glob

# ================= 配置区域 =================
# 输入数据的文件夹路径 (请根据实际情况修改)
INPUT_FOLDER = r'D:\PycharmProject\pytorch\libs\denoised_results\WT'
# 输出结果的保存路径
OUTPUT_FOLDER = r'D:\PycharmProject\pytorch\libs\normalized_results'

# 碳峰的理论波长 (nm)
TARGET_C_WAVELENGTH = 247.856


# ===========================================

def load_data(folder_path):
    """
    读取文件夹下的所有CSV文件。
    兼容两种模式：
    1. 单个大文件 (Rows=Samples, Cols=Wavelengths)
    2. 多个小文件 (每个文件是一个样品) -> 这种情况代码会自动合并
    这里默认处理我们之前的格式：文件名在第一列，后续是光谱数据
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"在 {folder_path} 中未找到 CSV 文件")

    # 假设是读取之前生成的那个大文件
    # 如果有多个文件，这里默认读取第一个作为演示，或者你需要循环处理
    # 为了通用，我们循环读取所有文件并分别处理
    print(f"找到 {len(all_files)} 个文件...")
    return all_files


def find_closest_wavelength(columns, target):
    """在列名中找到最接近 target 的波长"""
    # 排除非数字列 (如 Filename)
    numeric_cols = [c for c in columns if c.replace('.', '', 1).isdigit()]
    waves = np.array([float(c) for c in numeric_cols])

    idx = np.argmin(np.abs(waves - target))
    closest_wave = numeric_cols[idx]
    return closest_wave


def normalize_and_save(file_path, output_base):
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取错误 {filename}: {e}")
        return

    # 分离元数据和光谱数据
    # 假设第一列是 'Filename' 或其他标签，后续是数值
    data_col_start_idx = 1 if not df.columns[0].replace('.', '', 1).isdigit() else 0
    meta_data = df.iloc[:, :data_col_start_idx]
    spectra = df.iloc[:, data_col_start_idx:].astype(float)
    columns = spectra.columns

    # 1. Min-Max Normalization (最大最小归一化)
    # 公式: x_new = (x - min) / (max - min)
    min_vals = spectra.min(axis=1)
    max_vals = spectra.max(axis=1)
    # 防止除以0
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    norm_minmax = spectra.sub(min_vals, axis=0).div(range_vals, axis=0)

    # 2. Z-score Normalization (标准正态变量变换, SNV)
    # 公式: x_new = (x - mean) / std
    mean_vals = spectra.mean(axis=1)
    std_vals = spectra.std(axis=1)
    std_vals[std_vals == 0] = 1
    norm_zscore = spectra.sub(mean_vals, axis=0).div(std_vals, axis=0)

    # 3. Global Normalization (这里采用 L2 范数归一化，即向量归一化)
    # 公式: x_new = x / sqrt(sum(x^2))
    l2_norm = np.sqrt((spectra ** 2).sum(axis=1))
    l2_norm[l2_norm == 0] = 1
    norm_global = spectra.div(l2_norm, axis=0)

    # 4. Carbon Peak Normalization (碳峰归一化)
    # 公式: x_new = x / I_carbon
    c_col = find_closest_wavelength(columns, TARGET_C_WAVELENGTH)
    c_intensities = spectra[c_col]
    # 防止碳峰检测不到（为0）的情况
    c_intensities[c_intensities == 0] = 1
    norm_carbon = spectra.div(c_intensities, axis=0)

    # 保存结果
    methods = {
        'MinMax': norm_minmax,
        'ZScore': norm_zscore,
        'Global_L2': norm_global,
        'Carbon_Norm': norm_carbon
    }

    for method_name, res_df in methods.items():
        # 重组 DataFrame
        final_df = pd.concat([meta_data, res_df], axis=1)

        # 创建对应的子文件夹
        save_dir = os.path.join(output_base, method_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"norm_{method_name}_{filename}")
        final_df.to_csv(save_path, index=False)
        print(f"[{method_name}] 已保存: {save_path}")


# ================= 执行逻辑 =================
if __name__ == '__main__':
    csv_files = load_data(INPUT_FOLDER)
    for f in csv_files:
        normalize_and_save(f, OUTPUT_FOLDER)
    print("\n所有归一化处理完成！")