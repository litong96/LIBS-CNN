import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import os
import seaborn as sns

# ================= 配置区域 =================
DATA_FILE = r"D:\PycharmProject\pytorch\libs\final_dataset\Final_Merged_Dataset_1.csv"
OUTPUT_DIR = r"C:\Users\admin\Desktop\提取结果"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5


# ================= 模型定义 (保持原结构，只改提取逻辑) =================
class SpectralCNN(nn.Module):
    def __init__(self, num_classes, input_length):
        super(SpectralCNN, self).__init__()
        # 1. 特征提取部分 (完全保持原样，以兼容旧权重 keys)
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )

        self.flatten_dim = self._get_flatten_dim(input_length)

        # 2. 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_dim(self, length):
        dummy_input = torch.zeros(1, 1, length)
        with torch.no_grad():
            output = self.features(dummy_input)
        return output.view(1, -1).size(1)

    def forward(self, x):
        intermediate_features = {}

        # --- 1. Conv Layers ---
        # 直接通过整个 features 序列 (到达 Layer 3 结束)
        x = self.features(x)

        # 【提取点1：Conv Layer】(取最后一个卷积层的输出)
        # 保持原始的 view (Flatten) 方式
        intermediate_features['Conv_Feature'] = x.view(x.size(0), -1)

        # --- Flatten ---
        x = x.view(x.size(0), -1)

        # --- 2. FC Layer ---
        x = self.classifier[0](x)  # Linear
        fc_feature = self.classifier[1](x)  # ReLU

        # 【提取点2：FC Layer】(语义特征)
        intermediate_features['FC_Feature'] = fc_feature

        # --- 3. Output Layer ---
        x = self.classifier[2](fc_feature)  # Dropout
        out = self.classifier[3](x)  # Linear (Logits)

        # 【提取点3：Output Layer】(最终分类输出)
        intermediate_features['Output_Layer'] = out

        return out, intermediate_features


# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ 目录不存在: {OUTPUT_DIR}")
        return

    print("🚀 正在准备 2x2 全层级 t-SNE 分析 (Raw -> Conv -> FC -> Output)...")

    # 1. 读取数据
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return

    y_raw = df.iloc[:, 0].values
    X_raw = df.iloc[:, 2:].values.astype(np.float32)
    input_length = X_raw.shape[1]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    # 采样逻辑 (保持原始的随机采样)
    SAMPLE_LIMIT = 2280
    if len(X_raw) > SAMPLE_LIMIT:
        print(f" -> 数据量较大 ({len(X_raw)})，随机采样 {SAMPLE_LIMIT} 个样本...")
        indices = np.random.choice(len(X_raw), SAMPLE_LIMIT, replace=False)
        X_sample = X_raw[indices]
        y_sample = y_enc[indices]
        y_labels = y_raw[indices]
    else:
        X_sample = X_raw
        y_sample = y_enc
        y_labels = y_raw

    # 初始化 t-SNE 对象
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)

    # 【图1】Raw Data
    plot_data = {
        'Labels': y_labels,
        'Raw Data': tsne.fit_transform(X_sample)
    }
    print(" ✅ Raw Data t-SNE 完成")

    # 2. 加载模型并提取特征
    model = SpectralCNN(NUM_CLASSES, input_length).to(DEVICE)

    try:
        model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pth')]
        if not model_files: raise FileNotFoundError("未找到 .pth 模型文件")
        model_name = "best_model.pth" if "best_model.pth" in model_files else model_files[0]
        model_path = os.path.join(OUTPUT_DIR, model_name)
        print(f" -> 加载模型: {model_name}")

        model.load_state_dict(torch.load(model_path))
        model.eval()

        # 准备数据 Tensor
        X_tensor = torch.FloatTensor(X_sample).unsqueeze(1).to(DEVICE)

        # 容器：只收集我们需要的3个层
        collected_features = {
            'Conv_Feature': [],
            'FC_Feature': [],
            'Output_Layer': []
        }

        batch_size = 256
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                _, batch_feats = model(batch)

                # 将每个 batch 的特征转回 numpy 并存入列表
                for key in collected_features:
                    collected_features[key].append(batch_feats[key].cpu().numpy())

        # 合并 batch 并计算 t-SNE
        print(" -> 开始计算特征 t-SNE (请耐心等待)...")

        # 映射显示的名称
        layer_display_names = {
            'Conv_Feature': 'Conv Layer (Last)',
            'FC_Feature': 'FC Layer (Dense)',
            'Output_Layer': 'Output Layer (Result)'
        }

        for key, batches in collected_features.items():
            # 拼接
            feat_matrix = np.concatenate(batches, axis=0)
            # 计算 t-SNE
            print(f"    ...计算 {key} 的 t-SNE")
            tsne_result = tsne.fit_transform(feat_matrix)
            # 存入绘图数据
            plot_data[layer_display_names[key]] = tsne_result

    except Exception as e:
        print(f"⚠️ 特征提取或 t-SNE 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 导出数据到 CSV
    print("\n💾 正在导出数据...")
    export_df = pd.DataFrame({'Label': y_labels})
    for name, coords in plot_data.items():
        if name != 'Labels':
            export_df[f'{name}_X'] = coords[:, 0]
            export_df[f'{name}_Y'] = coords[:, 1]

    csv_path = os.path.join(OUTPUT_DIR, "Four_Layers_tSNE_Data.csv")
    export_df.to_csv(csv_path, index=False)
    print(f"✅ 数据已保存: {csv_path}")

    # 4. 绘制 2x2 的多子图
    print("🎨 生成 2x2 演化图...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))  # 改为 2x2 布局
    axes = axes.flatten()

    # 定义绘图顺序
    plot_order = [
        'Raw Data',
        'Conv Layer (Last)',
        'FC Layer (Dense)',
        'Output Layer (Result)'
    ]

    # 调色板
    palette = sns.color_palette("tab10", n_colors=len(np.unique(y_labels)))

    for i, name in enumerate(plot_order):
        ax = axes[i]
        coords = plot_data[name]

        sns.scatterplot(
            x=coords[:, 0], y=coords[:, 1], hue=y_labels,
            palette=palette, s=50, alpha=0.7, ax=ax, legend='full' if i == 0 else False
        )
        ax.set_title(name, fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 给子图加个边框，好看点
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

    # 添加整体标题
    plt.suptitle("Feature Evolution: Raw -> Conv -> FC -> Output", fontsize=20, y=0.96)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "2x2_Evolution.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ 图片已保存: {save_path}")


if __name__ == '__main__':
    main()