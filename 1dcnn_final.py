import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os
from scipy.interpolate import interp1d
import shap
import copy
import shutil

# ================= 1. 全局配置区域 =================
DATA_FILE = r"D:\PycharmProject\pytorch\libs\final_dataset\Final_merged_Dataset_1.csv"
OUTPUT_DIR = r"D:\PycharmProject\pytorch\libs\model_results_Rigorous_v2"

NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 1000
PATIENCE = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDS = 5

COUNTRY_TRANSLATION = {
    "China": "China",
    "India": "India",
    "Russia": "Russia",
    "Brazil": "Brazil",
    "South Africa": "South Africa",
    "Indonesia": "Indonesia",
    "Australia": "Australia",
    "The Philippines": "Philippines",
    "Canada": "Canada"
}


# ================= 2. 核心模块：光谱数据增强 =================
class SpectraAugment:
    def __init__(self, p=0.5, noise_std=0.002, scale_limit=0.05, shift_limit=3):
        self.p = p
        self.noise_std = noise_std
        self.scale_limit = scale_limit
        self.shift_limit = shift_limit

    def __call__(self, x):
        if np.random.rand() > self.p:
            return x
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        scale_factor = 1.0 + np.random.uniform(-self.scale_limit, self.scale_limit)
        x = x * scale_factor
        shift = np.random.randint(-self.shift_limit, self.shift_limit + 1)
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
            if shift > 0:
                x[:, :shift] = 0
            elif shift < 0:
                x[:, shift:] = 0
        return x


# ================= 3. 数据集类 =================
class LIBSDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        if self.transform:
            sample = self.transform(sample.clone())
        return sample, label


# ================= 4. 模型与其他工具类 =================
class SpectralCNN(nn.Module):
    def __init__(self, num_classes, input_length):
        super(SpectralCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, 1, 2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.flatten_dim = self._get_flatten_dim(input_length)
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _get_flatten_dim(self, length):
        dummy_input = torch.zeros(1, 1, length)
        with torch.no_grad():
            output = self.features(dummy_input)
        return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)


def plot_training_history_robust(history, output_dir, fold_idx):
    min_len = min(len(v) for v in history.values())
    if min_len == 0: return
    clean_history = {k: v[:min_len] for k, v in history.items()}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(clean_history['train_loss'], label='Train Loss')
    plt.plot(clean_history['val_loss'], label='Val Loss')
    plt.title(f'Loss Curve (Fold {fold_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(clean_history['train_acc'], label='Train Acc')
    plt.plot(clean_history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy Curve (Fold {fold_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"training_curves_fold_{fold_idx}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    pd.DataFrame(clean_history).to_csv(os.path.join(output_dir, f"training_history_fold_{fold_idx}.csv"), index=False)
    return filename


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if total == 0: return 0.0, 0.0
    return running_loss / total, 100.0 * correct / total


def run_shap_analysis(model, train_loader, test_loader, device, wavelengths, class_names, output_dir, top_n=3):
    """
    修正版 SHAP 分析函数：
    1. 自动适配不同的 SHAP 输出格式。
    2. 生成 SHAP 图。
    3. 🔥 输出 CSV 表格：每类前 top_n 个样本，分别输出 'All Values' 和 'Positive Only'。
    """
    print(f"\n[SHAP] 正在初始化 SHAP 分析...")
    try:
        # 1. 初始化 Explainer
        batch_X, _ = next(iter(train_loader))
        background = batch_X[:50].to(device)
        explainer = shap.DeepExplainer(model, background)

        candidates = {i: [] for i in range(len(class_names))}
        model.eval()

        # 2. 筛选高置信度样本
        print(" -> 正在筛选各类别的高置信度样本...")
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = preds[i].item()
                    if true_label == pred_label:
                        candidates[true_label].append({
                            'prob': probs[i, pred_label].item(),
                            'input': inputs[i:i + 1]
                        })

        # 3. 辅助函数：智能提取 SHAP 值
        def extract_shap_values(shap_raw, cls_idx, expected_len):
            if isinstance(shap_raw, list):
                if len(shap_raw) == len(class_names):
                    data = shap_raw[cls_idx]
                else:
                    data = shap_raw[0]
            else:
                data = np.array(shap_raw)

            flat = data.flatten()
            total_points = flat.shape[0]

            if total_points == expected_len:
                return flat
            if data.ndim >= 3 and data.shape[-1] == len(class_names):
                return data[0, ..., cls_idx].flatten()
            elif data.ndim >= 3 and data.shape[1] == len(class_names):
                return data[0, cls_idx, ...].flatten()
            if total_points == expected_len * len(class_names):
                try:
                    reshaped = flat.reshape(expected_len, len(class_names))
                    return reshaped[:, cls_idx]
                except:
                    pass
            return None

        # 4. 逐类别生成图表和 CSV
        shap_csv_dir = os.path.join(output_dir, "SHAP_Data_CSVs")
        if not os.path.exists(shap_csv_dir):
            os.makedirs(shap_csv_dir)

        for cls_idx in range(len(class_names)):
            cls_name = class_names[cls_idx]
            cls_name_en = COUNTRY_TRANSLATION.get(cls_name, cls_name)

            sorted_candidates = sorted(candidates[cls_idx], key=lambda x: x['prob'], reverse=True)
            top_candidates = sorted_candidates[:top_n]  # 取前3个样本

            if not top_candidates:
                continue

            print(f" -> 处理 {cls_name_en} (Top {len(top_candidates)})...")

            for rank, item in enumerate(top_candidates):
                sample_tensor = item['input']

                try:
                    # 计算 SHAP 值
                    shap_values_raw = explainer.shap_values(sample_tensor, check_additivity=False)

                    # 提取该类别的 Raw SHAP 值（包含正负）
                    sv_raw = extract_shap_values(shap_values_raw, cls_idx, len(wavelengths))

                    if sv_raw is None:
                        print(f"    ⚠️ 无法解析 SHAP 数据形状: {np.array(shap_values_raw).shape}")
                        continue

                    if sv_raw.shape[0] != len(wavelengths):
                        print(f"    ⚠️ 维度仍不匹配: Got {sv_raw.shape[0]}, Expected {len(wavelengths)}")
                        continue

                    # 处理2：仅正值
                    sv_positive = np.maximum(sv_raw, 0)

                    # 原始光谱数据
                    orig_spec = sample_tensor.cpu().numpy().flatten()

                    # 🔥 导出 CSV 1: 全部值 (All)
                    csv_name_all = f"SHAP_Values_{cls_name_en}_Sample{rank + 1}_All.csv"
                    df_all = pd.DataFrame({
                        'Wavelength': wavelengths,
                        'Original_Intensity': orig_spec,
                        'SHAP_Value': sv_raw
                    })
                    df_all.to_csv(os.path.join(shap_csv_dir, csv_name_all), index=False)

                    # 🔥 导出 CSV 2: 仅正值 (Positive)
                    csv_name_pos = f"SHAP_Values_{cls_name_en}_Sample{rank + 1}_Positive.csv"
                    df_pos = pd.DataFrame({
                        'Wavelength': wavelengths,
                        'Original_Intensity': orig_spec,
                        'SHAP_Value': sv_positive
                    })
                    df_pos.to_csv(os.path.join(shap_csv_dir, csv_name_pos), index=False)

                    # 绘图 (仅展示正值)
                    plt.figure(figsize=(14, 6))
                    ax1 = plt.gca()
                    ax1.plot(wavelengths, orig_spec, 'k-', alpha=0.3, label='Original Spectrum', linewidth=1)
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('Normalized Intensity')

                    ax2 = ax1.twinx()
                    ax2.bar(wavelengths, sv_positive, color='red', alpha=0.6, width=(wavelengths[1] - wavelengths[0]),
                            label='SHAP Importance')
                    ax2.set_ylabel('SHAP Value (Positive)', color='red')

                    lines, labels = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

                    plt.title(f'SHAP Interpretation: {cls_name_en} | Rank {rank + 1} (Conf: {item["prob"]:.4f})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"SHAP_{cls_name_en}_Rank{rank + 1}.png"), dpi=300)
                    plt.close()

                except Exception as e:
                    print(f"    ❌ 失败 {cls_name_en}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        print(f"✅ SHAP CSV 数据已保存至: {shap_csv_dir}")

    except Exception as e:
        print(f"❌ SHAP 分析初始化失败: {e}")


# ================= 5. 主程序 =================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"当前设置: Device={DEVICE}, Batch={BATCH_SIZE}, Folds={FOLDS}")
    print(f"数据路径: {DATA_FILE}")

    # 1. 读取数据
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return

    # 解析数据结构
    y_raw = df.iloc[:, 0].values
    sample_names_raw = df.iloc[:, 1].values.astype(str)
    X_raw = df.iloc[:, 2:].values.astype(np.float32)
    wavelengths = df.columns[2:].astype(float)
    input_length = X_raw.shape[1]

    # 标签编码
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_names = le.classes_
    english_labels = [COUNTRY_TRANSLATION.get(name, name) for name in class_names]

    print(f"样本总数 (Spectra): {len(X_raw)}")
    print(f"独立煤样数 (Samples): {len(np.unique(sample_names_raw))}")
    print(f"类别: {english_labels}")

    # 2. 准备交叉验证
    sgkf = StratifiedGroupKFold(n_splits=FOLDS)

    fold_metrics = {'Fold': [], 'Train Acc': [], 'Val Acc': [], 'Train Loss': [], 'Val Loss': []}
    best_fold_val_acc = 0.0
    best_fold_idx = -1
    best_model_state = None
    best_train_indices = None
    best_val_indices = None

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_raw, y_enc, groups=sample_names_raw)):
        print(f"\n=== Fold {fold + 1}/{FOLDS} ===")

        X_train, X_val = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_enc[train_idx], y_enc[val_idx]

        # 验证是否泄露
        train_samples = set(sample_names_raw[train_idx])
        val_samples = set(sample_names_raw[val_idx])
        overlap = train_samples.intersection(val_samples)
        if len(overlap) > 0:
            raise ValueError(f"❌ 严重错误：Fold {fold + 1} 存在数据泄露！Samples: {overlap}")
        else:
            print(f"  ✅ 数据隔离检查通过。训练集样品数: {len(train_samples)}, 验证集样品数: {len(val_samples)}")

        # 计算权重
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print(f"  -> Class Weights: {np.round(class_weights, 3)}")

        # 构建数据集
        augmentor = SpectraAugment(p=0.5, noise_std=0.002, scale_limit=0.05)
        train_ds = LIBSDataset(X_train, y_train, transform=augmentor)
        train_ds_eval = LIBSDataset(X_train, y_train, transform=None)
        val_ds = LIBSDataset(X_val, y_val, transform=None)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_eval = DataLoader(train_ds_eval, batch_size=BATCH_SIZE, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型
        model = SpectralCNN(num_classes=NUM_CLASSES, input_length=input_length).to(DEVICE)

        # 加权 Loss
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # 早停
        fold_save_path = os.path.join(OUTPUT_DIR, f"model_fold_{fold + 1}.pth")
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=False, path=fold_save_path)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        # --- 训练循环 ---
        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss, train_acc = evaluate_model(model, train_loader_eval, criterion, DEVICE)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch [{epoch + 1}/{EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Loss: {val_loss:.4f}")

            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"  🚀 Fold {fold + 1} 早停于 Epoch {epoch + 1}")
                break

        plot_training_history_robust(history, OUTPUT_DIR, fold + 1)

        model.load_state_dict(torch.load(fold_save_path))
        final_train_loss, final_train_acc = evaluate_model(model, train_loader_eval, criterion, DEVICE)
        final_val_loss, final_val_acc = evaluate_model(model, val_loader, criterion, DEVICE)

        print(f"  ✅ Fold {fold + 1} 最终: Val Acc: {final_val_acc:.2f}% (Train: {final_train_acc:.2f}%)")

        fold_metrics['Fold'].append(fold + 1)
        fold_metrics['Train Acc'].append(final_train_acc)
        fold_metrics['Val Acc'].append(final_val_acc)
        fold_metrics['Train Loss'].append(final_train_loss)
        fold_metrics['Val Loss'].append(final_val_loss)

        if final_val_acc > best_fold_val_acc:
            best_fold_val_acc = final_val_acc
            best_fold_idx = fold
            best_model_state = copy.deepcopy(model.state_dict())
            best_train_indices = train_idx
            best_val_indices = val_idx

    # --- 总结 ---
    print("\n" + "=" * 50)
    print("📊 训练总结 (基于独立样本划分)")
    df_metrics = pd.DataFrame(fold_metrics)
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "kfold_summary.csv"), index=False)

    # --- 最佳模型详细分析 ---
    print(f"\n🏆 分析最佳模型: Fold {best_fold_idx + 1}")
    final_model = SpectralCNN(num_classes=NUM_CLASSES, input_length=input_length).to(DEVICE)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    X_best_val = X_raw[best_val_indices]
    y_best_val = y_enc[best_val_indices]
    best_val_loader = DataLoader(LIBSDataset(X_best_val, y_best_val, transform=None), batch_size=BATCH_SIZE,
                                 shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in best_val_loader:
            inputs = inputs.to(DEVICE)
            outputs = final_model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    annot_labels = [[f"{v}\n({p:.1%})" for v, p in zip(row_v, row_p)] for row_v, row_p in zip(cm, cm_norm)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', xticklabels=english_labels, yticklabels=english_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Sample-Level Split)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_best.png"), dpi=300)

    report = classification_report(all_labels, all_preds, target_names=english_labels, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(OUTPUT_DIR, "classification_report_best.csv"))

    # SHAP
    X_best_train = X_raw[best_train_indices]
    y_best_train = y_enc[best_train_indices]
    best_train_loader = DataLoader(LIBSDataset(X_best_train, y_best_train, transform=None), batch_size=BATCH_SIZE,
                                   shuffle=True)
    # 调用修改后的 SHAP 函数，top_n=3
    run_shap_analysis(final_model, best_train_loader, best_val_loader, DEVICE, wavelengths, class_names, OUTPUT_DIR,
                      top_n=3)

    print("✅ 程序执行完毕。SHAP CSV 和图像已保存。")


if __name__ == '__main__':
    main()