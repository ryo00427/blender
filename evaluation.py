import torch
import matplotlib.pyplot as plt
import numpy as np
from train import NeRF

# 学習したモデルをロード
model = NeRF().float().cuda()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 画像サイズを定義
width, height = 512, 512
num_rays = width * height  # 画像のピクセル数と同じ光線を生成

# ダミーの光線データを作成
test_o = torch.randn(1, num_rays, 3).cuda()  # バッチサイズを1に変更
test_d = torch.randn(1, num_rays, 3).cuda()

# モデルの出力を得る
with torch.no_grad():
    coarse_output, fine_output = model(test_o, test_d)

# 出力の形状を確認
print("Coarse output shape:", coarse_output.shape)  # 期待: (1, 262144, 3)
print("Fine output shape:", fine_output.shape)      # 期待: (1, 262144, 3)

# 出力の値の範囲を確認
print("Coarse output min:", coarse_output.min().item(), "max:", coarse_output.max().item())
print("Fine output min:", fine_output.min().item(), "max:", fine_output.max().item())

# 各チャンネルごとに個別にスケーリング
coarse_output = (coarse_output - coarse_output.min(dim=1, keepdim=True)[0]) / (coarse_output.max(dim=1, keepdim=True)[0] - coarse_output.min(dim=1, keepdim=True)[0])
fine_output = (fine_output - fine_output.min(dim=1, keepdim=True)[0]) / (fine_output.max(dim=1, keepdim=True)[0] - fine_output.min(dim=1, keepdim=True)[0])

# NumPy に変換
coarse_output_np = coarse_output[0].cpu().numpy().astype(np.float32)
fine_output_np = fine_output[0].cpu().numpy().astype(np.float32)

# 画像サイズにリシェイプ
coarse_output_image = coarse_output_np.reshape(height, width, 3)  # 最初のサンプル
fine_output_image = fine_output_np.reshape(height, width, 3)

# 画像の表示
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.clip(coarse_output_image, 0, 1))  # 0-1 にクリップ
axes[0].set_title("Coarse Output")
axes[0].axis('off')

axes[1].imshow(np.clip(fine_output_image, 0, 1))
axes[1].set_title("Fine Output")
axes[1].axis('off')

plt.show()