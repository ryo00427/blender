import json
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# データセットのパスを指定
dataset_path = r''  # JSONファイルのパス
image_folder = r''  # 画像フォルダのパス

# JSONファイルを読み込む
with open(dataset_path, 'r') as f:
    dataset_info = json.load(f)

# カメラパラメータを設定
camera_angle_x = dataset_info["camera_angle_x"]
width, height = 512, 512  # 画像サイズを指定
focal_length = 0.5 * width / np.tan(0.5 * camera_angle_x)  # 焦点距離を計算
cx, cy = 0.5 * width, 0.5 * height  # 画像の中心座標
frames = dataset_info["frames"]  # フレーム情報を取得

# カメラパラメータからレイを生成する関数
def camera_params_to_rays(f, cx, cy, pose, width, height):
    v, u = np.mgrid[:height, :width].astype(np.float32)
    _x = (u - cx) / f
    _y = (v - cy) / f
    _z = np.ones_like(_x)
    _w = np.ones_like(_x)
    _d = np.stack([_x, _y, _z, _w], axis=2)
    _o = np.zeros((height, width, 4), dtype=np.float32)
    _o[:, :, 3] = 1
    o = (pose @ _o[..., None])[..., :3, 0]
    d = (pose @ _d[..., None])[..., :3, 0] - o
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return o.reshape(-1, 3), d.reshape(-1, 3)

# NeRF用のデータセットクラス
target="_blank"class NeRFDataset(Dataset):
    def __init__(self, frames, f, cx, cy, width, height, image_folder, num_rays=1024):
        self.frames = frames
        self.f = f
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.image_folder = image_folder
        self.num_rays = num_rays

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        image_path = os.path.join(self.image_folder, os.path.basename(frame['file_path']) + '.png')
        rgb = np.array(Image.open(image_path).convert('RGB')) / 255.0
        indices = np.random.choice(self.width * self.height, self.num_rays, replace=False)
        o, d = camera_params_to_rays(self.f, self.cx, self.cy, pose, self.width, self.height)
        return torch.tensor(o[indices]), torch.tensor(d[indices]), torch.tensor(rgb.reshape(-1, 3)[indices], dtype=torch.float32)

# 放射場（Radiance Field）モデル
target="_blank"class RadianceField(nn.Module):
    def __init__(self):
        super(RadianceField, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# NeRFモデル
target="_blank"class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.coarse_rf = RadianceField()
        self.fine_rf = RadianceField()

    def volume_rendering(self, model, o, d, num_samples=32):
        t_vals = torch.linspace(0, 1, num_samples).cuda()
        z_vals = t_vals[None, None, :] * 4.0
        z_vals = z_vals.expand(o.shape[0], o.shape[1], num_samples)
        pts = o[:, :, None, :] + d[:, :, None, :] * z_vals[..., None]
        pts = pts.reshape(-1, 3)
        rgb_sigma = model(pts)
        rgb = torch.sigmoid(rgb_sigma[:, :3])
        rgb = rgb.reshape(o.shape[0], o.shape[1], num_samples, 3)
        return torch.sum(rgb, dim=2) / num_samples

    def forward(self, o, d):
        coarse_output = self.volume_rendering(self.coarse_rf, o, d)
        fine_output = self.volume_rendering(self.fine_rf, o, d)
        return coarse_output, fine_output

# データセットとデータローダーを作成
dataset = NeRFDataset(frames, focal_length, cx, cy, width, height, image_folder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# モデルを定義し、学習用の設定を行う
model = NeRF().float().cuda()
optimizer = optim.Adam(model.parameters(), lr=3e-4)
n_epoch = 10

# 学習ループ
for epoch in range(n_epoch):
    total_loss = 0
    for i, (o, d, rgb) in enumerate(dataloader):
        o, d, rgb = o.cuda().float(), d.cuda().float(), rgb.cuda().float()
        optimizer.zero_grad()
        coarse_output, fine_output = model(o, d)
        loss = torch.nn.functional.mse_loss(coarse_output, rgb) + torch.nn.functional.mse_loss(fine_output, rgb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{n_epoch}, Loss: {total_loss / len(dataloader)}")

# 学習したモデルを保存
torch.save(model.state_dict(), 'model.pth')
