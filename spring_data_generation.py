import bpy
import os
import numpy as np
import json

# レンダリングするビューの数と画像の解像度を設定
VIEWS = 100
RESOLUTION = 512

# 結果を保存するパスを設定し、フォルダが存在しない場合は作成
RESULTS_PATH = os.path.expanduser(r"\data")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(os.path.join(RESULTS_PATH, 'train'))

# レンダリング設定
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION  # 画像の横解像度
scene.render.resolution_y = RESOLUTION  # 画像の縦解像度
scene.render.image_settings.file_format = str('PNG')  # 画像フォーマットをPNGに設定
scene.render.film_transparent = True  # 背景を透明に設定
scene.render.use_persistent_data = True  # レンダリングのキャッシュを有効化

# カメラ設定
cam = scene.objects['Camera']
cam.location = (0, -30, 60)  # カメラの初期位置を設定

# カメラの親オブジェクトを作成し、シーンに追加
camera_parent = bpy.data.objects.new("CameraParent", None)
camera_parent.location = (0, 0, 0)
scene.collection.objects.link(camera_parent)
cam.parent = camera_parent  # カメラを親オブジェクトに設定

# カメラにターゲット追尾の制約を追加
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'  # カメラの向きをZ軸負方向に設定
cam_constraint.up_axis = 'UP_Y'  # 上方向をY軸に設定
cam_constraint.target = camera_parent  # カメラが追尾するターゲットを親オブジェクトに設定

# JSONファイルに保存するデータの初期化
out_data = {'camera_angle_x': bpy.data.objects['Camera'].data.angle_x, 'frames': []}

# VIEWS回のレンダリングを実行
for i in range(VIEWS):
    
    # カメラをランダムに回転
    camera_parent.rotation_euler = np.array([
        np.random.uniform(np.pi / 12, np.pi / 2),  # X軸回転（上方向）
        0,  # Y軸回転は固定
        np.random.uniform(0, 2 * np.pi)  # Z軸回転（360度回転可能）
    ])

    # レンダリングを実行
    filename = 'r_{0:03d}'.format(i)  # ファイル名を作成
    scene.render.filepath = os.path.join(RESULTS_PATH, 'train', filename)  # 出力パスを設定
    bpy.ops.render.render(write_still=True)  # 画像をレンダリングして保存

    # 変換行列をリスト化する関数
    def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    # フレーム情報をJSONデータに追加
    frame_data = {
        'file_path': './train/' + filename,  # ファイルのパス
        'transform_matrix': listify_matrix(cam.matrix_world)  # カメラの変換行列
    }
    out_data['frames'].append(frame_data)

# JSONファイルとして保存
with open(os.path.join(RESULTS_PATH, 'transforms_train.json'), 'w') as out_file:
    json.dump(out_data, out_file, indent=4)
