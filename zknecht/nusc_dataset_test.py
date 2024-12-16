from nuscenes import NuScenes

# a4d6e99f30ed4c5589bf6f87503fa064
dataroot = "/media/sun/z/nuscenes/nuscenes"
token = '05bd120bea1d48cda7b2b096734d8e1b'     # 8fa07b104b904d22970d2f8946a0b771    80d56e801c7e465995bdb116b3e678aa
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
sample = nusc.get('sample', token)
lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
# {'token': '5879f3d0a1ac40498e4bd7649ef397d6', 'sample_token': '80d56e801c7e465995bdb116b3e678aa',
# 'ego_pose_token': '5879f3d0a1ac40498e4bd7649ef397d6', 'calibrated_sensor_token': 'f31e1b9f19e3496899bb7e0172000252',
# 'timestamp': 1533279682150532, 'fileformat': 'pcd', 'is_key_frame': True, 'height': 0, 'width': 0,
# 'filename': 'samples/LIDAR_TOP/n015-2018-08-03-15-00-36+0800__LIDAR_TOP__1533279682150532.pcd.bin',
# 'prev': '48b673ee2b204e20aa70c0c02d799adf', 'next': '89afd98494964bada17dbcabbe948212', 'sensor_modality': 'lidar',
# 'channel': 'LIDAR_TOP'}
lidar_calibrated_sensor = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
# {'token': 'f31e1b9f19e3496899bb7e0172000252', 'sensor_token': 'dc8b396651c05aedbb9cdaae573bb567',
# 'translation': [0.943713, 0.0, 1.84023], 'camera_intrinsic': []
# 'rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817], }

lidar_ego_pose = nusc.get('ego_pose', lidar['ego_pose_token'])
# {'token': '5879f3d0a1ac40498e4bd7649ef397d6', 'timestamp': 1533279682150532,
# 'rotation': [-0.7826393592967994, 0.00867598472714821, -0.010566539708221056, 0.6223252435881937],
# 'translation': [132.0368788137383, 997.2431802967292, 0.0]}
# ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
img_CAM_FRONT = nusc.get('sample_data', sample['data']['CAM_FRONT'])
# {'token': '9078da9a2fe34bc4972cb8c69a9c4ad4', 'sample_token': '80d56e801c7e465995bdb116b3e678aa',
# 'ego_pose_token': '9078da9a2fe34bc4972cb8c69a9c4ad4', 'calibrated_sensor_token': '8d26763fbe404b958de94d673a1643ce',
# 'timestamp': 1533279682112460, 'fileformat': 'jpg', 'is_key_frame': True, 'height': 900, 'width': 1600,
# 'filename': 'samples/CAM_FRONT/n015-2018-08-03-15-00-36+0800__CAM_FRONT__1533279682112460.jpg',
# 'prev': '515101c14fd248a58505b8b3c94ca13b', 'next': 'd309f105eb104a84b86ad277a1286180', 'sensor_modality': 'camera',
# 'channel': 'CAM_FRONT'}
img_CAM_BACK = nusc.get('sample_data', sample['data']['CAM_BACK'])
img_CAM_FRONT_LEFT = nusc.get('sample_data', sample['data']['CAM_FRONT_LEFT'])
img_CAM_FRONT_RIGHT = nusc.get('sample_data', sample['data']['CAM_FRONT_RIGHT'])
img_CAM_BACK_LEFT =nusc.get('sample_data', sample['data']['CAM_BACK_LEFT'])
img_CAM_BACK_RIGHT =nusc.get('sample_data', sample['data']['CAM_BACK_RIGHT'])

calibrated_sensor_CAM_FRONT = nusc.get('calibrated_sensor', img_CAM_FRONT['calibrated_sensor_token'])
# {'token': '8d26763fbe404b958de94d673a1643ce', 'sensor_token': '725903f5b62f56118f4094b46a4470d8',
# 'translation': [1.70079118954, 0.0159456324149, 1.51095763913],
# 'rotation': [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755],
# 'camera_intrinsic': [[1266.417203046554, 0.0, 816.2670197447984], [0.0, 1266.417203046554, 491.50706579294757], [0.0, 0.0, 1.0]]}

ego_pose_CAM_FRONT = nusc.get('ego_pose', img_CAM_FRONT['ego_pose_token'])
# {'token': '9078da9a2fe34bc4972cb8c69a9c4ad4', 'timestamp': 1533279682112460,
# 'rotation': [-0.7826162378208529, 0.008463341932761203, -0.01055200652286372, 0.6223574947735464],
# 'translation': [131.98652774136764, 997.4686079778229, 0.0]}
ego_pose_CAM_FRONT_LEFT = nusc.get('ego_pose', img_CAM_FRONT_LEFT['ego_pose_token'])
ego_pose_CAM_FRONT_RIGHT = nusc.get('ego_pose', img_CAM_FRONT_RIGHT['ego_pose_token'])
ego_pose_CAM_BACK = nusc.get('ego_pose', img_CAM_BACK['ego_pose_token'])
ego_pose_CAM_BACK_LEFT = nusc.get('ego_pose', img_CAM_BACK_LEFT['ego_pose_token'])
ego_pose_CAM_BACK_RIGHT = nusc.get('ego_pose', img_CAM_BACK_RIGHT['ego_pose_token'])


sensor = nusc.get('sensor', calibrated_sensor_CAM_FRONT['sensor_token'])
# {'token': 'ec4b5d41840a509984f7ec36419d4c09', 'channel': 'CAM_FRONT_LEFT', 'modality': 'camera'}
# print(lidar)
# print(lidar_calibrated_sensor)
# print('lidar_ego_pose', lidar_ego_pose)
# print('cam_calibrated_sensor', calibrated_sensor_CAM_FRONT)
print('lidar_ego_pose', img_CAM_BACK)
print('ego_pose_CAM_FRONT', img_CAM_BACK_LEFT)
print('ego_pose_CAM_FRONT_LEFT', img_CAM_BACK_RIGHT)
# print('ego_pose_CAM_FRONT_RIGHT', ego_pose_CAM_FRONT_RIGHT)
# print('ego_pose_CAM_BACK', ego_pose_CAM_BACK)
# print('ego_pose_CAM_BACK_LEFT',ego_pose_CAM_BACK_LEFT)
# print('ego_pose_CAM_BACK_RIGHT', ego_pose_CAM_BACK_RIGHT)