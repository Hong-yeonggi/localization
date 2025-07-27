# localization
## 1. 기존 navigation 실행
### nav2_params.yaml
/home/yeonggi/score_based_localization/src/navigation2/nav2_bringup/params/nav2_params.yaml
```
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: True
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field_prob"
    max_beams: 360
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan
```
```
ros2 launch nav2_bringup bringup_launch.py use_sim_time:=True autostart:=False map:=$HOME/score_based_localization/5_floor.yaml
```    
```
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz
```
## 2. score based navigation 실행 시 
### 2.1 nav2_params.yaml 수정
/home/yeonggi/score_based_localization/src/navigation2/nav2_bringup/params/nav2_params.yaml

do_beamskip: false 

laser_model_type:"likelihood_field"

scan_topic: static_scan 

그 외 나머지는 기존과 동일
```
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 360
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: static_scan
```
### 2.2 Score prediction model 실행
/home/yeonggi/score_based_localization/lidar_score/unet/static_unet_plot_최종.py
```
python3 static_unet_plot_최종.py
```
### 2.3 navigation 실행
```
ros2 launch nav2_bringup bringup_launch.py use_sim_time:=True autostart:=False map:=$HOME/score_based_localization/5_floor.yaml
```    
```
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz
```

## 3. SCORE MODEL 데이터 가공
/home/yeonggi/score_based_localization/lidar_score/데이터_생성
### 3.1 rosbag 데이터 추출
```
python3 point.py
```
### 3.2 labeling
slam 초기 위치를 도면 상에 마우스로 클릭
```
python3 coor_regr.py
```
### 3.3 labeling 확인
```
label_plot.py
```
## 4 SCORE prediction model
/home/yeonggi/score_based_localization/lidar_score/unet

모델 학습
```
python3 u_net_최종.py
```
data test 성능 확인
```
python3 label_plot_pred.py
```

모델 성능 확인, 터틀봇 전원 킨 후 실행
```
python3 static_unet_plot_최종.py
```

## localization 성능 평가
/home/yeonggi/score_based_localization/localization

### rosbag
```
rosbag2_2025_07_20-21_56_48 -spot1
rosbag2_2025_07_21-21_47_01 -spot2
rosbag2_2025_07_21-21_56_43 -spot3
rosbag2_2025_07_21-22_05_09 -spot4

spot1: 실험실 복도
spot2: 엘리베이터 복도
spot3: 연구실 복도
spot4: 화물 엘리베이터 및 쓰레기통 복도
```
