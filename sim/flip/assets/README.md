# Assets

The arm meshes, `LICENSE` and `../so_arm100.xml` are the SO-ARM100 model from MuJoCo Menagerie (`trs_so_arm100`), the same as in `sim/reach`.

The wrist camera is the official SO-101 wrist camera: `wrist_camera_mount_so101_v1.stl` and `wrist_camera_so101_v1.stl`, taken from [adityakamath/so_arm_ros2](https://github.com/adityakamath/so_arm_ros2) at commit [`53ba84b`](https://github.com/adityakamath/so_arm_ros2/tree/53ba84bfe7008b84551b33aff84fd4d5297f86d8/so_arm_description/meshes/so101) (Apache-2.0). The camera body changed upstream afterwards, so re-download from that commit, not `main`.

The camera body is an ASCII STL, and MuJoCo only loads binary STL. The scene therefore loads `wrist_camera_so101_v1_bin.stl`, built from the ASCII original with:

    python -m sim.flip.stl_ascii_to_binary sim/flip/assets/wrist_camera_so101_v1.stl sim/flip/assets/wrist_camera_so101_v1_bin.stl
