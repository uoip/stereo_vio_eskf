**Caution:** this project is unfinished, or in other words, failed, the fusion of visual odometry and imu works incorrectly.   
Parts of the project may be useful to others, so I put it on github, hope someone can help me find out what's wrong.  


**Progress:**    
- [x] Stereo visual odometry
- [x] Data loader of visual-inertial dataset [EuRoC MAV](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
- [x] Data publisher: publish time-aligned images, imu and groundtruth data online
- [ ] ESKF sensor fusion of visual odometry and imu


This is a learning project trying to implement ESKF based stereo vio algorithm in python, the ESKF visual-inertial sensor fusion part is based on project [StereoVIO](https://github.com/jim1993/StereoVIO) and paper "Loosely Coupled Stereo Inertial Odometry on Low-cost System" HaoChih LIN et al. IMAV17. There must be something wrong in my python code, but due of my little knowledge/experience of imu and filtering stuffs, I can't find it out. Hope someone can help me.  

Though this implementation is unsuccessful, because of efficient matrix manipulation of numpy, I still think Kalman Filter related algorithms are suitable to be implemented in python. 