# cmput428_tracker

Camshift + KF + Edge histgram detection

We acknowledge the orginal idea from paper 

I. Iraei and K. Faez, "Object tracking with occlusion handling using mean shift, Kalman filter and Edge Histogram," 2015 2nd International Conference on 
Pattern Recognition and Image Analysis (IPRIA), 2015, pp. 1-6, doi: 10.1109/PRIA.2015.7161637.


Edge detection : Sobel filter

We assume occlusion wont last longer than  1 second


## Run Instructions
1. clone the code on your machine
2. open terminal to run our tracker in your webcan
```bash
$ python3 main.py
```
3. Then select your interest region and press "space"; If you want to exit the program, simply press "q"

For historical data
```bash
$ python3 videoReadTracker.py
```bash
