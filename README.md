# raspberry-cats
Monitoring way in-out of cats through cat-flaps, using AI for cat recognition


This repository is linked to the project described here : https://github.com/users/NicholasJallan/projects/1
In short : 
- Use image processing (based on OpenCV) and a trained classifiers (TBD)…
- recognize cats going through cat-flap of the house, and produce logs out of it, plus a consultation API 
Hardware used : 
- Raspberry Pi 4 model B 4GB
- raspberry camera V2 NoIR 
Maybe used : 
- RFID sensors (to be attached on cats necklaces, to improve the cat identification)
- magnetics switches (to identify cats / way they go if results not good enough with the classifier)


Spotted problem :
- basic RFID detector has a very short range detection (5cm). Need to find a better reading antenna to reach 15/20 cm, or use a differrent hardware

Pull requests are welcome, as any full-text written ideas.

To set up open CV and you raspberry, just follow instructions : 
https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/

The code of this repository is bootstrapped from the link above.
