#!/bin/sh
# sudo mount -t drvfs R: ./input
sudo mount -t cifs -o username=osoriojr@gmail.com //192.168.1.139/replay input/
python monitor.py
