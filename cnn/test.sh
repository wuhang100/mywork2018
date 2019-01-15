#!/bin/bash
python /home/wuhang/cnn/split.py
start=`date +%s`
scp /home/wuhang/cnn/inputdata11.npy Data1:/home/wuhang/cnn
scp /home/wuhang/cnn/outdata11.npy Data1:/home/wuhang/cnn
ssh -tt Data1 "/home/wuhang/anaconda2/bin/python /home/wuhang/cnn/train2.py;exit" &
python /home/wuhang/cnn/train.py &
wait
end=`date +%s`
echo "TIME:`expr $end - $start`"
scp Data1:/home/wuhang/cnn/model02.h5 /home/wuhang/cnn
python /home/wuhang/cnn/getdata.py
