#!/bin/bash
python /home/wuhang/MLP/split.py
start=`date +%s`
scp /home/wuhang/MLP/inputdata11.npy Data1:/home/wuhang/MLP
scp /home/wuhang/MLP/outdata11.npy Data1:/home/wuhang/MLP
ssh -tt Data1 "/home/wuhang/anaconda2/bin/python /home/wuhang/MLP/train2.py;exit" &
python /home/wuhang/MLP/train.py &
wait
end=`date +%s`
echo "TIME:`expr $end - $start`"
scp Data1:/home/wuhang/MLP/model02.h5 /home/wuhang/MLP
python /home/wuhang/MLP/getdata.py
