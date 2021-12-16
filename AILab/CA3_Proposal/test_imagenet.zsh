echo 'Start'
python kd_train_imagenet.py -s 18 -t 2 -alpha 0.5 -epochs 25 -batch 256
python kd_train_imagenet.py -s 34 -t 2 -alpha 0.5 -epochs 25 -batch 256
python kd_train_imagenet.py -s 50 -t 2 -alpha 0.5 -epochs 25 -batch 128
echo 'Mischief Managed'