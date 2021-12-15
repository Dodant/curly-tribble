echo 'Start Trial -- 1 -----------------------------------------------'
python kd_train_cifar.py -s 20 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 20 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 20 -t 2 -alpha 0.75 -epochs 150 -batch 128

python kd_train_cifar.py -s 32 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 32 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 32 -t 2 -alpha 0.75 -epochs 150 -batch 128

python kd_train_cifar.py -s 56 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 56 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 56 -t 2 -alpha 0.75 -epochs 150 -batch 128


echo 'Start Trial -- 2 -----------------------------------------------'
python kd_train_cifar.py -s 20 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 20 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 20 -t 2 -alpha 0.75 -epochs 150 -batch 128

python kd_train_cifar.py -s 32 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 32 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 32 -t 2 -alpha 0.75 -epochs 150 -batch 128

python kd_train_cifar.py -s 56 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 56 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 56 -t 2 -alpha 0.75 -epochs 150 -batch 128


echo 'Start Trial -- 3 -----------------------------------------------'
python kd_train_cifar.py -s 20 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 20 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 20 -t 2 -alpha 0.75 -epochs 150 -batch 128

python kd_train_cifar.py -s 32 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 32 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 32 -t 2 -alpha 0.75 -epochs 150 -batch 128

python kd_train_cifar.py -s 56 -t 2 -alpha 0.25 -epochs 150 -batch 128
python kd_train_cifar.py -s 56 -t 2 -alpha 0.5 -epochs 150 -batch 128
python kd_train_cifar.py -s 56 -t 2 -alpha 0.75 -epochs 150 -batch 128

echo 'Start Test -----------------------------------------------------'
python kd_test_cifar.py -s 20 -batch 128
python kd_test_cifar.py -s 32 -batch 128
python kd_test_cifar.py -s 56 -batch 128

echo 'Mischief Managed'