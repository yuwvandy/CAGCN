echo "=====Yelp====="
echo "=====MF====="
python main.py --dataset='yelp2018' --model='MF' --neg_in_val_test=0

echo "=====LightGCN====="
python main.py --dataset='yelp2018' --model='LightGCN' --neg_in_val_test=0

echo "=====NGCF====="
python main.py --dataset='yelp2018' --model='NGCF' --neg_in_val_test=0 --l2=1e-3

echo "=====CAGCN-jc====="
python main.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='jc' --trend_coeff=1.2 --l2=1e-4

echo "=====CAGCN-lhn====="
python main.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='lhn' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN-sc====="
python main.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='sc' --trend_coeff=1.2 --l2=1e-4

echo "=====CAGCN-co====="
python main.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='co' --trend_coeff=1 --l2=1e-4


echo "=====CAGCN*-jc====="
python main_fusion.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='jc' --trend_coeff=1.7 --l2=1e-3

echo "=====CAGCN*-jc - 256 (Compared with GTN)====="
python main_fusion.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='jc' --trend_coeff=1.7 --l2=1e-3 --embedding_dim=256

echo "=====CAGCN*-sc====="
python main_fusion.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='sc' --trend_coeff=1.7 --l2=1e-3

echo "=====CAGCN*-sc - 256 (Compared with GTN)====="
python main_fusion.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='sc' --trend_coeff=1.7 --l2=1e-3 --embedding_dim=256

echo "=====CAGCN*-lhn====="
python main_fusion.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='lhn' --trend_coeff=1 --l2=1e-3

echo "=====CAGCN*-lhn - 256 (Compared with GTN)====="
python main_fusion.py --dataset='yelp2018' --model='CAGCN' --neg_in_val_test=0 --type='lhn' --trend_coeff=1 --l2=1e-3 --embedding_dim=256
