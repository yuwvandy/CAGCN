echo "=====Amazon====="
echo "=====MF====="
python main.py --dataset='amazon' --model='MF'

echo "=====LightGCN====="
python main.py --dataset='amazon' --model='LightGCN'

echo "=====NGCF====="
python main.py --dataset='amazon' --model='NGCF'


echo "=====CAGCN-jc====="
python main.py --dataset='amazon' --model='CAGCN' --type='jc' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN-sc====="
python main.py --dataset='amazon' --model='CAGCN' --type='sc' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN-co====="
python main.py --dataset='amazon' --model='CAGCN' --type='co' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN-lhn====="
python main.py --dataset='amazon' --model='CAGCN' --neg_in_val_test=0 --type='lhn' --trend_coeff=1 --l2=1e-4 > result_CAGCN_amazon_neg0lhn_tc1_l21e-4.txt



echo "=====CAGCN*-jc====="
python main_fusion.py --dataset='amazon' --model='CAGCN' --type='jc' --trend_coeff=1.7 --l2=1e-3

echo "=====CAGCN*-jc - 256 (Compared with GTN)====="
python main_fusion.py --dataset='amazon' --model='CAGCN' --type='jc' --trend_coeff=1.7 --l2=1e-3 --embedding_dim=256


echo "=====CAGCN*-sc====="
python main_fusion.py --dataset='amazon' --model='CAGCN' --type='sc' --trend_coeff=1.7 --l2=1e-3

echo "=====CAGCN*-sc - 256 (Compared with GTN)====="
python main_fusion.py --dataset='amazon' --model='CAGCN' --type='sc' --trend_coeff=1.7 --l2=1e-3 --embedding_dim=256


echo "=====CAGCN*-lhn====="
python main_fusion.py --dataset='amazon' --model='CAGCN' --type='lhn' --trend_coeff=1.5 --l2=1e-3


echo "=====CAGCN*-lhn - 256 (Compared with GTN)====="
python main_fusion.py --dataset='amazon' --model='CAGCN' --type='lhn' --trend_coeff=1.5 --l2=1e-3 --embedding_dim=256
