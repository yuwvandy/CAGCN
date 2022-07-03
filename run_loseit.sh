echo "=====Loseit====="
echo "=====MF====="
python main.py --dataset='loseit' --model='MF'

echo "=====LightGCN====="
python main.py --dataset='loseit' --model='LightGCN' --n_hops=1

echo "=====NGCF====="
python main.py --dataset='loseit' --model='NGCF'


echo "=====CAGCN-jc====="
python main.py --dataset='loseit' --model='CAGCN' --type='jc' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN-lhn====="
python main.py --dataset='loseit' --model='CAGCN' --type='lhn' --trend_coeff=1 --l2=1e-4 --n_hops=1

echo "=====CAGCN-sc====="
python main.py --dataset='loseit' --model='CAGCN' --type='sc' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN-co====="
python main.py --dataset='loseit' --model='CAGCN' --type='co' --trend_coeff=1 --l2=1e-4


echo "=====CAGCN*-jc====="
python main_fusion.py --dataset='loseit' --model='CAGCN' --type='jc' --trend_coeff=1 --l2=1e-3 --n_hops=2


echo "=====CAGCN*-sc====="
python main_fusion.py --dataset='loseit' --model='CAGCN' --type='sc' --trend_coeff=1 --l2=1e-3 --n_hops=2


echo "=====CAGCN*-lhn====="
python main_fusion.py --dataset='loseit' --model='CAGCN' --type='lhn' --trend_coeff=0.5 --l2=1e-3 --n_hops=2
