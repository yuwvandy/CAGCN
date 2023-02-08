echo "=====ml-1m====="
echo "=====MF====="
python main.py --dataset='ml-1m' --model='MF'

echo "=====LightGCN====="
python main.py --dataset='ml-1m' --model='LightGCN'

echo "=====NGCF====="
python main.py --dataset='ml-1m' --model='NGCF'

echo "=====CAGCN-jc====="
python main.py --dataset='ml-1m' --model='CAGCN' --type='jc' --trend_coeff=2 --l2=1e-4

echo "=====CAGCN-lhn====="
python main.py --dataset='ml-1m' --model='CAGCN'  --type='lhn' --trend_coeff=2 --l2=1e-4

echo "=====CAGCN-sc====="
python main.py --dataset='ml-1m' --model='CAGCN'  --type='sc' --trend_coeff=2 --l2=1e-4

echo "=====CAGCN-co====="
python main.py --dataset='ml-1m' --model='CAGCN'  --type='co' --trend_coeff=1 --l2=1e-4

echo "=====CAGCN*-jc====="
python main_fusion.py --dataset='ml-1m' --model='CAGCN'  --type='jc' --trend_coeff=1 --l2=1e-3

echo "=====CAGCN*-sc====="
python main_fusion.py --dataset='ml-1m' --model='CAGCN'  --type='sc' --trend_coeff=1 --l2=1e-3

echo "=====CAGCN*-lhn====="
python main_fusion.py --dataset='ml-1m' --model='CAGCN'  --type='lhn' --trend_coeff=1 --l2=1e-3
