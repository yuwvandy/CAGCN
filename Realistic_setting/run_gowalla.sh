echo "=====Gowalla====="
echo "=====MF====="
python main.py --dataset='gowalla' --model='MF' --model_name='MF' --device='cuda:0'

echo "=====LightGCN====="
python main.py --dataset='gowalla' --model='LightGCN' --model_name='LightGCN' --device='cuda:0'

echo "=====NGCF====="
python main.py --dataset='gowalla' --model='NGCF' --model_name='NGCF' --device='cuda:0'

echo "=====CAGCN-jc====="
python main.py --dataset='gowalla' --model='CAGCN' --type='jc' --model_name='CAGCN-jc' --device='cuda:0' --trend_coeff=1 --l2=2e-4

echo "=====CAGCN-sc====="
python main.py --dataset='gowalla' --model='CAGCN' --type='sc' --model_name='CAGCN-sc' --device='cuda:0' --trend_coeff=1 --l2=2e-4

echo "=====CAGCN-lhn====="
python main.py --dataset='gowalla' --model='CAGCN' --type='lhn' --model_name='CAGCN-lhn' --device='cuda:0' --trend_coeff=1 --l2=2e-4

echo "=====CAGCN-co====="
python main.py --dataset='gowalla' --model='CAGCN' --type='cn' --model_name='CAGCN-cn' --device='cuda:0' --trend_coeff=1

echo "=====CAGCN-jc_fusion====="
python main_fusion.py --dataset='gowalla' --model='CAGCN' --type='jc' --model_name='CAGCN-jc*' --device='cuda:0' --trend_coeff=1 --l2=1e-3

echo "=====CAGCN-sc_fusion====="
python main_fusion.py --dataset='gowalla' --model='CAGCN' --type='sc' --model_name='CAGCN-sc*' --device='cuda:0' --trend_coeff=1 --l2=1e-3

echo "=====CAGCN-lhn_fusion====="
python main_fusion.py --dataset='gowalla' --model='CAGCN' --type='lhn' --model_name='CAGCN-lhn*' --device='cuda:0' --trend_coeff=1 --l2=1e-3
