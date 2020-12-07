tranches=8
model='HybridLSTM'
agent='HierarchicalQ'
env='Recurrent'
reward='dense'
eps=1.0
streps='10'
episode=10000
device=1
quote_length=5
stocklist="./data/stocklist/sample-8.txt"

cat $stocklist| while read stock name; do
    echo "strat to train $stock at $(date)"
    cmd="CUDA_VISIBLE_DEVICES=$device python -u ./scripts/hrl/train.py 
        --stock $stock --episode $episode --agent $agent --eps $eps --model $model 
        --env $env --reward $reward --quote_length $quote_length --cuda
        2>&1 >./results/logs/vwap/hrl/$stock-eps$streps-$reward-len$quote_length.log"
    eval $cmd&
done
echo "training has done at $(date)"
