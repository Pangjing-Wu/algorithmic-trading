tranches=8
model='Linear'
agent='QLearning'
env='Historical'
reward='sparse'
eps=0.5
streps='05'
episode=10000
device=0
quote_length=1
stocklist="./data/stocklist/sample.txt"

cat $stocklist| while read stock name; do
    for i in $(seq 1 $tranches); do
    echo "strat to train $stock tranche $i/$tranches at $(date)"
        cmd="CUDA_VISIBLE_DEVICES=$device python -u ./scripts/m3t/micro/train.py 
            --stock $stock --episode $episode --agent $agent --eps $eps --i_tranche $i
            --model $model --env $env --reward $reward --quote_length $quote_length --cuda
            2>&1 >./results/logs/vwap/m3t/micro/$stock-$model-eps$streps-$reward-len$quote_length-$i-$tranches.log"
        eval $cmd&
        sleep 10
    done
    wait
    echo "training of $stock has done at $(date)"
done