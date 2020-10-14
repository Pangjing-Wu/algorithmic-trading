tranche=8
mode='train'
env='hard_constrain'
agent='linear'
episodes=200
device=0
stocklist="./data/stocklist/SSE50.txt"

cat $stocklist| while read stock
do
    for i in $(seq 0 $(expr $tranche - 1))
    do
        cmd="CUDA_VISIBLE_DEVICE=$device nohup python -u vwap.py --mode $mode --env $env --agent $agent --stock $stock
        --side sell --episodes $episodes --level 1 --tranche_id $i 2>&1 >./logs/$mode/$stock-hard-$agent-$i-$tranche.log"
        eval $cmd&
    done
    wait
done