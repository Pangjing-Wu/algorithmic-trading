tranche=8
mode='train'
env='recurrent_hard_constrain'
agent='lstm'
episodes=200
device=0
stocklist="./data/stocklist/sample.txt"

cat $stocklist| while read stock name; do
    echo "strat to train $stock at $(date)"
    for i in $(seq 0 $[ $tranche - 1 ]); do
        $device = $[$i%2]
        cmd="CUDA_VISIBLE_DEVICES=$device nohup python -u vwap.py --mode $mode --env $env --agent $agent
        --stock $stock --side sell --episodes $episodes --level 1 --tranche_id $i --overwrite
        2>&1 >./logs/$mode/$stock-hard-$agent-$i-$tranche.log"
        eval $cmd&
        sleep 10
    done
    wait
    echo "training of $stock has done at $(date)"
done