tranche=8
mode='train'
env='hard_constrain'
agent='linear'
episodes=200
device=-1
stocklist="./data/stocklist/sample.txt"
parallel=2

job=0
cat $stocklist| while read stock name; do
    let job+=1
    echo "strat to train $stock"
    for i in $(seq 0 $[ $tranche - 1 ]); do
        cmd="CUDA_VISIBLE_DEVICES=$device nohup python -u vwap.py --mode $mode --env $env --agent $agent
        --stock $stock --side sell --episodes $episodes --level 1 --tranche_id $i --overwrite
        2>&1 >./logs/$mode/$stock-hard-$agent-$i-$tranche.log"
        eval $cmd&
        sleep 10
    done
    if [ $[$job % $parallel] == 0 ]; then
        wait
        echo "wait"
    fi
done