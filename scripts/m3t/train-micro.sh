tranches=8
models=('Linear' 'HybridLSTM')
agent='QLearning'
rewards=('sparse' 'dense')
eps=1.0
streps='10'
episode=10000
quote_length=5
stocklist="./data/stocklist/sample-8.txt"

cat $stocklist| while read stock name; do
    for i in $(seq 1 $tranches); do
        echo "strat to train $stock tranche $i/$tranches at $(date)"
        for model in ${models[@]}; do
            if [ $model == 'Linear' ]; then device=0; else device=1; fi
            for reward in ${rewards[@]}; do
                cmd="CUDA_VISIBLE_DEVICES=$device python -u ./scripts/m3t/micro/train.py 
                    --stock $stock --episode $episode --agent $agent --eps $eps --i_tranche $i
                    --model $model --reward $reward --quote_length $quote_length --cuda
                    2>&1 >./results/logs/vwap/m3t/micro/$stock-$model-eps$streps-$reward-len$quote_length-$i-$tranches.log"
                eval $cmd&
                sleep 2
            done
        done
    done
    wait
done