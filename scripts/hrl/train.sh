models=('HybridLSTM' 'HybridAttenBiLSTM')
agent='HierarchicalQ'
rewards=('dense' 'sparse')
eps=1.0
streps='10'
episode=10000
quote_length=5
stocklist="./config/stocklist/sample-8.txt"

cat $stocklist| while read stock name; do
    echo "strat to train $stock at $(date)"
    for model in ${models[@]}; do
        if [ $model == 'HybridLSTM' ]; then device=0; else device=1; fi
        for reward in ${rewards[@]}; do
            cmd="CUDA_VISIBLE_DEVICES=$device python -u ./scripts/hrl/train.py 
                --stock $stock --episode $episode --agent $agent --eps $eps
                --model $model --reward $reward --quote_length $quote_length --cuda
                2>&1 >./results/logs/vwap/hrl/$stock-$model-eps$streps-$reward-len$quote_length.log"
            echo $cmd&
            sleep 2
        done
    done
done
echo "training script has done at $(date)"