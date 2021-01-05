n_tranche=8
models=('HybridLSTM' 'HybridAttenBiLSTM')
agent='HierarchicalQ'
rewards=('dense' 'sparse')
eps=1.0
streps='10'
episode=10000
quote_length=5
stocklist="./config/stocklist/sample-8.txt"

cat $stocklist| while read stock name; do
    cat /dev/null > ./results/outputs/vwap/hrl/$stock.txt
done

for model in ${models[@]}; do
    for reward in ${rewards[@]}; do
        for i in $(seq 1 $n_tranche); do
            cat $stocklist| while read stock name; do 
                cmd="python -u ./scripts/hrl/test.py --stock $stock 
                    --agent $agent --model $model --reward $reward
                    --i_tranche $i --quote_length $quote_length
                    --model_episode $episode >> ./results/outputs/vwap/hrl/$stock.txt"
                eval $cmd&
                sleep 5
                if [ $stock == 600104 ]; then wait; fi
            done
        done
    done
done
echo "test script has done at $(date)"