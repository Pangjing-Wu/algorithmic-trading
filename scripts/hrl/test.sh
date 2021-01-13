n_tranche=8
macros=('Baseline' 'LSTM')
micros=('HybridLSTM' 'HybridAttenBiLSTM')
agent='HierarchicalQ'
rewards=('dense' 'sparse')
eps=1.0
goal=200000
streps='10'
epoch=-1
episode=10000
quote_length=5
stocklist="./config/stocklist/sample-8.txt"

cat $stocklist| while read stock name; do
    cat /dev/null > ./results/outputs/vwap/hrl/$stock.txt
done

for macro in ${macros[@]}; do
    for micro in ${micros[@]}; do
        for reward in ${rewards[@]}; do
                echo "start test macro = $macro micro = $micro reward = $reward at $(date)"
                cat $stocklist| while read stock name; do 
                    cmd="python -u ./scripts/hrl/test.py
                        --goal $goal --stock $stock
                        --macro $macro --micro $micro --agent $agent 
                        --macro_epoch $epoch --micro_episode $episode
                        --quote_length $quote_length --reward $reward
                        >> ./results/outputs/vwap/hrl/$stock.txt"
                    eval $cmd&
                    sleep 5
                    if [ $stock == 600104 ]; then wait; fi
                done
            done
        done
    done
done
echo "test script has done at $(date)"