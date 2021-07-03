stocklist="./config/stocklist/sample-8.txt"
epoch=2000
models=("Baseline" "Linear" "MLP" "LSTM")

cat $stocklist | while read stock name; do
    echo "$name($stock) start at $(date)"
    for model in ${models[@]}; do
        echo "     [START] train $model at $(date)"
        cmd="python ./scripts/m2t/macro/train.py 
            --stock $stock --epoch $epoch --model $model --checkpoint 0
            2>&1 >./results/logs/vwap/m2t/macro/$stock-$model.log"
        eval $cmd
        echo "     [DONE] $(date)"
    done
    echo "$name($stock) has done at $(date)"
done