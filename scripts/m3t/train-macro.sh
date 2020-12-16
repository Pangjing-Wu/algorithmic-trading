stocklist="./config/stocklist/sample-8.txt"
epoch=200
models=("Linear" "MLP" "LSTM")

cat $stocklist | while read stock name; do
    echo "$name($stock) start at $(date)"
    for model in ${models[@]}; do
        echo "     [START] train $model at $(date)"
        cmd="python -u ./scripts/m3t/macro/train.py 
            --stock $stock --epoch 200 --model $model --checkpoint 0 --cuda
            2>&1 >./results/logs/vwap/m3t/macro/$stock-$model.log"
        eval $cmd
        echo "     [DONE] $(date)"
    done
    echo "$name($stock) has done at $(date)"
done