train_gen() {
    modeltype=LSTM
    data_list="mimic3o physionet19"
    explainer_list="fit winit"

    for data in ${data_list}; do
        datapath=data/${data}
        ckptpath=ckpt/LSTM/${data}/model.best.pth.tar
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
            --traingen \
            --data ${data} \
            --datapath ${datapath} \
            --modeltype ${modeltype} \
            --ckptpath ${ckptpath} \
            --cv 0 \
            --explainer ${explainer} \
            --skipexplain \
            2>&1
        wait_n
        i=$((i + 1))
        done
    done
}

wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=20

train_gen