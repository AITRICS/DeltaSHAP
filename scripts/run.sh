run_main_table() {
    cv=0
    num_ensemble=1
    seed_list="0 1 2 3 4"

    data_list="mimic3o physionet19"
    modeltype_list="LSTM"
    explainer_list="deltashap lime gradientshap ig deeplift fo afo fit winit"

    for modeltype in ${modeltype_list}; do
        for data in ${data_list}; do
            ckptpath=ckpt/${modeltype}/${data}/model.best.pth.tar
            datapath=data/${data}
            plotpath=plots/${data}
            for seed in ${seed_list}; do
                wait_n
                i=$((i + 1))

                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
                    --eval \
                    --cv ${cv} \
                    --data ${data} \
                    --testbs 200 \
                    --explainerseed ${seed} \
                    --datapath ${datapath} \
                    --modeltype ${modeltype} \
                    --numensemble ${num_ensemble} \
                    --ckptpath ${ckptpath} \
                    --explainer ${explainer} \
                    --outpath output/neurips/${modeltype}_${data}_${explainer}_${seed} \
                    --resultfile ${modeltype}_${data}_${explainer}_${seed} \
                    --logpath logs/${modeltype}_${data}_${explainer}_${seed} \
                    --logfile ${modeltype}_${data}_${explainer}_${seed} \
                    --plotpath plots/${modeltype}_${data}_${explainer}_${seed} \
                    --vis False \
                    2>&1
            done
        done
    done
}

ablation_study() {
    cv=0
    num_ensemble=1

    data=mimic3o
    modeltype=LSTM
    explainer=deltashap
    seed_list="0 1 2 3 4"

    deltashap_n_samples=25
    deltashap_normalize=True
    deltashap_baseline=carryforward

    ckptpath=ckpt/${modeltype}/${data}/model.best.pth.tar
    datapath=data/${data}
    plotpath=plots/${data}

    # standard
    for seed in ${seed_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
            --eval \
            --cv ${cv} \
            --data ${data} \
            --testbs 250 \
            --explainerseed ${seed} \
            --datapath ${datapath} \
            --modeltype ${modeltype} \
            --numensemble ${num_ensemble} \
            --ckptpath ${ckptpath} \
            --explainer ${explainer} \
            --deltashap_n_samples ${deltashap_n_samples} \
            --deltashap_normalize ${deltashap_normalize} \
            --deltashap_baseline ${deltashap_baseline} \
            --outpath output/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
            --resultfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
            --logpath logs/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
            --logfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
            --plotpath plots/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
            --vis False \
            2>&1
        wait_n
        i=$((i + 1))
    done

    deltashap_n_samples_list="1 10 100"
    deltashap_normalize=True
    deltashap_baseline=carryforward
    for deltashap_n_samples in ${deltashap_n_samples_list}; do
        for seed in ${seed_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
                --eval \
                --cv ${cv} \
                --data ${data} \
                --testbs 250 \
                --explainerseed ${seed} \
                --datapath ${datapath} \
                --modeltype ${modeltype} \
                --numensemble ${num_ensemble} \
                --ckptpath ${ckptpath} \
                --explainer ${explainer} \
                --deltashap_n_samples ${deltashap_n_samples} \
                --deltashap_normalize ${deltashap_normalize} \
                --deltashap_baseline ${deltashap_baseline} \
                --outpath output/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --resultfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logpath logs/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --plotpath plots/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --vis False \
                2>&1
            wait_n
            i=$((i + 1))
        done
    done

    deltashap_n_samples=25
    deltashap_normalize=True
    deltashap_baseline_list="zero"
    for deltashap_baseline in ${deltashap_baseline_list}; do
        for seed in ${seed_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
                --eval \
                --cv ${cv} \
                --data ${data} \
                --testbs 250 \
                --explainerseed ${seed} \
                --datapath ${datapath} \
                --modeltype ${modeltype} \
                --numensemble ${num_ensemble} \
                --ckptpath ${ckptpath} \
                --explainer ${explainer} \
                --deltashap_n_samples ${deltashap_n_samples} \
                --deltashap_normalize ${deltashap_normalize} \
                --deltashap_baseline ${deltashap_baseline} \
                --outpath output/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --resultfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logpath logs/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --plotpath plots/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --vis False \
                2>&1
            wait_n
            i=$((i + 1))
        done
    done

    deltashap_n_samples=25
    deltashap_normalize_list="False"
    deltashap_baseline=carryforward
    for deltashap_normalize in ${deltashap_normalize_list}; do
        for seed in ${seed_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
                --eval \
                --cv ${cv} \
                --data ${data} \
                --testbs 250 \
                --explainerseed ${seed} \
                --datapath ${datapath} \
                --modeltype ${modeltype} \
                --numensemble ${num_ensemble} \
                --ckptpath ${ckptpath} \
                --explainer ${explainer} \
                --deltashap_n_samples ${deltashap_n_samples} \
                --deltashap_normalize ${deltashap_normalize} \
                --deltashap_baseline ${deltashap_baseline} \
                --outpath output/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --resultfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logpath logs/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --plotpath plots/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --vis False \
                2>&1
            wait_n
            i=$((i + 1))
        done
    done

    deltashap_n_samples_list="100"
    deltashap_normalize=True
    deltashap_baseline=carryforward
    for deltashap_n_samples in ${deltashap_n_samples_list}; do
        for seed in ${seed_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
                --eval \
                --cv ${cv} \
                --data ${data} \
                --testbs 250 \
                --explainerseed ${seed} \
                --datapath ${datapath} \
                --modeltype ${modeltype} \
                --numensemble ${num_ensemble} \
                --ckptpath ${ckptpath} \
                --explainer ${explainer} \
                --deltashap_n_samples ${deltashap_n_samples} \
                --deltashap_normalize ${deltashap_normalize} \
                --deltashap_baseline ${deltashap_baseline} \
                --outpath output/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --resultfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logpath logs/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --logfile ${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --plotpath plots/${modeltype}_${data}_${explainer}_${seed}_${deltashap_n_samples}_${deltashap_normalize}_${deltashap_baseline} \
                --vis False \
                2>&1
            wait_n
            i=$((i + 1))
        done
    done
}

vis() {
    cv=0
    data_list="mimic3o"
    modeltype_list="LSTM"
    explainer_list="deltashap"
    seed_list="0"
    vis_dir="plots/ieee_access"

    for explainer in ${explainer_list}; do
        for modeltype in ${modeltype_list}; do
            for data in ${data_list}; do
                ckptpath=ckpt/${modeltype}/${data}/model.best.pth.tar
                datapath=data/${data}
                plotpath=plots/${data}
                for seed in ${seed_list}; do
                    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m deltashap.run \
                        --eval \
                        --cv ${cv} \
                        --data ${data} \
                        --testbs 1000 \
                        --explainerseed ${seed} \
                        --datapath ${datapath} \
                        --modeltype ${modeltype} \
                        --ckptpath ${ckptpath} \
                        --explainer ${explainer} \
                        --outpath output/${modeltype}_${data}_${explainer}_${seed} \
                        --resultfile ${modeltype}_${data}_${explainer}_${seed} \
                        --logpath logs/${modeltype}_${data}_${explainer}_${seed} \
                        --logfile ${modeltype}_${data}_${explainer}_${seed} \
                        --plotpath plots/${modeltype}_${data}_${explainer}_${seed} \
                        --top 25 \
                        --vis True \
                        --num_vis 200 \
                        --vis_dir ${vis_dir} \
                        2>&1
                    wait_n
                    i=$((i + 1))
                done
            done
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
num_max_jobs=5

run_main_table
# ablation_study
# vis
