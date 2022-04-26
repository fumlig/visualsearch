#!/usr/bin/env bash

num_envs=64
num_timesteps=25000000
ckpt_interval=1000000


for seed in 1 2 3
do
    for agent in "lstm" "map"
    do
        # search space and reward
        for shape in "[10,10]" "[15,15]" "[20,20]"
        do
            # r
            name="shape-$agent-gaussian-r0$shape-$seed"

            echo "$(date +%T): train $(tput bold)$name$(tput sgr0)"

            python3 train.py "Gaussian-v0" ppo $agent \
                --name=$name \
                --seed=$seed \
                --num-timesteps=$num_timesteps \
                --num-envs=$num_envs \
                --ckpt-interval=$ckpt_interval \
                --env-kwargs="{shape: $shape}" \
                --alg-kwargs="params/our.yaml" \
                --agent-kwargs="{}"

            echo "$(date +%T): test $(tput bold)$name$(tput sgr0)"

            python3 test.py "Gaussian-v0" \
                --name=$name \
                --hidden \
                --model="models/$name.pt" \
                --env-kwargs="{shape: $shape}"

            # r'
            name="shape-$agent-gaussian-r1$shape-$seed"

            echo "$(date +%T): train $(tput bold)$name$(tput sgr0)"

            python3 train.py "Gaussian-v0" ppo $agent \
                --name=$name \
                --seed=$seed \
                --num-timesteps=$num_timesteps \
                --num-envs=$num_envs \
                --ckpt-interval=$ckpt_interval \
                --env-kwargs="{shape: $shape, reward_explore: true}" \
                --alg-kwargs="params/our.yaml" \
                --agent-kwargs="{}"

            echo "$(date +%T): test $(tput bold)$name$(tput sgr0)"

            python3 test.py "Gaussian-v0" \
                --name=$name \
                --hidden \
                --model="models/$name.pt" \
                --env-kwargs="{shape: $shape}"
            
            # r''
            name="shape-$agent-gaussian-r2$shape-$seed"

            echo "$(date +%T): train $(tput bold)$name$(tput sgr0)"

            python3 train.py "Gaussian-v0" ppo $agent \
                --name=$name \
                --seed=$seed \
                --num-timesteps=$num_timesteps \
                --num-envs=$num_envs \
                --ckpt-interval=$ckpt_interval \
                --env-kwargs="{shape: $shape, reward_closer: true}" \
                --alg-kwargs="params/our.yaml" \
                --agent-kwargs="{}"

            echo "$(date +%T): test $(tput bold)$name$(tput sgr0)"

            python3 test.py "Gaussian-v0" \
                --name=$name \
                --hidden \
                --model="models/$name.pt" \
                --env-kwargs="{shape: $shape}"
        done

        # generalization

        # unlimited
        name="sample-$agent-terrain-inf-$seed"

        echo "$(date +%T): train $(tput bold)$name$(tput sgr0)"

        python3 train.py "Terrain-v0" ppo $agent \
            --name=$name \
            --seed=$seed \
            --num-timesteps=$num_timesteps \
            --num-envs=$num_envs \
                --ckpt-interval=$ckpt_interval \
            --env-kwargs="{}" \
            --alg-kwargs="params/our.yaml" \
            --agent-kwargs="{}"

        echo "$(date +%T): test $(tput bold)$name$(tput sgr0)"

        python3 test.py "Terrain-v0" \
            --name=$name \
            --hidden \
            --model="models/$name.pt"

        # limited
        for samples in 1000 10000 100000
        do
            name="sample-$agent-terrain-$samples-$seed"

            echo "$(date +%T): train $(tput bold)$name$(tput sgr0)"

            python3 train.py "Terrain-v0" ppo $agent \
                --name=$name \
                --seed=$seed \
                --num-timesteps=$num_timesteps \
                --num-envs=$num_envs \
                --ckpt-interval=$ckpt_interval \
                --env-kwargs="{train_samples: $samples}" \
                --alg-kwargs="params/our.yaml" \
                --agent-kwargs="{}"

            echo "$(date +%T): test $(tput bold)$name$(tput sgr0)"

            python3 test.py "Terrain-v0" \
                --name=$name \
                --hidden \
                --model="models/$name.pt"
        done
    done
done

# human comparison

# ablations