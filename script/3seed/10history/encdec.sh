python run_benchmark.py  --config-name=benchmark_libero10 \
            --multirun agents=oc_ddpm_agent \
            agent_name=ddpm_encdec \
            group=ddpm_encdec_10obs \
            obs_seq=10 \
            train_batch_size=160 \
            decoder_n_layer=6 \
            seed=0,1,2