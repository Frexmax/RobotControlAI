# ---------------------------------------- PARAMETERS RISE ---------------------------------------- #

parameter_rise_v0 = {"env_name": "rise",
                     "num_time_steps": 1_000_000,
                     "show_every": 25_000,
                     "save_every": 250_000,
                     "memory_size": 500_000,
                     "batch_size": 512,
                     "test_mode": False,
                     "critic_lrn": 0.004,
                     "actor_lrn": 0.0007,
                     "discount_factor": 0.98,
                     "polyak": 0.005,
                     "start_epsilon": 0.95,
                     "end_epsilon": 0.05,
                     "min_reward": -100,
                     "max_reward": 0,
                     "update_steps": 200,
                     "update_every": 10,
                     "buffer_type": "her",
                     "her_strategy": "future",
                     "reward_type": "sparse",
                     "connection_type": "direct",
                     "dynamic_reward": False,
                     "load": False,
                     "model_paths": [None, None],
                     "model_type": "ddpg",
                     "logger_path": None,
                     "version": 1,
                     "num_workers": 5,
                     "ou_std": 0.05,
                     "env_reward_stage": None,
                     "load_buffer": False,
                     "buffer_path": "",
                     "save_buffer": False,
                     "fill_buffer": False,
                     "save_every_buffer": 500_000,
                     "normalize": False,
                     "n_sample_goals": 4,
                     "internal_buffer_type": "uniform",
                     "stage": 0}

parameter_rise_test = {"env_name": "rise",   # NOT ACTIVE
                       "num_time_steps": 1_000_000,
                       "show_every": 25_000,
                       "save_every": 250_000,
                       "memory_size": 500_000,
                       "batch_size": 128,
                       "test_mode": True,
                       "critic_lrn": 0.0001,
                       "actor_lrn": 0.001,
                       "discount_factor": 0.98,
                       "polyak": 0.005,
                       "start_epsilon": 0,
                       "end_epsilon": 0,
                       "min_reward": -30,
                       "max_reward": 0,
                       "update_steps": 100,
                       "update_every": 10,
                       "buffer_type": "her",
                       "her_strategy": "future",
                       "reward_type": "sparse",
                       "connection_type": "gui",
                       "dynamic_reward": False,
                       "load": True,
                       "model_paths": [None, None],
                       "model_type": "ddpg",
                       "logger_path": None,
                       "version": 1,
                       "num_workers": 1,
                       "ou_std": 0.05,
                       "env_reward_stage": None,
                       "load_buffer": False,
                       "buffer_path": "",
                       "save_buffer": False,
                       "fill_buffer": False,
                       "save_every_buffer": 500_000,
                       "normalize": False,
                       "n_sample_goals": 4,
                       "internal_buffer_type": "uniform",
                       "stage": 0}
