# ---------------------------------------- PARAMETERS MULTI RISE ---------------------------------------- #

parameter_rise_multi_1 = {"env_name": "rise_multi",
                          "num_time_steps": 1_000_000,
                          "show_ev  ery": 5_000,
                          "save_every": 250_000,
                          "memory_size": 500_000,
                          "batch_size": 512,
                          "test_mode": False,
                          "critic_lrn": 0.001,
                          "actor_lrn": 0.0001,
                          "discount_factor": 0.98,
                          "polyak": 0.005,
                          "start_epsilon": 0.95,
                          "end_epsilon": 0.1,
                          "min_reward": -20,
                          "max_reward": 0,
                          "update_steps": 100,
                          "update_every": 30,
                          "buffer_type": "priority",
                          "her_strategy": "future",
                          "reward_type": "dense",
                          "connection_type": "direct",
                          "dynamic_reward": False,
                          "load": False,
                          "model_paths": [None, None],
                          "model_type": "ddpg",
                          "logger_path": None,
                          "version": 1,
                          "num_workers": 5,
                          "ou_std": 0.2,
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

parameter_rise_multi_2 = {"env_name": "rise_multi",
                          "num_time_steps": 1_000_000,
                          "show_every": 10_000,
                          "save_every": 250_000,
                          "memory_size": 50_000,
                          "batch_size": 256,
                          "test_mode": False,
                          "critic_lrn": 0.001,
                          "actor_lrn": 0.0001,
                          "discount_factor": 0.98,
                          "polyak": 0.005,
                          "start_epsilon": 0,
                          "end_epsilon": 0,
                          "min_reward": -50,
                          "max_reward": 0,
                          "update_steps": 100,
                          "update_every": 30,
                          "buffer_type": "priority",
                          "her_strategy": "future",
                          "reward_type": "dense",
                          "connection_type": "gui",
                          "dynamic_reward": False,
                          "load": True,
                          "model_paths-1": ["Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-value-stage-3-month-2-day-5-80000-95%-dense.h5", "Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-policy-stage-3-month-2-day-5-80000-95%-dense.h5"],
                          "model_paths-2": ["Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-value-stage-3-month-2-day-5-20000-82%-dense.h5", "Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-policy-stage-3-month-2-day-5-20000-82%-dense.h5"],
                          "model_type": "ddpg",
                          "logger_path": None,
                          "version": 1,
                          "num_workers": 1,
                          "ou_std": 0.2,
                          "env_reward_stage": None,
                          "load_buffer": False,
                          "buffer_path": "",
                          "save_buffer": False,
                          "fill_buffer": False,
                          "save_every_buffer": 500_000,
                          "normalize": False,
                          "n_sample_goals": 4,
                          "internal_buffer_type": "uniform",
                          "stage": 1}

parameter_rise_multi_test = {"env_name": "rise_multi",
                             "num_time_steps": 1_000_000,
                             "show_every": 10_000,
                             "save_every": 250_000,
                             "memory_size": 500_000,
                             "batch_size": 512,
                             "test_mode":  True,
                             "critic_lrn": 0.0001,
                             "actor_lrn": 0.001,
                             "discount_factor": 0.98,
                             "polyak": 0.005,
                             "start_epsilon": 0,
                             "end_epsilon": 0,
                             "min_reward": -50,
                             "max_reward": 0,
                             "update_steps": 100,
                             "update_every": 20,
                             "buffer_type": "her",
                             "her_strategy": "future",
                             "reward_type": "dense",
                             "connection_type": "gui",
                             "dynamic_reward": False,
                             "load": True,
                             "model_paths-1": ["Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-value-stage-3-month-2-day-5-80000-95%-dense.h5", "Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-policy-stage-3-month-2-day-5-80000-95%-dense.h5"],
                             "model_paths-2": ["Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-value-stage-3-month-2-day-5-20000-82%-dense.h5", "Trainer/TrainDDPG/SavedModels/RiseModels/rise_multi-ddpg-policy-stage-3-month-2-day-5-20000-82%-dense.h5"],
                             "model_type": "ddpg",
                             "logger_path": f"Logger/Logs/log-rise_multi-ddpg-month-2-day-5-ep-500000-82.5%.pkl",
                             "version": 1,
                             "num_workers": 1,
                             "ou_std": 0,
                             "env_reward_stage": None,
                             "load_buffer": False,
                             "buffer_path": "",
                             "save_buffer": False,
                             "fill_buffer": False,
                             "save_every_buffer": 500_000,
                             "normalize": False,
                             "n_sample_goals": 4,
                             "internal_buffer_type": "uniform",
                             "stage": 1}
