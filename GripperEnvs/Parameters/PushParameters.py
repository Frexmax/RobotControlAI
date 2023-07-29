# ---------------------------------------- PARAMETERS PUSH ---------------------------------------- #

parameter_push_v0 = {"env_name": "push",
                     "num_time_steps": 1_000_000,
                     "show_every": 5_000,
                     "save_every": 250_000,
                     "memory_size": 50_000,
                     "batch_size": 256,
                     "test_mode": False,
                     "critic_lrn": 0.004,
                     "actor_lrn": 0.0007,
                     "discount_factor": 0.98,
                     "polyak": 0.05,
                     "start_epsilon": 0.95,
                     "end_epsilon": 0.1,
                     "min_reward": -50,
                     "max_reward": 0,
                     "update_steps": 100,
                     "update_every": 20,
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
                     "num_workers": 1,
                     "ou_std": 0.1,
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

parameter_push_v0_load = {"env_name": "push",
                          "num_time_steps": 1_000_000,
                          "show_every": 5_000,
                          "save_every": 250_000,
                          "memory_size": 250_000,
                          "batch_size": 256,
                          "test_mode": False,
                          "critic_lrn": 0.004,
                           "actor_lrn": 0.0007,
                          "discount_factor": 0.98,
                          "polyak": 0.05,
                          "start_epsilon": 0.4,
                          "end_epsilon": 0.05,
                          "min_reward": -35,
                          "max_reward": 0,
                          "update_steps": 100,
                          "update_every": 10,
                          "buffer_type": "priority",
                          "her_strategy": "future",
                          "reward_type": "dense",
                          "connection_type": "direct",
                          "dynamic_reward": False,
                          "load": True,
                          "model_paths": ["Trainer/TrainDDPG/SavedModels/PushModels/push-ddpg-value-stage-3-month-2-day-8-555000-72%-dense.h5", "Trainer/TrainDDPG/SavedModels/PushModels/push-ddpg-policy-stage-3-month-2-day-8-555000-72%-dense.h5"],
                          "model_type": "ddpg",
                          "logger_path": "Logger/Logs/log-push-ddpg-month-2-day-8-ep-555000-72.08931419457734%.pkl",
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

parameter_push_test = {"env_name": "push",
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
                       "model_paths": ["Trainer/TrainDDPG/SavedModels/PushModels/push-ddpg-value-stage-3-month-2-day-7-370000-72%-dense.h5", "Trainer/TrainDDPG/SavedModels/PushModels/push-ddpg-policy-stage-3-month-2-day-7-370000-72%-dense.h5"],
                       "model_type": "ddpg",
                       "logger_path": "Logger/Logs/log-push-ddpg-month-2-day-7-ep-370000-72.41379310344827%.pkl",
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
                       "stage": 0}