from GripperRiseEnv import GripperRiseEnv
from GripperPushEnv import GripperPushEnv
from GripperReachPositionEnv import GripperReachPositionEnv
from GripperRiseEvnMutliNetwork import GripperRiseEnvMultiNetwork

from CreateModels import create_rise_policy_model_tf, create_rise_value_model_tf, create_reach_policy_model_tf, \
    create_reach_value_model_tf, create_push_policy_model_tf, create_push_value_model_tf


def create_env(env_type="reach", reward_type="dense", connection_type="direct",
               dynamic_reward=False, version=1, stage=0):
    if env_type == "reach":
        return GripperReachPositionEnv(reward_type=reward_type, connection_type=connection_type)

    elif env_type == "push":
        return GripperPushEnv(reward_type=reward_type, connection_type=connection_type)

    elif env_type == "rise":
        return GripperRiseEnv(reward_type=reward_type, connection_type=connection_type,
                              dynamic_reward=dynamic_reward, version=version)
    else:
        return GripperRiseEnvMultiNetwork(reward_type=reward_type, connection_type=connection_type, train_stage=stage)


def create_model(observation_shape=9, action_shape=4, env_type="reach", version=1):
    if env_type == "reach":
        return create_reach_policy_model_tf(observation_shape, action_shape), \
               create_reach_value_model_tf(observation_shape, action_shape)
    elif env_type == "push":
        return create_push_policy_model_tf(observation_shape, action_shape), \
               create_push_value_model_tf(observation_shape, action_shape)
    else:
        return create_rise_policy_model_tf(observation_shape, action_shape, version), \
               create_rise_value_model_tf(observation_shape, action_shape, version)
