import tensorflow as tf
from tensorflow.keras import layers


# ---------------------------- RND NETWORKS ---------------------------- #


def create_rnd_networks(num_states):
    initializer = tf.keras.initializers.HeNormal()

    state_input = layers.Input(shape=num_states)
    out = layers.Dense(64, activation="relu", kernel_initializer=initializer)(state_input)
    out = layers.Dense(64, activation="relu", kernel_initializer=initializer)(out)
    out = layers.Dense(64, activation="linear", kernel_initializer=initializer)(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model(state_input, outputs)

    return model

# ---------------------------- GRIPPER REACH ENV (HER) ---------------------------- #


def create_reach_policy_model_her_tf_position(num_states, num_action, upper_bound, goal_size):

    inputs = layers.Input(shape=(num_states + goal_size,))
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(inputs)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)
    output = layers.Dense(num_action, activation="tanh", kernel_regularizer='l2', dtype=tf.float32)(out)
    model = tf.keras.Model(inputs, outputs=output)

    return model


def create_reach_value_model_her_tf_position(num_states, num_actions, goal_size):

    state_input = layers.Input(shape=num_states + goal_size)
    action_input = layers.Input(shape=num_actions)

    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1, dtype=tf.float32)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# ---------------------------- GRIPPER REACH ENV (NO HER) ---------------------------- #


def create_reach_policy_model_tf(num_states, num_action):

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(inputs)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)
    output = layers.Dense(num_action, activation="tanh", kernel_regularizer='l2', dtype=tf.float32)(out)
    model = tf.keras.Model(inputs, outputs=output)

    return model


def create_reach_value_model_tf(num_states, num_actions):

    state_input = layers.Input(shape=num_states)
    action_input = layers.Input(shape=num_actions)

    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1, dtype=tf.float32)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# ---------------------------- GRIPPER RISE ENV ---------------------------- #


def create_rise_policy_model_tf(num_states, num_action, version=0):

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(inputs)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)

    out_pos = layers.Dense(3, activation="tanh", kernel_regularizer='l2')(out)
    out_finger = layers.Dense(1, activation="sigmoid")(out) * 0.04

    if version == 0:
        out_orientation = layers.Dense(3, activation="tanh", kernel_regularizer='l2')(out)
        output = layers.Concatenate()([out_pos, out_finger, out_orientation])
    else:
        output = layers.Concatenate()([out_pos, out_finger])

    model = tf.keras.Model(inputs, outputs=output)

    return model


def create_rise_value_model_tf(num_states, num_actions, version=0):

    state_input = layers.Input(shape=num_states)
    action_input = layers.Input(shape=num_actions)

    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# ---------------------------- GRIPPER PUSH ENV ---------------------------- #


def create_push_policy_model_tf(num_states, num_action):

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(inputs)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)
    out = layers.Dense(64, activation="tanh", kernel_regularizer='l2')(out)

    out_pos = layers.Dense(num_action - 1, activation="tanh", kernel_regularizer='l2')(out)
    out_finger = layers.Dense(1, activation="sigmoid")(out) * 0.04
    output = layers.Concatenate()([out_pos, out_finger])

    model = tf.keras.Model(inputs, outputs=output)

    return model


def create_push_value_model_tf(num_states, num_actions):

    state_input = layers.Input(shape=num_states)
    action_input = layers.Input(shape=num_actions)

    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# ---------------------------- PENDULUM ENV ---------------------------- #


def create_pendulum_policy_model_tf(num_states, num_action, upper_bound):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation=tf.keras.layers.LeakyReLU())(inputs)
    out = layers.Dense(128, activation=tf.keras.layers.LeakyReLU())(out)
    out = layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(out)
    outputs = layers.Dense(num_action, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def create_pendulum_value_model_tf(num_states, num_actions):

    state_input = layers.Input(shape=num_states)
    state_out = layers.Dense(32, activation=tf.keras.layers.LeakyReLU())(state_input)
    state_out = layers.Dense(64, activation=tf.keras.layers.LeakyReLU())(state_out)

    action_input = layers.Input(shape=num_actions)
    action_out = layers.Dense(32, activation=tf.keras.layers.LeakyReLU())(action_input)
    action_out = layers.Dense(32, activation=tf.keras.layers.LeakyReLU())(action_out)

    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(128, activation=tf.keras.layers.LeakyReLU())(concat)
    out = layers.Dense(128, activation=tf.keras.layers.LeakyReLU())(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model
