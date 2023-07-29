import tensorflow as tf


def train(env, online_value_model, target_value_model, online_policy_model, target_policy_model, buffer, discount, p,
          value_optimizer, actor_optimizer):

    if buffer.buffer_counter < 100 * 5:
        return 0, 0, [], []

    state_batch, action_batch, reward_batch, next_state_batch, dones_batch, \
        weights, batch_indices = buffer.get_mini_batch()

    critic_loss, actor_loss, batch_indices, td = update_gradient_ddpg_tf(online_value_model, target_value_model,
                                                                         online_policy_model, target_policy_model,
                                                                         state_batch, action_batch, reward_batch,
                                                                         next_state_batch, dones_batch, discount, p,
                                                                         value_optimizer, actor_optimizer, weights,
                                                                         batch_indices)

    update_target_networks_tf(online_value_model, target_value_model, online_policy_model, target_policy_model, p)
    return critic_loss, actor_loss, batch_indices, td


@tf.function
def update_gradient_ddpg_tf(online_value_model, target_value_model, online_policy_model,
                            target_policy_model, state_batch, action_batch, reward_batch, next_state_batch,
                            dones_batch, discount, p, value_optimizer, actor_optimizer, weights, batch_indices):

    with tf.GradientTape() as tape:

        target_actions = target_policy_model(next_state_batch, training=True)
        y = reward_batch + discount * (1 - dones_batch) * target_value_model(
            [next_state_batch, target_actions], training=True)
        critic_value = online_value_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value) * weights)
    critic_grad = tape.gradient(critic_loss, online_value_model.trainable_variables)

    value_optimizer.apply_gradients(zip(critic_grad, online_value_model.trainable_variables))

    with tf.GradientTape() as tape:
        actions = online_policy_model(state_batch, training=True)
        critic_value = online_value_model([state_batch, actions], training=True)
        actor_loss = -tf.math.reduce_mean(critic_value)
    actor_grad = tape.gradient(actor_loss, online_policy_model.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grad, online_policy_model.trainable_variables))

    # ----------------------- UPDATE TARGET MODELS ----------------------- #

    return critic_loss, actor_loss, batch_indices, y - critic_value


@tf.function
def update_target_networks_tf(online_value_model, target_value_model, online_policy_model, target_policy_model, p):

    online_value_weights = online_value_model.variables
    target_value_weights = target_value_model.variables

    for (a, b) in zip(target_value_weights, online_value_weights):
        a.assign(b * p + a * (1 - p))

    online_policy_weights = online_policy_model.variables
    target_policy_weights = target_policy_model.variables

    for (a, b) in zip(target_policy_weights, online_policy_weights):
        a.assign(b * p + a * (1 - p))
