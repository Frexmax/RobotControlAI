def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            observation, reward, done = env.step(data)
            if done:
                observation = env.reset()
            remote.send((observation, reward, done))
        elif cmd == "reset":
            observation = env.reset()
            remote.send(observation)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "compute_rewards":
            substitute_reward = env.compute_rewards(data[0], data[1], data[2], data[3])
            remote.send(substitute_reward)
        else:
            raise NotImplementedError
