#!/usr/bin/env python3

import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

import numpy as np
import tensorflow as tf

import parameters
import Ray_ACNet


def main():
    tf.reset_default_graph()

    a_size = 5
    obs_size = parameters.OBS_SIZE
    num_channel = parameters.NUM_CHANNEL
    global_scope = parameters.GLOBAL_NET_SCOPE

    print("TensorFlow:", tf.__version__)
    print("OBS_SIZE:", obs_size)
    print("NUM_CHANNEL:", num_channel)
    print("GLOBAL_NET_SCOPE:", global_scope)

    with tf.Graph().as_default():
        network = Ray_ACNet.ACNet(
            scope="global",
            a_size=a_size,
            trainer=None,
            TRAINING=False,
            NUM_CHANNEL=num_channel,
            OBS_SIZE=obs_size,
            GLOBAL_NET_SCOPE=global_scope,
            GLOBAL_NETWORK=False,
        )

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            obs = np.zeros(
                (1, num_channel, obs_size, obs_size),
                dtype=np.float32,
            )

            goal_pos = np.zeros((1, 3), dtype=np.float32)
            state_in = network.state_init

            feed = {
                network.inputs: obs,
                network.goal_pos: goal_pos,
                network.state_in[0]: state_in[0],
                network.state_in[1]: state_in[1],
            }

            policy, value, state_out, valids = sess.run(
                [
                    network.policy,
                    network.value,
                    network.state_out,
                    network.valids,
                ],
                feed_dict=feed,
            )

            print("policy shape:", policy.shape)
            print("policy:", policy)
            print("value shape:", value.shape)
            print("value:", value)
            print("valids shape:", valids.shape)
            print("valids:", valids)
            print("state_out shapes:", state_out[0].shape, state_out[1].shape)

    print("Legacy forward probe OK")


if __name__ == "__main__":
    main()