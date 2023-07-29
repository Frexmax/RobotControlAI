import pybullet as pb
import pybullet_data
import numpy as np
import math
import random
import time

from numba import jit
from copy import copy


class GripperPushEnv:
    def __init__(self, reward_type="dense", connection_type="direct"):

        self.win_margin = 0.075
        self.step_count = 0
        self.num_sim_steps = 10
        self.max_steps = 100 * self.num_sim_steps
        self.connection_type = connection_type
        self.reward_type = reward_type
        self.num_env_episodes = 0
        self.goal_update_rate = 500
        self.updated_goals = False

        if self.connection_type == "direct":
            self.physicsClient = pb.connect(pb.DIRECT)
        elif self.connection_type == "gui":
            self.physicsClient = pb.connect(pb.GUI)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setGravity(0, 0, -10)

        if self.updated_goals:
            self.goal_limits_x = [0.5, 0.7]
            self.goal_limits_y = [-0.1, 0.1]
            self.goal_limits_z = [0, 0]
        else:
            self.goal_limits_x = [0.5, 0.7]
            self.goal_limits_y = [-0.3, 0.15]
            self.goal_limits_z = [0, 0]

        self.work_area_limits_x = (-2, 3)
        self.work_area_limits_y = (-2, 2)

        self.planeId = pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.targetId = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i in range(7):
            pb.resetJointState(self.targetId, i, rest_poses[i])

        self.tableId = pb.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

        self.resetVelocity = pb.getJointState(self.targetId, 0)[1]
        self.gripper_id = 11
        self.cubeId, self.cube_position = self.add_cube(self.physicsClient)
        self.sphereId, self.goal_position = self.add_sphere(self.physicsClient)

        self.joints_control = []
        for i in range(pb.getNumJoints(self.targetId)):
            joint_type = pb.getJointInfo(self.targetId, i)[2]
            if joint_type is pb.JOINT_REVOLUTE or joint_type is pb.JOINT_PRISMATIC:
                self.joints_control.append(i)

        self.num_joints = pb.getNumJoints(self.targetId)

        self.observation_space = len(self.get_state()['observation'])
        self.action_space = 4
        self.goal_size = 3
        self.current_position = self.get_position()
        self.orientation = self.get_orientation()

        self.aLower = [-1, -1, -1]
        self.aUpper = [1, 1, 1]

        self.fingerUpper = 0.04
        self.fingerLower = 0

        self.locations_x = None
        self.locations_y = None

    @staticmethod
    def add_cube(env):

        cube_id = pb.loadURDF(f"cube.urdf", basePosition=[0.6, 0, 0.01], physicsClientId=env, globalScaling=0.05)
        cube_position = np.array(pb.getBasePositionAndOrientation(cube_id)[0], dtype=np.float32)
        return cube_id, cube_position

    def add_sphere(self, env):
        self.locations_x = random.uniform(self.goal_limits_x[0], self.goal_limits_x[1])
        self.locations_y = random.uniform(self.goal_limits_y[0], self.goal_limits_y[1])

        if 0.55 < self.locations_x > 0.65:
            if self.locations_x > 0.6:
                self.locations_x += np.random.uniform(low=0, high=0.05)
            else:
                self.locations_x += np.random.uniform(low=-0.05, high=0)
        coords = [[0.6, -0.2], [0.5, -0.2], [0.5, -0.1], [0.7, 0.05], [0.7, 0.1], [0.55, -0.225],
                  [0.55, -0.3], [0.5, 0.15], [0.65, 0.15], [0.6, -0.1], [0.5, -0.1]]
        sample = random.sample(coords, 1)[0]
        self.locations_x = sample[0]
        self.locations_y = sample[1]
        sphere_id = pb.loadURDF(f"sphere_small.urdf", basePosition=[self.locations_x, self.locations_y, 0],
                                flags=pb.URDF_IGNORE_COLLISION_SHAPES, physicsClientId=env, useFixedBase=True)

        sphere_position = np.array(pb.getBasePositionAndOrientation(sphere_id)[0], dtype=np.float32)
        return sphere_id, sphere_position

    def update_goal_space(self):
        if (self.num_env_episodes % self.goal_update_rate) and self.updated_goals:
            self.goal_limits_x[1] += 0.01
            self.goal_limits_y[0] -= 0.03
            self.goal_limits_y[1] += 0.01

    def reset(self):

        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setGravity(0, 0, -10)

        self.planeId = pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.targetId = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i in range(9):
            pb.resetJointState(self.targetId, i, rest_poses[i])

        self.tableId = pb.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

        self.current_position = pb.getLinkState(self.targetId, self.gripper_id)[0]
        self.cubeId, self.cube_position = self.add_cube(self.physicsClient)
        self.sphereId, self.goal_position = self.add_sphere(self.physicsClient)
        self.step_count = 0

        return copy(self.get_state())

    def get_state(self):
        final_state = dict()
        state = []
        gripper_position = pb.getLinkState(self.targetId, self.gripper_id)[0]
        gripper_velocity = pb.getLinkState(self.targetId, self.gripper_id, computeLinkVelocity=1)[6]

        cube_position = pb.getBasePositionAndOrientation(self.cubeId)[0]
        cube_rotation = pb.getBasePositionAndOrientation(self.cubeId)[1]
        cube_linear_velocity = pb.getBaseVelocity(self.cubeId)[0]
        cube_angular_velocity = pb.getBaseVelocity(self.cubeId)[1]
        cube_relative_position = cube_position - self.goal_position

        hand_state = (pb.getJointState(self.targetId, 9)[0] + pb.getJointState(self.targetId, 10)[0]) / 2
        hand_velocity = (pb.getJointState(self.targetId, 9)[1] + pb.getJointState(self.targetId, 10)[1]) / 2

        for i in range(3):
            state.append(gripper_position[i])

        for i in range(3):
            state.append(gripper_velocity[i])

        for i in range(3):
            state.append(cube_position[i])

        for i in range(4):
            state.append(cube_rotation[i])

        for i in range(3):
            state.append(cube_relative_position[i])

        for i in range(3):
            state.append(cube_linear_velocity[i])

        for i in range(3):
            state.append(cube_angular_velocity[i])

        state.append(hand_velocity)
        state.append(hand_state)

        final_state['observation'] = np.array(state, dtype=np.float32)
        final_state['achieved_goal'] = np.array(cube_position, dtype=np.float32)
        final_state['desired_goal'] = np.array(self.goal_position, dtype=np.float32)

        return copy(final_state)

    def get_position(self):
        return pb.getLinkState(self.targetId, self.gripper_id)[0]

    def get_orientation(self):
        return pb.getLinkState(self.targetId, self.gripper_id)[1]

    def check_win(self):
        cube_position = np.array(pb.getBasePositionAndOrientation(self.cubeId)[0], dtype=np.float32)
        if self.calculate_distances(cube_position, self.goal_position) < self.win_margin:
            return True
        return False

    def compute_rewards(self, achieved_goal, goal, gripper_pos, cube_velocity):

        @jit(nopython=True, fastmath=True)
        def compute_rewards_get_sparse(distance_to_target_, win_margin):
            return np.where(distance_to_target_ < win_margin, 0, -1)

        @jit(nopython=True, fastmath=True)
        def compute_rewards_get_dense(distance_to_target_, distance_to_gripper_, win_margin):
            return np.where(distance_to_target_ < win_margin, 0, -(distance_to_target_ ** 2 + distance_to_gripper_ * 0.5))

        def reward_from_velocity_her(velocity):
            velocity = np.linalg.norm(velocity)
            return (-velocity * (velocity - 2)) / 20

        distance_to_target = self.calculate_distances(achieved_goal, goal)
        if self.reward_type == "sparse":
            return compute_rewards_get_sparse(distance_to_target, self.win_margin)

        else:
            distance_to_gripper = self.calculate_distances(achieved_goal, gripper_pos)

            v_reward = reward_from_velocity_her(cube_velocity)
            reward = -(0.25 + 0.5 * (math.e ** distance_to_target - 1) + distance_to_gripper) + v_reward
            return np.clip(reward, -1, -0.01)

    @staticmethod
    def calculate_distances(pos1, pos2):
        return np.linalg.norm(pos1 - pos2, axis=-1)

    def check_cube_contact(self):
        contact_left_finger = pb.getContactPoints(bodyA=self.targetId, bodyB=self.cubeId, linkIndexA=9)
        contact_right_finger = pb.getContactPoints(bodyA=self.targetId, bodyB=self.cubeId, linkIndexA=10)
        if len(contact_right_finger) != 0 or len(contact_left_finger) != 0:
            return True
        return False

    def check_ground_contact(self):
        contact_left_finger = pb.getContactPoints(bodyA=self.targetId, bodyB=self.tableId, linkIndexA=9)
        contact_right_finger = pb.getContactPoints(bodyA=self.targetId, bodyB=self.tableId, linkIndexA=10)
        if len(contact_right_finger) != 0 or len(contact_left_finger) != 0:
            return True
        return False

    def reward_from_velocity(self):
        velocity = pb.getBaseVelocity(self.cubeId)[0]
        velocity = np.linalg.norm(velocity)
        return (-velocity * (velocity - 2)) / 20

    def give_reward(self):

        finger_touch = self.check_cube_contact()
        cube_position = np.array(pb.getBasePositionAndOrientation(self.cubeId)[0], dtype=np.float32)
        gripper_position = np.array(pb.getLinkState(self.targetId, self.gripper_id)[0], dtype=np.float32)

        distance_to_gripper = self.calculate_distances(cube_position, gripper_position)
        distance_to_target = self.calculate_distances(cube_position, self.goal_position)
        v_reward = self.reward_from_velocity()

        reward = -(0.25 + 0.5 * (math.e ** distance_to_target - 1) + distance_to_gripper) + v_reward
        return np.clip(reward, -1, -0.01)

    def apply_action(self, action):

        for i in range(self.num_sim_steps):
            dv = 0.05
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv

            current_pose = pb.getLinkState(self.targetId, self.gripper_id)
            current_position = current_pose[0]

            orientation = pb.getQuaternionFromEuler([0., -math.pi, math.pi / 2.])
            new_position = [current_position[0] + dx,
                            current_position[1] + dy,
                            np.clip(current_position[2] + dz, 0.001, 10)]
            finger_position = np.clip(action[3], 0, 0.04)
            joint_positions = pb.calculateInverseKinematics(self.targetId, self.gripper_id, new_position, orientation)[0:7]
            pb.setJointMotorControlArray(self.targetId, list(range(7)) + [9, 10], pb.POSITION_CONTROL,
                                         targetPositions=list(joint_positions) + 2*[finger_position])

            pb.stepSimulation()
            if self.connection_type == "gui":
                time.sleep(4/240)

            if self.check_win() or self.step_count >= self.max_steps:
                break
            self.step_count += 1

    def step(self, action):

        self.apply_action(action)
        state = self.get_state()
        done = False

        if self.reward_type == "dense":
            reward = self.give_reward()
        else:
            reward = -1

        if self.check_win():
            self.num_env_episodes += 1
            done = True
            reward = 0
            self.update_goal_space()
            return copy(state), reward, done

        elif self.step_count >= self.max_steps:
            self.num_env_episodes += 1
            done = True
            self.update_goal_space()
            return copy(state), reward, done

        return copy(state), reward, done
