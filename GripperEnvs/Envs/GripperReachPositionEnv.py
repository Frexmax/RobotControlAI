import pybullet as pb
import pybullet_data
import numpy as np
import math
import random
import time
from copy import copy
from numba import jit


class GripperReachPositionEnv:
    def __init__(self, reward_type="dense", connection_type="direct", normalized=False):

        self.sphereHeight = 0.1
        self.reward_type = reward_type
        self.step_count = 0
        self.win_margin = 0.05
        self.num_sim_steps = 10
        self.max_steps = 100 * self.num_sim_steps
        self.connection_type = connection_type
        self.normalized = normalized

        if self.connection_type == "direct":
            self.physicsClient = pb.connect(pb.DIRECT)
        elif self.connection_type == "gui":
            self.physicsClient = pb.connect(pb.GUI)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setGravity(0, 0, -10)
        pb.setTimeStep(0.002)

        self.goal_limits_x = (0.5, 0.75)
        self.goal_limits_y = (-0.4, 0.2)
        self.goal_limits_z = (self.sphereHeight, self.sphereHeight)
        self.max_goal_area = (0.6, 0.2, self.sphereHeight)
        self.min_goal_area = (0, -0.5, self.sphereHeight)

        self.work_area_limits_x = (-1, 2)
        self.work_area_limits_y = (-1, 1)
        self.max_work_area = (2, 1, 3)
        self.min_work_area = (-1, -1, 0)

        self.planeId = pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.65], useFixedBase=True)

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.0, 0.0]
        self.targetId = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i in range(7):
            pb.resetJointState(self.targetId, i, rest_poses[i])

        self.tableId = pb.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], useFixedBase=True)

        self.resetVelocity = pb.getJointState(self.targetId, 0)[1]

        self.gripper_id = 11
        self.sphereId, self.goal_position = self.add_sphere(self.physicsClient)

        self.joints_control = []
        for i in range(pb.getNumJoints(self.targetId)):
            joint_type = pb.getJointInfo(self.targetId, i)[2]
            if joint_type is pb.JOINT_REVOLUTE or joint_type is pb.JOINT_PRISMATIC:
                self.joints_control.append(i)

        self.num_joints = pb.getNumJoints(self.targetId)
        self.observation_space = len(self.get_state()['observation'])
        self.action_space = 4

        self.goal_size = self.goal_position.shape[0]
        self.current_position = self.get_position()
        self.orientation = self.get_orientation()

        self.aLower = [-1, -1, -1]
        self.aUpper = [1, 1, 1]

        self.jLower = []
        self.jUpper = []
        for i in self.joints_control:

            if i > 8:
                self.jLower.append(0)
                self.jUpper.append(0)
            else:
                self.jLower.append(pb.getJointInfo(self.targetId, i)[8])
                self.jUpper.append(pb.getJointInfo(self.targetId, i)[9])

        self.locations_x = None
        self.locations_y = None

    def add_sphere(self, env):

        self.locations_x = random.uniform(self.goal_limits_x[0], self.goal_limits_x[1])
        self.locations_y = random.uniform(self.goal_limits_y[0], self.goal_limits_y[1])

        sphere_id = pb.loadURDF(f"sphere_small.urdf", basePosition=[self.locations_x,
                                                                    self.locations_y,
                                                                    self.sphereHeight],
                                flags=pb.URDF_IGNORE_COLLISION_SHAPES, physicsClientId=env, useFixedBase=True)

        sphere_position = np.array(pb.getBasePositionAndOrientation(sphere_id)[0], dtype=np.float32)

        return sphere_id, sphere_position

    def reset(self):

        pb.resetSimulation()
        pb.setPhysicsEngineParameter(numSolverIterations=150)
        pb.setGravity(0, 0, -10)
        pb.setTimeStep(0.002)
        self.planeId = pb.loadURDF("plane.urdf", basePosition=[0, 0, -0.65], useFixedBase=True)

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.targetId = pb.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i in range(9):
            pb.resetJointState(self.targetId, i, rest_poses[i])

        self.tableId = pb.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], useFixedBase=True)
        self.current_position = pb.getLinkState(self.targetId, self.gripper_id)[0]
        self.sphereId, self.goal_position = self.add_sphere(self.physicsClient)
        self.step_count = 0
        return copy(self.get_state())

    def get_state(self):
        final_state = dict()
        state = []
        gripper_position = pb.getLinkState(self.targetId, self.gripper_id)[0]
        gripper_velocity = pb.getLinkState(self.targetId, self.gripper_id, computeLinkVelocity=1)[6]

        if not self.normalized:
            for i in range(3):
                state.append(gripper_position[i])

            for i in range(3):
                state.append(gripper_velocity[i])

            final_state['observation'] = np.array(state, dtype=np.float32)
            final_state['achieved_goal'] = np.array(gripper_position, dtype=np.float32)
            final_state['desired_goal'] = np.array(self.goal_position, dtype=np.float32)

        else:
            scaled_position, scaled_velocity, scaled_goal = self.scale_data(gripper_position,
                                                                            gripper_velocity, self.goal_position)
            for i in range(3):
                state.append(scaled_position[i])

            for i in range(3):
                state.append(scaled_velocity[i])

            final_state['observation'] = np.array(state, dtype=np.float32)
            final_state['achieved_goal'] = np.array(scaled_position, dtype=np.float32)
            final_state['desired_goal'] = np.array(scaled_goal, dtype=np.float32)

        return copy(final_state)

    def scale_data(self, position, velocity, goal):
        scaled_position = []
        scaled_velocity = []
        scaled_goal = []
        for i in range(3):
            scaled_position.append((position[i] - self.min_work_area[i]) / (self.max_work_area[i] -
                                                                            self.min_work_area[i]))
            scaled_velocity.append(velocity[i] / 10)
            if i < 2:
                scaled_goal.append((goal[i] - self.min_goal_area[i]) / (self.max_goal_area[i] -
                                                                        self.min_goal_area[i]))
        scaled_goal.append(self.sphereHeight)

        return scaled_position, scaled_velocity, scaled_goal

    def unscale_position(self, scaled_goal):
        unscaled_goal = []
        for i in range(2):
            unscaled_goal.append(scaled_goal[i] *
                                 (self.max_work_area[i] - self.min_work_area[i]) + self.min_work_area[i])
        unscaled_goal.append(self.sphereHeight)
        return unscaled_goal

    def get_position(self):
        return pb.getLinkState(self.targetId, self.gripper_id)[0]

    def get_orientation(self):
        return pb.getLinkState(self.targetId, self.gripper_id)[1]

    def check_win(self):
        if self.calculate_distance() < self.win_margin:
            return True
        return False

    def compute_rewards(self, achieved_goal, goal, gripper_pos, cube_velocity=0):

        @jit(nopython=True, fastmath=True)
        def compute_rewards_get_sparse(distance_to_target, win_margin):
            return np.where(distance_to_target < win_margin, 0, -1)

        @jit(nopython=True, fastmath=True)
        def compute_rewards_get_dense(distance_to_target, win_margin):
            return np.where(distance_to_target < win_margin, 0, -distance_to_target)

        if self.normalized:
            actual_achieved = np.array(self.unscale_position(achieved_goal))
            actual_goal = np.array(self.unscale_position(achieved_goal))
            distance_to_target = np.linalg.norm(actual_achieved - actual_goal, axis=-1)
        else:
            distance_to_target = np.linalg.norm(achieved_goal - goal, axis=-1)

        if self.reward_type == "dense":
            # return -distance_to_target
            return compute_rewards_get_dense(distance_to_target, self.win_margin)

        else:
            # return np.where(distance < self.win_margin, 0, -1)
            return compute_rewards_get_sparse(distance_to_target, self.win_margin)

    def give_reward(self):
        f = self.calculate_distance()
        return -f

    def calculate_distance(self):
        finger_position = np.array(pb.getLinkState(self.targetId, self.gripper_id)[0])
        return np.linalg.norm(finger_position - self.goal_position, axis=-1)

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
                            current_position[2] + dz]
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
        reward = -1

        if self.step_count >= self.max_steps:
            done = True
            if self.reward_type == "dense":
                reward = self.give_reward()

            return copy(state), reward, done

        if self.check_win():
            done = True
            reward = 0
            return copy(state), reward, done

        if self.reward_type == "dense":
            return copy(state), self.give_reward(), done

        return copy(state), reward, done
