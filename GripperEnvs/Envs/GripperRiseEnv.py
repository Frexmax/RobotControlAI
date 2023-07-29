import pybullet_utils.bullet_client as bc
import pybullet as pb
import pybullet_data
import numpy as np
import time

from numba import jit
from copy import copy


class GripperRiseEnv:
    def __init__(self, reward_type="dense", connection_type="direct", dynamic_reward=False, version=0):

        self.version = version
        self.dynamic_reward = dynamic_reward
        self.reward_stage = 0
        self.win_margin = 0.05
        self.step_count = 0
        self.num_sim_steps = 10
        self.max_steps = 100 * self.num_sim_steps
        self.connection_type = connection_type
        self.reward_type = reward_type

        if self.connection_type == "direct":
            self.physicsClient = bc.BulletClient(connection_mode=pb.DIRECT)

        elif self.connection_type == "gui":
            self.physicsClient = bc.BulletClient(connection_mode=pb.GUI)

        self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physicsClient.setPhysicsEngineParameter(numSolverIterations=150)
        self.physicsClient.setGravity(0, 0, -10)

        if self.dynamic_reward:
            self.x_increment = 0.2 / 10
            self.y_increment_up = 0.15 / 10
            self.y_increment_down = -0.35 / 10
            self.z_increment = 0.45 / 10

            self.goal_limits_x = [0.5, 0.6]
            self.goal_limits_y = [-0.05, 0.05]
            self.goal_limits_z = [0.01, 0.05]

        else:
            self.goal_limits_x = [0.5, 0.8]
            self.goal_limits_y = [-0.4, 0.2]
            self.goal_limits_z = [0, 0.5]

            limits_x = (0.5, 0.8)
            self.goal_limits_y = (-0.4, 0.2)

        self.work_area_limits_x = (-2, 3)
        self.work_area_limits_y = (-2, 2)
        self.work_area_limits_z = (-1, 20)

        self.planeId = self.physicsClient.loadURDF("plane.urdf", basePosition=[0, 0, -0.65])

        # rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        rest_poses = [0.037, 0.725, -0.05, -1.82, -0.032, 2.35, 2.33, 0.0, 0.0, 0.025, 0.025, 0.0]
        self.targetId = self.physicsClient.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        for i in range(12):
            self.physicsClient.resetJointState(self.targetId, i, rest_poses[i])

        self.tableId = self.physicsClient.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], useFixedBase=True)
        self.resetVelocity = self.physicsClient.getJointState(self.targetId, 0)[1]

        self.gripper_id = 11
        self.cubeId, self.cube_position = self.add_cube(self.physicsClient)
        self.sphereId, self.goal_position = self.add_sphere(self.physicsClient)

        self.joints_control = []
        for i in range(pb.getNumJoints(self.targetId)):
            joint_type = pb.getJointInfo(self.targetId, i)[2]
            if joint_type is pb.JOINT_REVOLUTE or joint_type is pb.JOINT_PRISMATIC:
                self.joints_control.append(i)

        self.num_joints = self.physicsClient.getNumJoints(self.targetId)
        self.observation_space = len(self.get_state()['observation'])
        if self.version == 0:
            self.action_space = 7
        else:
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
        self.locations_z = None

    def close(self):
        self.physicsClient.disconnect()

    def update_stage(self):

        if self.reward_stage < 2:
            self.reward_stage += 1
        elif 2 <= self.reward_stage < 2.9:
            self.reward_stage += 0.1
            self.goal_limits_x[1] += self.x_increment
            self.goal_limits_y[0] += self.y_increment_down
            self.goal_limits_y[1] += self.y_increment_up
            self.goal_limits_z[1] += self.z_increment
        else:
            pass

    @staticmethod
    def add_cube(env):

        cube_id = env.loadURDF(f"cube.urdf", basePosition=[0.6, 0, 0.01], globalScaling=0.05)
        cube_position = np.array(env.getBasePositionAndOrientation(cube_id)[0], dtype=np.float32)

        return cube_id, cube_position

    def add_sphere(self, env):

        self.locations_x = np.random.uniform(low=self.goal_limits_x[0], high=self.goal_limits_x[1])
        self.locations_y = np.random.uniform(low=self.goal_limits_y[0], high=self.goal_limits_y[1])
        self.locations_z = np.random.uniform(low=self.goal_limits_z[0], high=self.goal_limits_z[1])

        sphere_id = env.loadURDF(f"sphere_small.urdf", basePosition=[self.locations_x, self.locations_y, self.locations_z],
                                 flags=pb.URDF_IGNORE_COLLISION_SHAPES, useFixedBase=True)

        sphere_position = np.array(env.getBasePositionAndOrientation(sphere_id)[0], dtype=np.float32)
        return sphere_id, sphere_position

    def reset(self):

        # rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        rest_poses = [0.037, 0.725, -0.05, -1.82, -0.032, 2.35, 2.33, 0.0, 0.0, 0.025, 0.025, 0.0]
        for i in range(12):
            self.physicsClient.resetJointState(self.targetId, i, rest_poses[i])

        self.current_position = self.physicsClient.getLinkState(self.targetId, self.gripper_id)[0]
        self.physicsClient.removeBody(self.cubeId)
        self.physicsClient.removeBody(self.sphereId)

        self.cubeId, self.cube_position = self.add_cube(self.physicsClient)
        self.sphereId, self.goal_position = self.add_sphere(self.physicsClient)
        self.step_count = 0

        return copy(self.get_state())

    def get_state(self):
        final_state = dict()
        state = []

        gripper_position = self.physicsClient.getLinkState(self.targetId, self.gripper_id)[0]
        gripper_velocity = self.physicsClient.getLinkState(self.targetId, self.gripper_id, computeLinkVelocity=1)[6]

        cube_position = self.physicsClient.getBasePositionAndOrientation(self.cubeId)[0]
        cube_rotation = self.physicsClient.getBasePositionAndOrientation(self.cubeId)[1]
        cube_linear_velocity = self.physicsClient.getBaseVelocity(self.cubeId)[0]
        cube_angular_velocity = self.physicsClient.getBaseVelocity(self.cubeId)[1]
        cube_relative_position = cube_position - self.goal_position

        hand_state = (self.physicsClient.getJointState(self.targetId, 9)[0] + self.physicsClient.getJointState(self.targetId, 10)[0]) / 2
        hand_velocity = (self.physicsClient.getJointState(self.targetId, 9)[1] + self.physicsClient.getJointState(self.targetId, 10)[1]) / 2

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
        return self.physicsClient.getLinkState(self.targetId, self.gripper_id)[0]

    def get_orientation(self):
        return self.physicsClient.getLinkState(self.targetId, self.gripper_id)[1]

    def check_gripper_catch(self):

        x_margin = 0.05
        y_margin = 0.05
        z_margin = 0.1

        left_finger_pos = np.array(self.physicsClient.getLinkState(self.targetId, 9)[0], dtype=np.float32)
        right_finger_pos = np.array(self.physicsClient.getLinkState(self.targetId, 10)[0], dtype=np.float32)
        cube_position = np.array(self.physicsClient.getBasePositionAndOrientation(self.cubeId)[0], dtype=np.float32)

        area_between_fingers_x = (min(left_finger_pos[0], right_finger_pos[0]) - x_margin,
                                  max(left_finger_pos[0], right_finger_pos[0]) + x_margin)

        area_between_fingers_y = (min(left_finger_pos[1], right_finger_pos[1]) - y_margin,
                                  max(left_finger_pos[1], right_finger_pos[1]) + y_margin)

        z_min = cube_position[2] - z_margin
        z_max = cube_position[2] + z_margin
        if (area_between_fingers_x[0] <= cube_position[0] <= area_between_fingers_x[1] and
                area_between_fingers_y[0] <= cube_position[1] <= area_between_fingers_y[1] and
                z_min <= right_finger_pos[2] <= z_max and z_min <= left_finger_pos[2] <= z_max):

            # LOOK CONTACT POINTS
            contact_left_finger = self.physicsClient.getContactPoints(bodyA=self.targetId, bodyB=self.cubeId, linkIndexA=9)
            contact__right_finger = self.physicsClient.getContactPoints(bodyA=self.targetId, bodyB=self.cubeId, linkIndexA=10)
            if len(contact_left_finger) != 0 and len(contact__right_finger) != 0:
                return True
            else:
                return False
        else:
            return False

    def check_win(self):
        cube_position = np.array(self.physicsClient.getBasePositionAndOrientation(self.cubeId)[0], dtype=np.float32)

        if self.dynamic_reward:
            if self.reward_stage == 0:
                gripper_position = np.array(self.physicsClient.getLinkState(self.targetId, self.gripper_id)[0], dtype=np.float32)

                if self.calculate_distances(gripper_position, cube_position) < self.win_margin:
                    return True
                else:
                    return False
            elif self.reward_stage == 1:
                if self.check_gripper_catch():
                    return True
                return False

            elif self.reward_stage == 2:
                if self.calculate_distances(cube_position, self.goal_position) < self.win_margin:
                    return True
                return False

        else:
            if self.calculate_distances(cube_position, self.goal_position) < self.win_margin:
                return True
            return False

    def compute_rewards(self, achieved_goal, goal, gripper_pos):

        @jit(nopython=True, fastmath=True)
        def compute_rewards_get_sparse(distance_to_target_, win_margin):
            return np.where(distance_to_target_ < win_margin, 0, -1)

        @jit(nopython=True, fastmath=True)
        def compute_rewards_get_dense(distance_to_target_, distance_to_gripper_, win_margin):
            return np.where(distance_to_target_ < win_margin, 0, -(distance_to_target_ ** 2 + distance_to_gripper_ * 0.5))

        distance_to_target = self.calculate_distances(achieved_goal, goal)
        if self.reward_type == "sparse":
            # return np.where(distance_to_target < self.win_margin, 0, -1)
            return compute_rewards_get_sparse(distance_to_target, self.win_margin)
        else:
            distance_to_gripper = self.calculate_distances(achieved_goal, gripper_pos)
            return compute_rewards_get_dense(distance_to_target, distance_to_gripper, self.win_margin)

    @staticmethod
    def calculate_distances(pos1, pos2):
        # return np.sqrt(np.sum(np.square(pos1 - pos2)))
        return np.linalg.norm(pos1 - pos2, axis=-1)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def calculate_distances_augmented(pos1, pos2):
        squared_diff = np.square(pos1 - pos2)
        augmented = squared_diff[0:2] + np.square(squared_diff[2])
        return np.sqrt(np.sum(augmented))

    def give_reward(self):

        cube_position = np.array(self.physicsClient.getBasePositionAndOrientation(self.cubeId)[0], dtype=np.float32)
        gripper_position = np.array(self.physicsClient.getLinkState(self.targetId, self.gripper_id)[0], dtype=np.float32)

        distance_to_gripper = self.calculate_distances(cube_position, gripper_position)
        distance_to_target = self.calculate_distances_augmented(cube_position, self.goal_position)

        return -(distance_to_target ** 2 + distance_to_gripper * 0.5)

    def give_dynamic_reward(self):

        cube_position = np.array(self.physicsClient.getBasePositionAndOrientation(self.cubeId)[0], dtype=np.float32)
        gripper_position = np.array(self.physicsClient.getLinkState(self.targetId, self.gripper_id)[0], dtype=np.float32)

        distance_to_gripper = self.calculate_distances(cube_position, gripper_position)
        if self.reward_stage == 0:
            return -distance_to_gripper

        elif self.reward_stage == 1:
            return -distance_to_gripper

        elif self.reward_stage >= 2:
            distance_to_target = self.calculate_distances(cube_position, self.goal_position)
            return -(distance_to_target ** 2 + distance_to_gripper * 0.5)

    def apply_action(self, action):

        for i in range(self.num_sim_steps):
            dv = 0.05
            dx = action[0] * dv
            dy = action[1] * dv
            dz = action[2] * dv
            current_pose = self.physicsClient.getLinkState(self.targetId, self.gripper_id)
            current_position = current_pose[0]

            new_position = [current_position[0] + dx,
                            current_position[1] + dy,
                            current_position[2] + dz]
            finger_position = np.clip(action[3], 0, 0.04)
            if self.version == 0:

                drx = action[4] * dv
                dry = action[5] * dv
                drz = action[6] * dv
                current_orientation = self.physicsClient.getEulerFromQuaternion(current_pose[1])
                new_orientation_euler = [current_orientation[0] + drx,
                                         current_orientation[1] + dry,
                                         current_orientation[2] + drz]
                new_orientation = self.physicsClient.getQuaternionFromEuler(new_orientation_euler)
                joint_positions = self.physicsClient.calculateInverseKinematics(self.targetId, self.gripper_id,
                                                                                new_position, new_orientation)[0:7]
            else:
                joint_positions = self.physicsClient.calculateInverseKinematics(self.targetId, self.gripper_id,
                                                                                new_position)[0:7]

            self.physicsClient.setJointMotorControlArray(self.targetId, list(range(7)) + [9, 10], pb.POSITION_CONTROL,
                                                         targetPositions=list(joint_positions) + 2*[finger_position])

            self.physicsClient.stepSimulation()
            if self.connection_type == "gui":
                time.sleep(2/240)

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
                if self.dynamic_reward:
                    reward = self.give_dynamic_reward()
                else:
                    reward = self.give_reward()

            return copy(state), reward, done

        if self.check_win():
            done = True
            reward = 0

            if self.reward_type == "dense":
                reward = 0

            return copy(state), reward, done

        if self.reward_type == "dense":
            if self.dynamic_reward:
                reward = self.give_dynamic_reward()
            else:
                reward = self.give_reward()
            return copy(state), reward, done

        return copy(state), reward, done
