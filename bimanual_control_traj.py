#!/usr/bin/env python3

from aloha.constants import (
    DT_DURATION,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
    InterbotixRobotNode
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rclpy

from threading import Thread
from typing import Optional

from interbotix_common_modules.common_robot.exceptions import InterbotixException
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

import numpy as np
np.set_printoptions(suppress=True,precision=4)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout, Bool

from utils import *
from math_tools import *
import threading
import time

new_action = None
current_action = None
follower_bot_left = None
follower_bot_right = None
gripper_left_command = None
gripper_right_command = None
last_left_openness = None
last_right_openness = None

node = None
current_idx = 0
save_data_idx = 0
current_traj = []

lock = threading.Lock()

env = load_yaml( "/home/jiahe/data/config/env.yaml" )
world_2_head = np.array( env.get("world_2_head") )
world_2_left_base = np.array( env.get("world_2_left_base") )
world_2_right_base = np.array( env.get("world_2_right_base") )
left_ee_2_left_cam = np.array( env.get("left_ee_2_left_cam") )    
right_ee_2_right_cam = np.array( env.get("right_ee_2_right_cam") )
left_bias = world_2_left_base
left_tip_bias = np.array( env.get("left_ee_2_left_tip") )
right_bias = world_2_right_base
right_tip_bias = np.array( env.get("right_ee_2_right_tip") )
right_tip_bias2 = np.array( env.get("right_ee_2_right_tip2") )
right_tip_bias3 = np.array( env.get("right_ee_2_right_tip3") )

task_name = "open_marker"
data_idx = 21
if(task_name == "open_marker"):
    right_tip_bias = right_tip_bias3
if(task_name == "lift_ball"):
    right_tip_bias = right_tip_bias3
if(task_name == "ziploc"):
    right_tip_bias = right_tip_bias2
if(task_name == "handover_block"):
    right_tip_bias = right_tip_bias3


def opening_ceremony(
    follower_bot_left: InterbotixManipulatorXS,
    follower_bot_right: InterbotixManipulatorXS,
) -> None:

    global task_name
    global data_idx



    """Move all 4 robots to a pose where it is easy to start demonstration."""
    # reboot gripper motors, and set operating modes for all motors
    follower_bot_left.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_left.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_left.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_left.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    follower_bot_right.core.robot_reboot_motors('single', 'gripper', True)
    follower_bot_right.core.robot_set_operating_modes('group', 'arm', 'position')
    follower_bot_right.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    follower_bot_right.core.robot_set_motor_registers('single', 'gripper', 'current_limit', 300)

    torque_on(follower_bot_left)
    torque_on(follower_bot_right)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    # print("start_arm_qpos: ", [start_arm_qpos] * 2)
    # start_arm_qpos[4] += 0.4
    data = np.load("/home/jiahe/data/raw_demo/{}/traj/{}.npy".format(task_name, data_idx), allow_pickle=True)

    left_joint = data[0]['left_pos'][0:6] 
    right_joint = data[0]['right_pos'][0:6] 
    start_poses = [ 
        # open_marker
        left_joint,
        right_joint
    ]
    # start_poses = [start_arm_qpos] * 2
    print("start_poses: ", start_poses)
    move_arms(
        [follower_bot_left, follower_bot_right],
        # [start_arm_qpos] * 2,
        start_poses,
        moving_time=4.0,
    )


    # move grippers to starting position
    move_grippers(
        [follower_bot_left, follower_bot_right],
        [1.62, 1.62],
        moving_time=0.5
    )

def custom_ik(goal_transform, current_joints, debug=False ):

    K = 0.4
    result_q, finalerr, success =  RRcontrol(goal_transform, current_joints, K, debug = debug)
    # print("FwdKin: ", FwdKin(result_q))
    # print("Goal: ",goal_transform)
    return result_q, finalerr, success



def callback(multiarray):
    with lock:
        action = np.array(multiarray.data).reshape(-1,2,8)
        global current_action
        global new_action
        new_action = copy.deepcopy(action)
    
def save_data():
    global save_data_idx
    global current__traj
    np.save("traj_track_{}".format(save_data_idx), current_traj, allow_pickle = True)
    save_data_idx += 1

def timer_callback():
    print("in timer call back")
    global current_action
    global new_action
    global current_idx
    global follower_bot_left
    global follower_bot_right
    global gripper_left_command
    global gripper_right_command
    global node
    global current_traj

    if(new_action is not None):
        current_action = copy.deepcopy( new_action )
        new_action = None
        current_idx = 0

    if(current_action is None):
        return
    # if(current_idx >=15):
    #     msg = Bool()
    #     msg.data = True
    #     node.state_publisher.publish(msg)

    # if(current_idx >= current_action.shape[0]):
    #     # save_data()
    #     current_action = None
    #     msg = Bool()
    #     msg.data = True
    #     node.state_publisher.publish(msg)
    #     print("finished !!!")
    #     print("finished !!!")
    #     print("finished !!!")
    #     return
    # print("current: ", current_action[0:3,:,:])
    print("current_idx: ", current_idx)

    # left_bias = get_transform(   [ -0.01, 0.365, 0.00 ,0., 0.,0.02617695, 0.99965732] )
    # left_tip_bias = get_transform( [-0.028, 0.01, 0.01,      0., 0., 0., 1.] ) @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
    # left_goal_bias = get_transform(   [ 0.0, 0., 0.0, 0., 0., 0., 1.0] )

    # right_bias = get_transform(   [ 0.01, -0.315, 0.0, 0., 0., 0., 1.0] )
    # right_tip_bias = get_transform( [-0.035, 0.01, -0.008,      0., 0., 0., 1.] ) @ get_transform([0.087, 0, 0., 0., 0., 0., 1.] )
    # right_goal_bias = get_transform(   [ 0.01, 0.0, 0.01, 0., 0., 0., 1.0] )

    # print("now: ", time.time())
    follower_left_state_joints = follower_bot_left.core.joint_states.position[:6]
    follower_right_state_joints = follower_bot_right.core.joint_states.position[:6]

    # all in robot arm frame
    left_hand_goal = get_7D_transform( inv(left_bias) @ get_transform(current_action[current_idx,0,0:7]) @ inv(left_tip_bias) )
    left_openness = current_action[current_idx,0, 7]

    right_hand_goal = get_7D_transform( inv(right_bias) @ get_transform(current_action[current_idx,1,0:7]) @ inv(right_tip_bias) )
    right_openness = current_action[current_idx,1, 7]

    current_left_joints = np.array( follower_left_state_joints )
    current_right_joints = np.array( follower_right_state_joints )

    start = time.time()
    
    # left hand
    left_transform = get_transform(left_hand_goal)
    left_ik_result, err, success_left = custom_ik( left_transform, current_left_joints, debug=False )
    # right hand
    right_transform = get_transform(right_hand_goal)
    right_ik_result, err, success_right = custom_ik( right_transform, current_right_joints, debug=False )
    end = time.time()
    print("ik time: ", end -start)
    print("err: ", err)
    print("left goal: ", current_action[current_idx,0,0:7])
    
    left_transform = FwdKin( follower_left_state_joints )
    left_transform = left_bias @ left_transform @ left_tip_bias
    
    print("left position: ", get_7D_transform( left_transform ))

    success = success_left and success_left
    # print("success: ", success)
    print()
    
    current_idx += 1
    if(success == False ):
        print("left: ", current_left_joints)
        print("right: ", current_right_joints)
        print("left goal: ", current_action[current_idx,0, 0:7 ])
        print("right goal: ", current_action[current_idx,1, 0:7 ])
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        print("don't have a solution!!!!!!!!!!!!!!!!!!")
        return
    # print("left_ik_result: ", left_ik_result)
    # print("right_ik_result: ", right_ik_result)
    follower_bot_left.arm.set_joint_positions(left_ik_result, blocking=False)
    follower_bot_right.arm.set_joint_positions(right_ik_result, blocking=False)

    current_traj.append([current_left_joints, current_right_joints, left_ik_result, right_ik_result])

    # for pouring task
    # if(left_openness < 0.5):
    #     left_openness = 0.3

    # if(right_openness < 0.88):
    #     right_openness = 0.4


    if(left_openness < 0.7):
        left_openness = 0.2
    else:
        left_openness = 1.0

    if(right_openness < 0.7):
        right_openness = 0.2
    else:
        right_openness = 1.0

    gripper_left_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        left_openness
    )
    gripper_right_command.cmd = LEADER2FOLLOWER_JOINT_FN(
        right_openness
    )
    # print("gripper: ", data_point["right_pos"][6])
    follower_bot_left.gripper.core.pub_single.publish(gripper_left_command)
    follower_bot_right.gripper.core.pub_single.publish(gripper_right_command)


    if(current_idx >= current_action.shape[0]):
        # save_data()
        current_action = None
        msg = Bool()
        msg.data = True
        node.state_publisher.publish(msg)
        print("finished !!!")
        print("finished !!!")
        print("finished !!!")
        return

def main() -> None:
    
    global current_action
    global new_action
    global follower_bot_left
    global follower_bot_right
    global gripper_left_command
    global gripper_right_command
    global node

    node = create_interbotix_global_node('aloha')
    # node = create_bimanual_global_node('bimanual')

    follower_bot_left = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_left',
        node=node,
        iterative_update_fk=False,
    )
    follower_bot_right = InterbotixManipulatorXS(
        robot_model='vx300s',
        robot_name='follower_right',
        node=node,
        iterative_update_fk=False,
    )
    robot_startup(node)

    opening_ceremony(
        follower_bot_left,
        follower_bot_right,
    )

    # press_to_start(leader_bot_left, leader_bot_right)

    # Teleoperation loop
    gripper_left_command = JointSingleCommand(name='gripper')
    gripper_right_command = JointSingleCommand(name='gripper')

    node.bimanual_ee_cmd_sub = node.create_subscription( Float32MultiArray, "bimanual_ee_cmd", callback, 1)
    idx = 0
    timer_period = 0.2  # second
    node.timer = node.create_timer(timer_period, timer_callback)

    node.state_publisher = node.create_publisher(Bool, 'controller_finished', 1)

    rclpy.spin(node)

    robot_shutdown(node)


if __name__ == '__main__':
    main()
