import pybullet as p
import os
from importlib.resources import files

from rbcx.leap.utils.dynamixel_client import *
import rbcx.leap.utils.leap_hand_utils as lhu
from rbcx.handtracker.mediapipe import MediaPipeHandTracker
from rbcx.leap.node import Leap_Node
from rbcx.smoother.leap_smoother import LeapSmoother

from EMG_regression import EMG_regressor
import config


class Leap_Hand:
    """
    Implementation of a physical control and virtual display of the Leap Hand 
    Call the update method with the angles from the mediapipe handtracker to move the hand
    
    The Leap hand has 16 actuated joints (four per finger for thumb, index,
    middle and ring). Pinky joints are ignored to keep the DOF consistent with the Leap hand.  
    """

    def __init__(self, is_left: bool = False, virtual: bool = True, physical: bool = True):

        self.is_left = is_left
        self.virtual = virtual
        self.physical = physical
        self.mcp_rest_bias = np.deg2rad(45.0)

        if virtual:
            print("Initialising virtual Leap Hand components..")
            self.leap_id, self.virtual_leap_n_joints = self.build_virtual_environment()

        if physical:
            print("Initialising physical Leap Hand components..")
            self.leap_node = Leap_Node()
            self.smoother = LeapSmoother(
                min_cutoff=1.0,  # raise slightly (1.2-1.6) if idle jitter persists
                beta=0.4,        # raise (0.3-0.6) for better fast motion
                v_max=8.0,       # command slew limit (rad/s)
                a_max=120.0,     # command accel limit (rad/s^2)
                deadzone_deg=0.25,
                dip_from_pip=False, dip_ratio=0.66
            )
        
    def build_virtual_environment(self):
        """Create a virtual environment in PyBullet with the leap hand URDF"""

        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Load the hand around the origin
        base_pos = [0.0, 0.0, 0.0]

        if self.is_left:
            urdf_path = os.fspath(
                files("rbcx.leap.assets") / "leap_hand_mesh_left" / "robot_pybullet.urdf"
            )
            base_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        else:
            urdf_path = os.fspath(
                files("rbcx.leap.assets") / "leap_hand_mesh_right" / "robot_pybullet.urdf"
            )
            base_quat = p.getQuaternionFromEuler([np.pi / 2, 0, np.pi])

        # Rotate the hand 90 degrees so the fingers point upward.
        # If it points downward instead, change -np.pi/2 to +np.pi/2.
        extra_rot = p.getQuaternionFromEuler([np.pi/5, -np.pi/8, np.pi / 2])

        # Apply extra rotation on top of the base orientation
        urdf_quat = p.multiplyTransforms(
            [0, 0, 0], base_quat,
            [0, 0, 0], extra_rot
        )[1]

        leap_id = p.loadURDF(
            urdf_path,
            basePosition=base_pos,
            baseOrientation=urdf_quat,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # self._configure_self_collision(leap_id)

        # Center the camera on the hand geometry
        mins, maxs = [], []
        for link_idx in range(-1, p.getNumJoints(leap_id)):
            aabb_min, aabb_max = p.getAABB(leap_id, link_idx)
            mins.append(aabb_min)
            maxs.append(aabb_max)

        aabb_min = np.min(np.array(mins), axis=0)
        aabb_max = np.max(np.array(maxs), axis=0)
        hand_center = ((aabb_min + aabb_max +0.05)  / 2.0).tolist()

        p.resetDebugVisualizerCamera(
            cameraDistance=0.25,
            cameraYaw=15,
            cameraPitch=-50,
            cameraTargetPosition=hand_center
        )

        p.setGravity(0, 0, 0)
        p.setRealTimeSimulation(0)

        virtual_leap_n_joints = p.getNumJoints(leap_id)
        return leap_id, virtual_leap_n_joints
    
    def _configure_self_collision(self, body_id):
        """
        Keep self-collision enabled globally, but disable direct parent-child contacts.
        This avoids the most common self-contact jitter while preserving thumb-vs-finger contact.
        """
        for joint_idx in range(p.getNumJoints(body_id)):
            parent_idx = p.getJointInfo(body_id, joint_idx)[16]  # parentIndex
            if parent_idx >= 0:
                p.setCollisionFilterPair(body_id, body_id, joint_idx, parent_idx, 0)

    def update(self, mp_angles, mp_landmarks=None):
        
        # here leap_angles are the list of 20 elements used for the pyBullet virtual leap hand
        leap_angles = np.array(self.map_mediapipe_angles_to_leap(mp_angles, mp_landmarks))

        if self.virtual:

            # Update the hand joints of the leap hand in virtual environment
            for i in range(self.virtual_leap_n_joints):
                p.setJointMotorControl2(
                    bodyIndex=self.leap_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=leap_angles[i],
                    targetVelocity=0,
                    force=500,
                    positionGain=0.3,
                    velocityGain=1,
                )
            # Step simulation
            p.stepSimulation()
            time.sleep(0.03)

        if self.physical:

            # Update the hand joints of the physical leap hand

            # Smooth angles for the leap hand
            smoothed_leap_angles = self.smoother.update(leap_angles)

            # 0 is a straight hand in ve but 90° closed in physical -> apply -90° to first motor except the thumb (15)
            smoothed_leap_angles[[0, 5, 10]] -= np.deg2rad(90)

            # Swap first two motors of each finger (ve: 1 is the side, in physical: 0 is the side)
            smoothed_leap_angles[0], smoothed_leap_angles[1] = smoothed_leap_angles[1], smoothed_leap_angles[0]
            smoothed_leap_angles[5], smoothed_leap_angles[6] = smoothed_leap_angles[6], smoothed_leap_angles[5]
            smoothed_leap_angles[10], smoothed_leap_angles[11] = smoothed_leap_angles[11], smoothed_leap_angles[10]
            smoothed_leap_angles[15], smoothed_leap_angles[16] = smoothed_leap_angles[16], smoothed_leap_angles[15] 


            # Select the 16 indices corresponding to the physical leap hand motors
            smoothed_leap_angles = smoothed_leap_angles[[0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]] 

            # Move the leap hand
            self.leap_node.set_leap(lhu.allegro_to_LEAPhand(smoothed_leap_angles))
            # print(smoothed_leap_angles)
            # self.leap_node.set_leap(lhu.angle_safety_clip(lhu.allegro_to_LEAPhand(smoothed_leap_angles)))
            

    def wrap_radian(self, angle):
        """
        Map any radian angle to the interval (–π, π].
        In degrees: θ′ = atan2(sin θ, cos θ) always lies in (–π, π] and represents the same orientation.
        """
        return (angle + np.pi) % (2*np.pi) - np.pi
    

    @staticmethod
    def _unit(v, eps=1e-8):
        v = np.asarray(v, dtype=np.float64)
        return v / (np.linalg.norm(v) + eps)

    def _palm_normal(self, lm):
        wrist = lm[0]
        index_mcp = lm[5]
        pinky_mcp = lm[17]
        return self._unit(np.cross(index_mcp - wrist, pinky_mcp - wrist))

    def _finger_mcp_forward(self, lm, mcp_idx, pip_idx):
        palm_n = self._palm_normal(lm)
        v = self._unit(lm[pip_idx] - lm[mcp_idx])   # proximal phalanx direction
        v_in_palm = v - np.dot(v, palm_n) * palm_n
        # 0 rad = finger lies in palm plane (open), increases as the finger curls
        return np.arctan2(abs(np.dot(v, palm_n)), np.linalg.norm(v_in_palm))


    def map_mediapipe_angles_to_leap(self, mp_angles, mp_landmarks=None):
        """
        Input:
        ------- 
        Mediapipe angles format is a list of the following 15 angles:
        Full list of 15 keys: 

            0               1           2           3           4             5             6             7             8
        ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Index_MCP', 'Index_PIP', 'Index_DIP', 'Middle_MCP', 'Middle_PIP', 'Middle_DIP',

            9           10          11          12            13           14
        'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']

        Ouput:
        ------
        Python list of 20 elements used for the virtual representation of the Leap Hand in pyBullet
        """


        # Convert mediapipe angles in degrees to radians
        #print("Mediapipe angles: ", mp_angles)
        mp_angles = np.array(mp_angles)

        if (mp_angles == 0).all():
            defaultAngles = np.zeros(20)
            defaultAngles[15] = np.deg2rad(35)
            defaultAngles[16] = np.deg2rad(35)
            return defaultAngles

        mp_angles -= 180
        mp_angles[[3,6,9]] += 90 
        mp_angles = np.deg2rad(mp_angles)

        leap_angles = np.zeros(self.virtual_leap_n_joints)
        

        # Index finger
        leap_angles[0] = -self.wrap_radian(mp_angles[3])     # Index MCP
        leap_angles[1] = 0                                   # Index MCP side
        leap_angles[2] = -self.wrap_radian(mp_angles[4])     # Index PIP
        leap_angles[3] = -self.wrap_radian(mp_angles[5])     # Index DIP
        leap_angles[4] = 0                                   # Empty / tip
        
        # Middle finger
        leap_angles[5] = -self.wrap_radian(mp_angles[6])     # Middle MCP
        leap_angles[6] = 0                                   # Middle MCP side
        leap_angles[7] = -self.wrap_radian(mp_angles[7])     # Middle PIP
        leap_angles[8] = -self.wrap_radian(mp_angles[8])     # Middle DIP
        leap_angles[9] = 0                                   # Empty / tip

        # Ring finger
        leap_angles[10] = -self.wrap_radian(mp_angles[9])    # Ring MCP
        leap_angles[11] = 0                                  # Ring MCP side
        leap_angles[12] = -self.wrap_radian(mp_angles[10])   # Ring PIP
        leap_angles[13] = -self.wrap_radian(mp_angles[11])   # Ring DIP
        leap_angles[14] = 0                                  # Empty / tip

        pinch = 0.0

        if mp_landmarks is None:
            def map_range(x, a, b, c, d):
                return (x - a) * (d - c) / (b - a) + c
            leap_angles[0] = map_range(leap_angles[0], -0.9, 0.1, -1, 1.5)
            leap_angles[5] = map_range(leap_angles[5], -0.9, 0.1, -1, 1.5)
            leap_angles[10] = map_range(leap_angles[10], -0.9, 0.1, -1, 1.0)
            leap_angles[15] = map_range(leap_angles[15], 0.95, 1.0, 0, 1.5)

        if mp_landmarks is not None:
            lm = np.asarray(mp_landmarks, dtype=np.float64)

            # Drive the proximal LEAP joint from true MCP flexion relative to the palm plane,
            # then add a small synergy from the downstream joints so the base joint closes enough.
            leap_angles[0] = np.clip(
                1.8 * self._finger_mcp_forward(lm, 5, 6) +
                0.20 * max(leap_angles[2], 0.0) +
                0.10 * max(leap_angles[3], 0.0) -
                self.mcp_rest_bias,
                0.0, 2.40
            )

            leap_angles[5] = np.clip(
                1.8 * self._finger_mcp_forward(lm, 9, 10) +
                0.20 * max(leap_angles[7], 0.0) +
                0.10 * max(leap_angles[8], 0.0) -
                self.mcp_rest_bias,
                0.0, 2.40
            )

            leap_angles[10] = np.clip(
                1.8 * self._finger_mcp_forward(lm, 13, 14) +
                0.20 * max(leap_angles[12], 0.0) +
                0.10 * max(leap_angles[13], 0.0) -
                self.mcp_rest_bias,
                0.0, 2.40
            )

            # Pinch cue from human thumb-index distance
            palm_width = np.linalg.norm(lm[5] - lm[17]) + 1e-6
            pinch_dist = np.linalg.norm(lm[4] - lm[8]) / palm_width
            pinch = np.clip((0.85 - pinch_dist) / 0.85, 0.0, 1.0)

            # Let the index come further toward the thumb during pinch
            leap_angles[0] = np.clip(leap_angles[0] + np.deg2rad(18) * pinch, 0.0, 2.40)

        # Thumb
        # Human thumb CMC should drive the two LEAP thumb-base DOFs.
        thumb_cmc = np.clip(-self.wrap_radian(mp_angles[0]), 0.0, np.deg2rad(75))
        thumb_mcp = np.clip(-self.wrap_radian(mp_angles[1]), 0.0, np.deg2rad(95))
        thumb_ip = np.clip(-self.wrap_radian(mp_angles[2]), 0.0, np.deg2rad(95))

        if self.physical:
            # Keep the current natural resting pose, but add opposition from CMC.
            thumb_lat = np.deg2rad(40) + 0.25 * thumb_cmc  # LEAP joint 15
            # thumb_rot = thumb_cmc   # LEAP joint 16
            thumb_rot = np.deg2rad(20) + 1.60 * thumb_cmc  # LEAP joint 16
            thumb_mcp = np.deg2rad(0.0) + 1.3 * thumb_mcp  # LEAP joint 17
            thumb_ip = np.deg2rad(0.0) + 1.3 * thumb_ip  # LEAP joint 18
        else:
            # Keep the current natural resting pose, but add opposition from CMC.
            thumb_lat = np.deg2rad(50) + 0.25 * thumb_cmc  # LEAP joint 15
            thumb_rot = thumb_cmc   # LEAP joint 16

        # Pinch assist: when thumb tip approaches index tip, add extra opposition.
        # Works best with MediaPipe world landmarks, but normalized landmarks also work because we divide by palm width.
        # Doesn't seem to do much though :/
        if mp_landmarks is not None:
            lm = np.asarray(mp_landmarks, dtype=np.float32)

            palm_width = np.linalg.norm(lm[5] - lm[17]) + 1e-6   # index MCP to pinky MCP
            pinch_dist = np.linalg.norm(lm[4] - lm[8]) / palm_width   # thumb tip to index tip
            pinch = np.clip((0.70 - pinch_dist) / 0.70, 0.0, 1.0)

            thumb_lat += np.deg2rad(12) * pinch
            thumb_rot += np.deg2rad(20) * pinch
            thumb_mcp += np.deg2rad(8) * pinch

        leap_angles[15] = np.clip(thumb_lat, np.deg2rad(35), np.deg2rad(80))
        leap_angles[16] = np.clip(thumb_rot, np.deg2rad(35), np.deg2rad(95))
        leap_angles[17] = np.clip(thumb_mcp, 0.0, np.deg2rad(95))
        leap_angles[18] = np.clip(thumb_ip, 0.0, np.deg2rad(95))
        leap_angles[19] = 0



        return leap_angles.tolist()


    def to_pybullet_frame(self, positions: np.ndarray) -> np.ndarray:
        """Convert Unity (left‑handed, Y‑up) vectors to PyBullet’s right‑handed, Z‑up frame."""
        # positions: (N, 3) array [x, y, z] in Unity space
        converted = positions.copy()
        # 1) invert the forward axis to convert left‑handed to right‑handed:contentReference[oaicite:1]{index=1}
        converted[:, 2] *= -1.0
        # 2) swap Unity’s Y‑up with PyBullet’s Z‑up (optional but often needed)
        # PyBullet uses (x right, y forward, z up).  Move Unity Y→Z and Z→Y.
        converted = converted[:, [0, 2, 1]]
        return converted

    def create_target_vis(self) -> None:
        """Create four coloured spheres to visualise the target finger tips."""
        
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        self.ballMbt: list[int] = []
        
        for _ in range(4):
            ball = p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=small_ball_shape, basePosition=basePosition)
            p.setCollisionFilterGroupMask(ball, -1, collisionFilterGroup=0, collisionFilterMask=0)
            self.ballMbt.append(ball)
        
        # Colour them: red (index tip), green (index intermediate), blue (ring tip), white (thumb tip)
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1])
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1])
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])


    def update_target_vis(self, hand_pos: np.ndarray) -> None:
        """Update the visual markers based on current target positions.

        `hand_pos` should be an (8, 3) array representing the selected
        intermediate and tip joints for the index, middle, ring and thumb
        fingers, in that order:

            [thumb_mid, thumb_tip, index_mid, index_tip,
             middle_mid, middle_tip, ring_mid, ring_tip]

        The markers highlight: index tip (red), index middle (green),
        ring tip (blue), thumb tip (white).
        """
        if hand_pos.shape != (8, 3):
            return
        mapping = [3, 2, 7, 1]  # positions in hand_pos for each marker
        for sphere_idx, pos_idx in enumerate(mapping):
            _, ori = p.getBasePositionAndOrientation(self.ballMbt[sphere_idx])
            p.resetBasePositionAndOrientation(self.ballMbt[sphere_idx], hand_pos[pos_idx], ori)

    
if __name__ == "__main__":
    leap_hand = Leap_Hand(virtual=config.VIRTUAL, physical=config.PHYSICAL)

    # Recommended on macOS when PyBullet already gives you a GUI:
    hand_tracker_mediapipe = MediaPipeHandTracker(camera_index=0, mirror=False, show_window=False, use_2D_coord_for_angles=True)
    if config.EMG:
        hand_tracker = EMG_regressor(hand_tracker_mediapipe)
    else:
        hand_tracker = hand_tracker_mediapipe
    hand_tracker.start()

    print("Started")

    try:
        while True:
            hand_state = hand_tracker.get_hand_state("Right")
            mediapipe_joints_angles, mediapipe_landmarks = hand_state["angles_list"], hand_state["landmarks"]
            mediapipe_landmarks = None
            leap_hand.update(mediapipe_joints_angles, mediapipe_landmarks)

            # Optional preview window. Must remain on main thread.
            if hand_tracker.show_window:
                if not hand_tracker.poll_gui():
                    break

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        hand_tracker.stop()
        if leap_hand.virtual:
            p.disconnect()
