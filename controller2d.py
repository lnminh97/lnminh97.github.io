#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
import pid
import mrac
import model_estimator
import stanley_controller

# Car dimensions
WHEEL_BASE     = 2.72
WHEEL_RADIUS   = 0.7026
MAX_STEER      = 1.22
T_SAMP         = 0.033
THROTTLE_MODEL = np.array([[-1.959],
                           [0.9592],
                           [     0],
                           [0.1517],
                           [-0.149]])
THROTTLE_REF_MODEL  = np.array([[0.0044, 0.0042],
                                [-1.868, 0.8763]])
INIT_GUESS_THROTTLE = np.array([[0.1517, -0.149],
                                [-1.959, 0.9592]])

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                   = cutils.CUtils()
        self._current_x             = 0
        self._current_y             = 0
        self._current_yaw           = 0
        self._current_speed         = 0
        self._desired_speed         = 0
        self._desired_yaw           = 0
        self._cross_error           = 0
        self._lookahead_cross_error = 0
        self._current_frame         = 0
        self._current_timestamp     = 0
        self._start_control_loop    = False
        self._init_param            = False
        self._set_throttle          = 0
        self._set_brake             = 0
        self._set_steer             = 0
        self._waypoints             = waypoints
        self._conv_rad_to_steer     = 180.0 / 70.0 / np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_motion(self):
        min_idx       = 0
        min_dist      = float("inf")
    
        # Get the reference point based on the shortest distance between car and waypoints
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([self._waypoints[i][0] - self._current_x,
                                            self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        self._desired_speed = self._waypoints[min_idx][2]
        self._desired_yaw = np.arctan2(self._waypoints[-1][1] - self._waypoints[-2][1], \
                                     self._waypoints[-1][0] - self._waypoints[-2][0])

        direction = np.array([np.cos(self._current_yaw), np.sin(self._current_yaw)])
        r = np.array([self._waypoints[min_idx][0] - self._current_x,
                      self._waypoints[min_idx][1] - self._current_y])
        cross_error_sign = np.sign(np.cross(direction, r))
        # cross_error_sign = - np.sign(np.tan(self._current_yaw) * self._waypoints[min_idx][0] -\
        #                            self._waypoints[min_idx][1] -
        #                            np.tan(self._current_yaw) * self._current_x +\
        #                            self._current_y)
        lookahead_cross_error = np.linalg.norm(r)
        self._lookahead_cross_error = cross_error_sign * lookahead_cross_error
        self._cross_error           = cross_error_sign * min_dist

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake

    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_motion()
        v_desired       = self._desired_speed
        yaw_desired     = self._desired_yaw
        cross_error     = self._lookahead_cross_error
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """

        # Init global function
        if not self._init_param:
            # Throttle Self - Tuning and Model Reference vars
            self.vars.create_var('v_previous', np.zeros(2))
            self.vars.create_var('v_desired_previous', 0.0)

            self.vars.create_var('u_previous', 0.0)

            self.vars.create_var('throttle_output_previous', np.zeros(2))

            # self.vars.create_var('b0', INIT_GUESS_THROTTLE[0, 0])
            # self.vars.create_var('b1', INIT_GUESS_THROTTLE[0, 1])
            # self.vars.create_var('b2', INIT_GUESS_THROTTLE[0, 2])
            # self.vars.create_var('a1', INIT_GUESS_THROTTLE[1, 1])
            # self.vars.create_var('a2', INIT_GUESS_THROTTLE[1, 2])
            self.vars.create_var('plant', INIT_GUESS_THROTTLE)
            self.vars.create_var('P_k_1', np.eye(4, dtype=float))

            # # Throttle controller vars
            # self.vars.create_var('throttle_pid_previous', 0.0)
            # self.vars.create_var('throttle_e_previous', 0.0)
            # self.vars.create_var('throttle_e_previous_2', 0.0)

            # Brake controller vars
            self.vars.create_var('brake_previous', 0.0)
            self.vars.create_var('brake_e_previous', np.zeros(2))
            self.vars.create_var('last_brake', False)

            # Stanley controller vars
            self.vars.create_var('e_yaw_previous', np.zeros(2))
            self.vars.create_var('cross_error_previous', np.zeros(2))
            self.vars.create_var('u_yaw_previous', 0.0)
            self.vars.create_var('u_cross_previous', 0.0)

            self._init_param = True

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.

            # Throttle controller based on Model Predictive Control
            # model = longitudinal_model.longitudinal_model()
            # throttle_controller = mpc.mpc(model, T_SAMP, v_desired)
            # # estimator = mhe.mhe(model, T_SAMP)
            # # estimator.y0 = v
            # # x0 = estimator.make_step(y0)
            # x0 = np.dot(np.linalg.pinv(model.C), v)
            # x0 = np.append(x0, [[v]], axis=0)
            # throttle_controller.x0 = x0
            # throttle_controller.set_initial_guess()
            # u0 = throttle_controller.make_step(x0)

            # throttle_output = throttle_controller.u0['u']

            # # Throttle controller is Complementary of PID and Feed Forward Controller
            # # Feed Forward Controller
            # # Relationship between Car's speed and Throttle var is linear regressed as below equation
            # # speed = FF_A * throttle_var + FF_B
            # FF_A = 20.3713
            # FF_B = -4.9084
            # throttle_FF_output = (v_desired - FF_B) / FF_A

            # # PID Controller
            # throttle_P = 1.5
            # throttle_I = 8
            # throttle_D = 2
            # throttle_PID_controller = pid.pid(T_SAMP, throttle_P, throttle_I, throttle_D)
            # throttle_k_1 = self.vars.throttle_pid_previous
            # throttle_e_k_1 = self.vars.throttle_e_previous
            # throttle_e_k_2 = self.vars.throttle_e_previous_2

            # throttle_e_k = v_desired - v

            # throttle_PID_output = throttle_PID_controller.make_control(throttle_k_1, throttle_e_k, throttle_e_k_1, throttle_e_k_2)
            # self.vars.throttle_pid_previous = throttle_PID_output
            # self.vars.throttle_e_previous_2 = self.vars.throttle_e_previous
            # self.vars.throttle_e_previous = throttle_e_k

            # complement_ratio = 0.5          # Ratio of FF Controller in the Complementary Controller
            # throttle_output = complement_ratio * throttle_FF_output + (1 - complement_ratio) * throttle_PID_output


            # Throttel Controller based on Self - Tuning and Model Reference
            # Estimate Model
            # Model estimation excutes if brake was unactive, last command
            if not self.vars.last_brake:
                phi = np.array([[-self.vars.v_previous[0]],
                                [-self.vars.v_previous[1]],
                                [self.vars.throttle_output_previous[0]],
                                [self.vars.throttle_output_previous[1]]])

                theta = self.vars.plant.flatten()
                theta = theta[[2, 3, 0, 1]].reshape(4, 1)

                estimated_model = model_estimator.model(phi, theta)
                estimated_model.make_control(v, self.vars.P_k_1)

                theta = estimated_model.get_theta()
                self.vars.plant = theta[[2, 3, 0, 1]].reshape((2, 2))
                self.vars.P_k_1 = estimated_model.get_P_k_1()

            # Model Reference Control
            throttle_controller = mrac.mrac(THROTTLE_REF_MODEL, self.vars.plant)

            uc = np.array([v_desired, self.vars.v_desired_previous])
            y  = np.array([v, self.vars.v_previous[0]])

            throttle_output = throttle_controller.make_control(uc, y, self.vars.u_previous)

            self.vars.v_desired_previous = v_desired
            self.vars.v_previous[1]      = self.vars.v_previous[0]
            self.vars.v_previous[0]      = v
            self.vars.u_previous         = throttle_output

            self.vars.throttle_output_previous[1] = self.vars.throttle_output_previous[0]
            self.vars.throttle_output_previous[0] = np.fmax(np.fmin(throttle_output, 1.0), 0.0)

            # Brake controller is a PID Controller
            if throttle_output < 0:
                brake_P = 3.7618
                brake_I = 0.9738
                brake_D = 0.3308
                brake_controller = pid.pid(T_SAMP, brake_P, brake_I, brake_D)
                
                brake_e = np.insert(self.vars.brake_e_previous, 0, v - v_desired)

                brake_output = brake_controller.make_control(self.vars.brake_previous, brake_e)

                self.vars.brake_previous      = brake_output
                self.vars.brake_e_previous[1] = self.vars.brake_e_previous[0]
                self.vars.brake_e_previous[0] = v - v_desired

                if brake_output > 0:
                    self.vars.last_brake = True
                else:
                    self.vars.last_brake = False
            else:
                self.vars.last_brake = False
                brake_output = 0

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change the steer output with the lateral controller. 
            # Lateral Controller is based on Stanley Controller
            steer_yaw_P   = 1
            steer_yaw_I   = 0.15
            steer_yaw_D   = 0.9
            steer_cross_P = 1.281
            steer_cross_I = 0.25
            steer_cross_D = 1.5
            steer_K_s     = 10
            steer_controller = stanley_controller.stanley_controller(steer_yaw_P, steer_yaw_I, steer_yaw_D,\
                                                                     steer_cross_P, steer_cross_I, steer_cross_D,\
                                                                     steer_K_s)

            # e_yaw = np.array([yaw_desired - yaw, self.vars.e_yaw_previous])
            e_yaw = np.insert(self.vars.e_yaw_previous, 0, yaw_desired - yaw)
            error = np.insert(self.vars.cross_error_previous, 0, cross_error)
            # error = np.array([cross_error, self.vars.cross_error_previous])

            steer_output = steer_controller.make_control(self.vars.u_yaw_previous,\
                                                         self.vars.u_cross_previous,\
                                                         e_yaw, v, error)

            self.vars.u_yaw_previous, self.vars.u_cross_previous = steer_controller.get_u_previous()

            self.vars.e_yaw_previous[1]       = self.vars.e_yaw_previous[0]
            self.vars.e_yaw_previous[0]       = yaw_desired - yaw
            self.vars.cross_error_previous[1] = self.vars.cross_error_previous[0]
            self.vars.cross_error_previous[0] = cross_error

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        # self.vars.v_previous = v  # Store forward speed to be used in next step
