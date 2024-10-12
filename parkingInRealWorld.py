import numpy as np
from scipy.optimize import minimize
from picarx import Picarx
import time

class ModelPredictiveControl:
    def __init__(self):
        self.horizon = 2
        self.dt = 0.2
        self.reference1 = [1, 2, 0]  # Target position of the parking spot

    def plant_model(self, prev_state, dt, pedal, steering):
        x_t, y_t, psi_t, v_t = prev_state

        a_t = pedal
        beta = steering
        v_t_1 = v_t + a_t * dt - v_t / 25

        x_dot = v_t * np.cos(psi_t)
        y_dot = v_t * np.sin(psi_t)
        psi_dot = v_t * np.tan(beta) / 2.5

        x_t += x_dot * dt
        y_t += y_dot * dt
        psi_t += psi_dot * dt

        return [x_t, y_t, psi_t, v_t_1]

    def cost_function(self, u, *args):
        state, ref = args
        cost = 0.0

        for k in range(self.horizon):
            state = self.plant_model(state, self.dt, u[k * 2], u[k * 2 + 1])
            # Calculate the cost: error between the current position and the target parking spot
            cost += (ref[0] - state[0]) ** 2 + (ref[1] - state[1]) ** 2
            cost += (ref[2] - state[2]) ** 2

        return cost

    def solve_mpc(self, state):
        u0 = [0, 0] * self.horizon
        result = minimize(self.cost_function, u0, args=(state, self.reference1), method='SLSQP')
        return result.x[0], result.x[1]

# Using Picarx library to control the car
if __name__ == "__main__":
    px = Picarx()  # Initialize the Picarx car
    mpc = ModelPredictiveControl()

    # Initial state: x, y, angle, speed
    state = [0, 0, 0, 0]

    # Define maximum values, adjust based on your system specifications
    max_servo_angle = 35  # Maximum servo angle
    max_possible_steering = np.pi / 4  # Assume maximum steering angle is pi/4 radians (adjust as needed)
    max_speed = 6  # Maximum speed for the car
    max_pedal = 5  # Adjust based on MPC output range

    try:
        while True:
            # Use MPC to calculate the optimal pedal (acceleration) and steering angle
            pedal, steering = mpc.solve_mpc(state)

            # Scale the steering value to the servo angle
            steering_angle = np.clip(steering * (max_servo_angle / max_possible_steering), -max_servo_angle, max_servo_angle)
            px.set_dir_servo_angle(int(steering_angle))  # Set servo angle based on scaled steering

            # Scale the pedal value to the speed range
            speed = int(np.clip((pedal / max_pedal) * max_speed, -max_speed, max_speed))

            if speed > 0:
                px.forward(speed)
            elif speed < 0:
                px.backward(-speed)
            else:
                px.forward(0)  # Stop if speed is zero

            # Update the car's state
            state = mpc.plant_model(state, mpc.dt, pedal, steering)

            # Delay to maintain control loop
            time.sleep(mpc.dt)

    finally:
        # Stop the car when the program ends
        px.set_dir_servo_angle(0)
        px.stop()
        time.sleep(0.2)
