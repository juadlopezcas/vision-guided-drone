import time
import json
import logging
import threading

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig

WAYPOINTS_FILE = "drone_path.json"
URI = 'radio://0/80/2M'
TAKEOFF_HEIGHT = 0.5  # meters
TRAVEL_DURATION = 1.5  # seconds per waypoint

range_distances = {"Down": 0.0}
current_z = 0.0

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0, output_limit=0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = 0.0
        self.output_limit = output_limit

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return max(min(output, self.output_limit), -self.output_limit)

def log_callback(timestamp, data, logconf):
    global range_distances, current_z
    range_distances["Down"] = data.get('range.zrange', 0) / 1000.0
    current_z = data.get('stateEstimate.z', current_z)

def log_thread_func(scf):
    log_conf = LogConfig(name='ObstacleAvoid', period_in_ms=50)
    log_conf.add_variable('range.zrange', 'uint16_t')
    log_conf.add_variable('stateEstimate.z', 'float')
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def load_path(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        return [(pt['x']*0.15, pt['y']*0.15) for pt in data['path']]

def main():
    global current_z
    logging.basicConfig(level=logging.INFO)
    cflib.crtp.init_drivers()

    path = load_path(WAYPOINTS_FILE)
    for i in path:
        print(i)

    pid_z = PIDController(kp=0.3, ki=0.0, kd=0.1, setpoint=TAKEOFF_HEIGHT)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        # Start logging thread for sensor readings
        threading.Thread(target=log_thread_func, args=(scf,), daemon=True).start()
        time.sleep(2.0)  # Allow Kalman and sensors to stabilize

        commander = HighLevelCommander(scf.cf)

        print("Taking off...")
        commander.takeoff(TAKEOFF_HEIGHT, 2.0)
        time.sleep(3.0)

        for idx, (x, y) in enumerate(path):
            # Smooth altitude adjustment based on down range
            down_range = range_distances.get("Down", 1.0)
            if down_range < 0.3:
                pid_z.setpoint = TAKEOFF_HEIGHT + 0.3
            else:
                pid_z.setpoint = TAKEOFF_HEIGHT

            new_z = TAKEOFF_HEIGHT + pid_z.update(current_z, dt=0.05)

            print(f"Flying to waypoint {idx + 1}/{len(path)}: x={x:.2f}, y={y:.2f}, z={new_z:.2f}")
            commander.go_to(x, y, new_z, yaw=0.0, duration_s=TRAVEL_DURATION, relative=False)
            time.sleep(TRAVEL_DURATION + 0.2)

        print("Landing...")
        commander.land(0.0, 2.0)
        time.sleep(3.0)

if __name__ == '__main__':
    main()
