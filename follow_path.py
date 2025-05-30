import time
import json
import logging
import threading
import matplotlib.pyplot as plt
import matplotlib as mpl

SIM_MODE = False  # Set to False when using real drone; False: Sim Mode (bypass connection to drone); True: Drone flight mode

if not SIM_MODE:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.high_level_commander import HighLevelCommander
    from cflib.crazyflie.syncLogger import SyncLogger
    from cflib.crazyflie.log import LogConfig

WAYPOINTS_FILE = "drone_path.json" # input path json here
URI = 'radio://0/80/2M/E7E7E7E7E8' # input the drone's URI here
TAKEOFF_HEIGHT = 0.25
TRAVEL_DURATION = 0.5
GRID_CELL_SIZE = 0.1524  # 6x6 inches grid in meters (grid size)

current_z = 0.0
trajectory = []
current_pos = [0.0, 0.0, 0.0]
offset = [0.0, 0.0]

mpl.style.use('seaborn-v0_8-darkgrid')

def log_callback(timestamp, data, logconf):
    global current_z, current_pos
    current_z = data.get('stateEstimate.z', current_z)
    x = data.get('stateEstimate.x', current_pos[0])
    y = data.get('stateEstimate.y', current_pos[1])
    current_pos = [x, y, current_z]
    trajectory.append((x, y))

def log_thread_func(scf):
    log_conf = LogConfig(name='AltitudeLog', period_in_ms=50)
    log_conf.add_variable('stateEstimate.z', 'float')
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    with SyncLogger(scf, log_conf) as logger:
        for log_entry in logger:
            timestamp, data, logconf = log_entry
            log_callback(timestamp, data, logconf)

def load_path(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        path = [(pt['x'], pt['y']) for pt in data['path']]
        corners = set((pt['x'], pt['y']) for pt in data.get('corners', []))
        return path, corners

def scale_and_shift_path(path, corners, offset):
    dx, dy = offset
    scaled_path = [((x + 0.5) * GRID_CELL_SIZE + dx, (y + 0.5) * GRID_CELL_SIZE + dy) for x, y in path]
    scaled_corners = {((x + 0.5) * GRID_CELL_SIZE + dx, (y + 0.5) * GRID_CELL_SIZE + dy) for x, y in corners}
    return scaled_path, scaled_corners

def setup_plot(path):
    fig, ax = plt.subplots(figsize=(10, 6))
    px, py = zip(*[((x + 0.5) * GRID_CELL_SIZE, (y + 0.5) * GRID_CELL_SIZE) for x, y in path])
    ax.plot(px, py, 'k--', linewidth=1, label='Planned Path')
    drone_dot, = ax.plot([], [], 'r-', linewidth=2, label='Drone Trajectory')

    ax.set_xticks([i * GRID_CELL_SIZE for i in range(21)])
    ax.set_yticks([i * GRID_CELL_SIZE for i in range(14)])
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(0.0, 2.1)
    ax.set_xlabel("X [m] (Grid Left-Right)", fontsize=12)
    ax.set_ylabel("Y [m] (Grid Forward-Back)", fontsize=12)
    ax.set_title("Live Drone Trajectory Tracking", fontsize=14, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.ion()
    plt.show()
    return fig, ax, drone_dot

def update_plot(drone_dot):
    if trajectory:
        x_vals = [p[0] for p in trajectory]
        y_vals = [p[1] for p in trajectory]
        drone_dot.set_data(x_vals, y_vals)
    plt.draw()
    plt.pause(0.01)

def main():
    global current_z, offset
    logging.basicConfig(level=logging.INFO)
    raw_path, raw_corners = load_path(WAYPOINTS_FILE)
    fig, ax, drone_dot = setup_plot(raw_path)

    if SIM_MODE:
        print("\U0001F680 SIMULATION MODE ENABLED")
        print("Taking off...")
        time.sleep(2.0)

        path, corners = scale_and_shift_path(raw_path, raw_corners, (0.0, 0.0))

        print("[SIM] Hovering to stabilize...")
        time.sleep(3.0)

        for idx, (x, y) in enumerate(path):
            print(f"[SIM] Flying to waypoint {idx + 1}/{len(path)}: x={x:.2f}, y={y:.2f}, z={TAKEOFF_HEIGHT:.2f}")
            trajectory.append((x, y))
            start = time.time()
            while time.time() - start < TRAVEL_DURATION:
                update_plot(drone_dot)
                time.sleep(0.05)
            if raw_path[idx] in raw_corners:
                print(f"[SIM] Hovering at corner: x={x:.2f}, y={y:.2f}")
                hover_start = time.time()
                while time.time() - hover_start < 2.0:
                    update_plot(drone_dot)
                    time.sleep(0.05)

        print("[SIM] Hovering before landing...")
        time.sleep(2.0)
        print("[SIM] Landing...")
        time.sleep(2.0)
        update_plot(drone_dot)
        plt.ioff()
        plt.show()
        return

    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        threading.Thread(target=log_thread_func, args=(scf,), daemon=True).start()
        time.sleep(2.0)

        commander = HighLevelCommander(scf.cf)

        print("Taking off...")
        commander.takeoff(TAKEOFF_HEIGHT, 2.0)
        time.sleep(3.0)

        print("Hovering to stabilize...")
        commander.go_to(current_pos[0], current_pos[1], TAKEOFF_HEIGHT, yaw=0.0, duration_s=3.0, relative=False)
        time.sleep(3.0)

        offset = current_pos[0] - (raw_path[0][0] + 0.5) * GRID_CELL_SIZE, current_pos[1] - (raw_path[0][1] + 0.5) * GRID_CELL_SIZE
        path, corners = scale_and_shift_path(raw_path, raw_corners, offset)

        for idx, (x, y) in enumerate(path):
            print(f"Flying to waypoint {idx + 1}/{len(path)}: x={x:.2f}, y={y:.2f}, z={TAKEOFF_HEIGHT:.2f}")
            commander.go_to(x, y, TAKEOFF_HEIGHT, yaw=0.0, duration_s=TRAVEL_DURATION, relative=False)
            start = time.time()
            while time.time() - start < TRAVEL_DURATION + 0.2:
                update_plot(drone_dot)
                time.sleep(0.05)

            trajectory.append((x, y))

            if raw_path[idx] in raw_corners:
                print(f"Hovering at corner: x={x:.2f}, y={y:.2f}")
                commander.go_to(x, y, TAKEOFF_HEIGHT, yaw=0.0, duration_s=1.0, relative=False)
                time.sleep(1.0)

        print("Hovering before landing...")
        commander.go_to(x, y, TAKEOFF_HEIGHT, yaw=0.0, duration_s=2.0, relative=False)
        time.sleep(2.0)

        print("Landing...")
        commander.land(0.0, 2.0)
        time.sleep(3.0)
        update_plot(drone_dot)

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
