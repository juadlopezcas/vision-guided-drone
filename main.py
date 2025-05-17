import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
from scipy import ndimage
from matplotlib.patches import Circle, Patch
import random
import math
from src import *

def run():
    # This function is the main entry point for the path planning and obstacle detection process.
    # It initializes the grid, detects obstacles, inflates them, finds valid points, and computes the shortest path.
    # It also visualizes the results and saves the path to a JSON file.

    # Load the grid images
    grid_path = "./img/grid.JPG" #clean grid with nothing on it
    obstacle_path = "./img/obs.JPG" #grid with obstacles on it
    # The grid is divided into a 6x12 grid
    rows=6
    cols=12
    # Analyze the grid and detect obstacles
    # The function returns a list of coordinates for each obstacle box
    # The coordinates are in the format (row, col) where row and col are 1-indexed
    obstacle_coords = analyze_grid_and_detect_obstacles(grid_path, obstacle_path, rows, cols, threshold=150)
    original_grid = np.zeros((rows, cols), dtype=np.uint8)
    for coords in obstacle_coords:
        original_grid[coords[0]-1, coords[1]-1] = 1
        
    drone_diameter=2
    start_point = (2,0)
    end_point = (5,11)
    # Inflate the obstacles in the grid to account for the drone's diameter
    inflated_grid = inflate_obstacles(original_grid, drone_diameter)
    valid_points, start, end = find_valid_points(inflated_grid, start_point, end_point)

    print(f"Selected start point: {start}")
    print(f"Selected end point: {end}")
    print(f"Found {len(valid_points)} valid points for drone movement")
    G = create_navigation_graph(inflated_grid)
    path, distance = find_shortest_path(G, start, end)

    if path is None:
        print("No valid path found.")
        return

    corners = find_corners(path, angle_threshold=0)
    #for point, angle in corners:
    #    print(f"Corner at {point} with angle {angle}°")
    # Visualize the result
    print(f"Found path with {len(path)} steps and total distance {distance:.2f}")
    # Visualize with original and inflated obstacles clearly distinguished
    visualize_path_with_safety_buffers(original_grid, inflated_grid, drone_diameter, valid_points, path, start, end)
    # Save path to JSON
    save_path_to_json(path, corners)
    
    return None 

if __name__ == "__main__":
    run()
