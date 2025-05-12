### Write here your master script
### Import libraries
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
from scipy import ndimage
from matplotlib.patches import Circle, Patch
import random
import math
from src import *
from

def run():
    grid_path = ".\img\grid.JPG" #clean grid with nothing on it
    obstacle_path = ".\img\obs.JPG" #grid with obstacles on it
    rows=5
    cols=13
    obstacle_coords = analyze_grid_and_detect_obstacles(grid_path, obstacle_path, rows, cols, threshold=150)
    original_grid = np.zeros((rows, cols), dtype=np.uint8)
    for coords in obstacle_coords:
        original_grid[coords[1] - 1, coords[0] - 1] = 1
    drone_diameter=2
    start_point = (0,0)
    end_point = (5,12)
    inflated_grid = inflate_obstacles(original_grid, drone_diameter)
    valid_points, start, end = find_valid_points(inflated_grid, start_point, end_point)
    
    return None 

if __name__ == "__main__":
    run()
