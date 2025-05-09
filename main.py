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
import src.segmentation as seg

def run ():
    finder = seg.obstaclefinder()
    grid_path = "anyphotopath" #clean grid with nothing on it
    obstacle_path = "anyphotopath" #grid with obstacles on it
    rows, cols = 5, 13
    image, obstacles = finder.analyze_grid_and_detect_obstacles(grid_path, obstacle_path, rows, cols)
    return None 

if __name__ == "__main__":
    run()
