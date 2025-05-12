import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json
from scipy import ndimage
from matplotlib.patches import Circle, Patch
import random
import math
def generate_test_image(width=8, height=20):
    """
    Generate a test image with obstacles for path planning testing.

    Args:
        width: Width of the grid
        height: Height of the grid
    
    Returns:
        grid: Binary numpy array where 0 is free space and 1 is obstacle
    """
    # Create empty grid
    grid = np.zeros((height, width), dtype=np.uint8)

    # Create L-shaped obstacle
    l_x, l_y = 3, 3
    l_width, l_height = 10, 12
    l_thickness = 3

    # Horizontal part of L
    #grid[l_y:l_y+l_thickness, l_x:l_x+l_width] = 1
    # Vertical part of L
    #grid[l_y:l_y+l_height, l_x:l_x+l_thickness] = 1

    # Create rectangular obstacle in the upper right
    #rect_x, rect_y = 14, 3
    rect_width, rect_height = 5, 5

    obstacles = [[3, 8], [3, 9], [3, 10], [4, 8], [4, 9], [4, 10], [5, 8], [5, 9], [5, 10]]

    for coords in obstacles:
        grid[coords[1] - 1, coords[0] - 1] = 1

    return grid

def load_and_process_image(image_path=None, test_image_size=(6, 13)):
    """
    Load and process an image into a binary grid where 0 is free space and 1 is obstacle.
    If image_path is None, creates a test image with specified obstacles.

    Args:
        image_path: Path to the binary image file (1024x1024 pixels)
        test_image_size: Size of the test image to create if image_path is None

    Returns:
        grid: Binary numpy array where 0 is free space and 1 is obstacle
    """
    if image_path is None:
        # Create a test image with simple obstacles
        return generate_test_image(test_image_size[0], test_image_size[1])
    else:
        # Load real image and convert to binary
        try:
            from PIL import Image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            # Threshold to binary (0 for free space, 1 for obstacle)
            grid = np.array(img) > 127
            grid = grid.astype(np.uint8)
            return grid
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

def inflate_obstacles(grid, drone_diameter):
    """
    Inflate obstacles by the drone's radius to ensure safety margins.
    Uses a more aggressive inflation for corners to prevent diagonal cutting issues.

    Args:
        grid: Binary numpy array where 0 is free space and 1 is obstacle
        drone_diameter: Diameter of the drone in grid cells

    Returns:
        inflated_grid: Binary numpy array with inflated obstacles
    """
    # Calculate drone radius in grid cells
    drone_radius = drone_diameter / 2

    # Create a circular structuring element for dilation
    # Use a slightly larger radius to ensure safety at corners
    safety_factor = 1.1  # 10% extra safety margin
    element_size = int(np.ceil(drone_radius * safety_factor) * 2 + 1)
    center = element_size // 2

    # Create circular structuring element
    y, x = np.ogrid[-center:element_size-center, -center:element_size-center]
    element = x*x + y*y <= (drone_radius * safety_factor)**2

    # Dilate obstacles to create safety margin
    inflated_grid = ndimage.binary_dilation(grid, structure=element).astype(np.uint8)

    return inflated_grid

def find_valid_points(grid, start=(1,1), end=(6, 13)):
    """
    Find all valid points in the grid for the drone center.
    Optionally, randomly select valid start and end points if not provided.

    Args:
        grid: Binary numpy array where 0 is free space and 1 is obstacle
        start: Optional (x, y) coordinates for start point
        end: Optional (x, y) coordinates for end point

    Returns:
        valid_points: List of (x, y) tuples of valid points
        selected_start: Selected or verified start point
        selected_end: Selected or verified end point
    """
    height, width = grid.shape
    valid_points = []

    # Find all valid points (where the drone center can be)
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:  # Free space
                valid_points.append((x, y))

    if not valid_points:
        print("No valid points found in the grid!")
        return [], None, None

    # Verify or select start point
    selected_start = start
    if start is None or grid[start[1], start[0]] == 1:
        selected_start = random.choice(valid_points)

    # Verify or select end point
    selected_end = end
    if end is None or grid[end[1], end[0]] == 1:
        # Make sure end is different from start
        remaining_points = [p for p in valid_points if p != selected_start]
        if remaining_points:
            selected_end = random.choice(remaining_points)
        else:
            print("Cannot select distinct end point!")
            selected_end = None

    return valid_points, selected_start, selected_end

def create_navigation_graph(grid):
    """
    Create a graph representation for navigation, where nodes are valid points
    and edges connect adjacent valid points with proper handling of diagonal movements.

    Args:
        grid: Binary numpy array where 0 is free space and 1 is obstacle

    Returns:
        G: NetworkX graph for navigation
    """
    height, width = grid.shape
    G = nx.Graph()

    # Add nodes for each valid cell
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:  # Free space
                G.add_node((x, y))

    # Add edges between adjacent valid cells
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:  # Free space
                # Check all four adjacent cells (orthogonal moves)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx_pos, ny_pos = x + dx, y + dy

                    # Check if adjacent cell is within grid bounds and is free space
                    if (0 <= nx_pos < width and 0 <= ny_pos < height and
                        grid[ny_pos, nx_pos] == 0):
                        # Add edge with weight 1
                        G.add_edge((x, y), (nx_pos, ny_pos), weight=1)

                # Check diagonal moves with additional safety checks for corners
                for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    nx_pos, ny_pos = x + dx, y + dy

                    # First check if diagonal cell is valid
                    if (0 <= nx_pos < width and 0 <= ny_pos < height and
                        grid[ny_pos, nx_pos] == 0):

                        # CRITICAL: Check that both adjacent cells are also free
                        # This prevents diagonal cutting around corners
                        if grid[y, nx_pos] == 0 and grid[ny_pos, x] == 0:
                            # Add diagonal edge with weight sqrt(2) ≈ 1.414
                            G.add_edge((x, y), (nx_pos, ny_pos), weight=1.414)

    return G

def find_shortest_path(G, start, end):
    """
    Find the shortest path in the graph using Dijkstra's algorithm.

    Args:
        G: NetworkX graph
        start: Start point as (x, y) tuple
        end: End point as (x, y) tuple

    Returns:
        path: List of (x, y) points representing the shortest path
        distance: Total path distance
    """
    # Check if start and end are in the graph
    if start not in G.nodes() or end not in G.nodes():
        print("Start or end point is not in the graph (may be an obstacle)")
        return None, float('inf')

    try:
        # Find the shortest path using Dijkstra's algorithm
        path = nx.dijkstra_path(G, start, end)

        # Calculate the total path distance
        distance = nx.dijkstra_path_length(G, start, end)

        return path, distance
    except nx.NetworkXNoPath:
        print("No path exists between start and end points")
        return None, float('inf')

def visualize_path_with_safety_buffers(original_grid, inflated_grid, drone_diameter, valid_points=None, path=None, start=[1,1], end=[6,13]):
    """
    Visualize the path planning with clear distinction between original obstacles and safety buffers.

    Args:
        original_grid: Binary grid showing original obstacles
        inflated_grid: Binary grid with inflated obstacles
        drone_diameter: Diameter of the drone in grid cells
        valid_points: List of valid points for the drone center
        path: List of (x, y) points representing the path
        start: Start point as (x, y) tuple
        end: End point as (x, y) tuple
    """
    height, width = original_grid.shape
    drone_radius = drone_diameter / 2

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw grid
    for x in range(width + 1):
        ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y, color='lightgray', linestyle='-', linewidth=0.5)

    # First, draw the areas that are in the inflated grid but not in the original grid (safety buffers)
    buffer_zone = np.logical_and(inflated_grid == 1, original_grid == 0)
    for y in range(height):
        for x in range(width):
            if buffer_zone[y, x]:
                ax.fill([x, x+1, x+1, x], [y, y, y+1, y+1], 'lightgray', alpha=0.7)

    # Then, draw the original obstacles (with dark gray color as requested)
    for y in range(height):
        for x in range(width):
            if original_grid[y, x] == 1:
                ax.fill([x, x+1, x+1, x], [y, y, y+1, y+1], 'darkgray')

    # Draw valid points
    if valid_points:
        valid_x = [p[0] for p in valid_points]
        valid_y = [p[1] for p in valid_points]
        ax.scatter([x + 0.5 for x in valid_x],
                  [y + 0.5 for y in valid_y],
                  color='lightblue', s=10, alpha=0.3)

    # Draw path if provided
    if path:
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        path_line = ax.plot(path_x, path_y, 'red', linestyle='-', linewidth=2)[0]

    # Draw drone at select points along the path (to avoid visual clutter)
    if path:
        sampling = max(1, len(path) // 5)  # Show at most 5 intermediate positions
        for i in range(0, len(path), sampling):
            if i == 0 or i == len(path) - 1:
                continue  # Skip start and end, we'll draw them specially
            x, y = path[i]
            drone = Circle((x + 0.5, y + 0.5), drone_radius,
                          fill=False, color='blue', linestyle='-', linewidth=1, alpha=0.3)
            ax.add_patch(drone)

    # Draw start and end positions with drone visualization
    if start and path:
        start_x, start_y = path[0]
        start_point = ax.scatter(start_x + 0.5, start_y + 0.5, color='green', s=100, zorder=5)
        start_drone = Circle((start_x + 0.5, start_y + 0.5), drone_radius,
                            fill=False, color='green', linestyle='-', linewidth=2)
        ax.add_patch(start_drone)

    if end and path:
        end_x, end_y = path[-1]
        end_point = ax.scatter(end_x + 0.5, end_y + 0.5, color='red', s=100, zorder=5)
        end_drone = Circle((end_x + 0.5, end_y + 0.5), drone_radius,
                          fill=False, color='red', linestyle='-', linewidth=2)
        ax.add_patch(end_drone)

    # Create proper legend items
    original_obstacle_patch = Patch(facecolor='darkgray', label='Original Obstacle')
    safety_buffer_patch = Patch(facecolor='lightgray', alpha=0.7, label='Safety Buffer')

    # Add legend with proper colors
    if path:
        ax.legend([path_line, original_obstacle_patch, safety_buffer_patch, start_drone, end_drone],
                 ['Path', 'Original Obstacle', 'Safety Buffer', 'Start Position', 'End Position'],
                 loc='upper left')

    # Set axis properties
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title('Drone Path Planning with Safety Buffers')
    ax.invert_yaxis()  # Invert y-axis to match traditional grid coordinates

    plt.tight_layout()
    plt.show()

def visualize_grid_with_drone(grid, drone_diameter, valid_points=None, path=None, start=None, end=None, show_original_grid=False):
    """
    Visualize the grid with obstacles, valid points, and path.
    Also visualize the drone at start and end positions.

    Args:
        grid: Binary numpy array where 0 is free space and 1 is obstacle (inflated grid)
        drone_diameter: Diameter of the drone in grid cells
        valid_points: List of valid points for the drone center
        path: List of (x, y) points representing the path
        start: Start point as (x, y) tuple
        end: End point as (x, y) tuple
        show_original_grid: If True, will also generate a visualization with the original grid
    """
    height, width = grid.shape
    drone_radius = drone_diameter / 2

    # First visualization - with inflated obstacles
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid
    for x in range(width + 1):
        ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.5)
    for y in range(height + 1):
        ax.axhline(y, color='lightgray', linestyle='-', linewidth=0.5)

    # Draw obstacles (inflated)
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 1:  # Obstacle (inflated)
                ax.fill([x, x+1, x+1, x], [y, y, y+1, y+1], 'darkgray')

    # Draw valid points
    if valid_points:
        valid_x = [p[0] for p in valid_points]
        valid_y = [p[1] for p in valid_points]
        ax.scatter([x + 0.5 for x in valid_x],
                   [y + 0.5 for y in valid_y],
                   color='lightblue', s=20, alpha=0.5)

    # Draw path if provided
    if path:
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        ax.plot(path_x, path_y, 'red', linestyle='-', linewidth=2, marker='o', markersize=4)

    # Draw start and end points
    if start:
        ax.scatter(start[0] + 0.5, start[1] + 0.5, color='green', s=100, zorder=5)
        # No need to draw drone circle here - we're working with inflated obstacles

    if end:
        ax.scatter(end[0] + 0.5, end[1] + 0.5, color='red', s=100, zorder=5)
        # No need to draw drone circle here - we're working with inflated obstacles

    # Set axis limits and labels
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks(np.arange(0.5, width + 0.5))
    ax.set_xticklabels([str(i) for i in range(width)])
    ax.set_yticks(np.arange(0.5, height + 0.5))
    ax.set_yticklabels([str(i) for i in range(height)])
    ax.invert_yaxis()  # Invert y-axis to match traditional grid coordinates

    ax.set_title('Drone Path Planning with Inflated Obstacles')
    plt.tight_layout()
    plt.show()

    # Additional visualization with drone at each path step
    if path and show_original_grid:
        # Create original grid (remove inflation)
        # This is just a simple approximation for visualization
        # In a real implementation, we'd store the original grid
        original_grid = np.zeros_like(grid)
        element_size = int(np.ceil(drone_radius))
        for y in range(height):
            for x in range(width):
                # Mark as obstacle only if center of a larger area is obstacle
                # This is an approximate way to "deflate" the grid
                if grid[max(0, min(y, height-1)), max(0, min(x, width-1))] == 1:
                    original_grid[y, x] = 1

        fig, ax = plt.subplots(figsize=(10, 10))

        # Create approximate original grid for visualization
        # In a real implementation, we'd store the original grid
        original_grid = np.zeros_like(grid)
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 1:
                    # Mark a smaller region as the original obstacle
                    radius = int(drone_radius)
                    center_y, center_x = y, x
                    for dy in range(-radius, radius+1):
                        for dx in range(-radius, radius+1):
                            if dx*dx + dy*dy <= (radius//2)*(radius//2):  # Inner circle
                                ny, nx = center_y + dy, center_x + dx
                                if 0 <= ny < height and 0 <= nx < width:
                                    original_grid[ny, nx] = 1

        # Draw original obstacles first
        '''for y in range(height):
            for x in range(width):
                if original_grid[y, x] == 1:  # Original obstacle
                    ax.fill([x, x+1, x+1, x], [y, y, y+1, y+1], 'black')

        # Draw inflated obstacles (safety buffer) on top
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 1 and original_grid[y, x] == 0:  # Inflated but not original
                    ax.fill([x, x+1, x+1, x], [y, y, y+1, y+1], 'lightgray', alpha=0.5)
'''

        # Draw path
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        path_line = ax.plot(path_x, path_y, 'red', linestyle='-', linewidth=2)[0]

        # Draw drone at select points along the path (to avoid visual clutter)
        sampling = max(1, len(path) // 5)  # Show at most 5 intermediate positions
        drone_circles = []
        for i in range(0, len(path), sampling):
            x, y = path[i]
            alpha = 0.3
            color = 'blue'
            drone = Circle((x + 0.5, y + 0.5), drone_radius,
                          fill=False, color=color, linestyle='-', linewidth=1, alpha=alpha)
            ax.add_patch(drone)
            if i == 0:  # First circle for legend
                drone_circles.append(drone)
        # Always show start and end positions clearly
        # Start position
        #start_x, start_y = path[0]
        start_x, start_y = 1, 1
        start_point = ax.scatter(start_x + 0.5, start_y + 0.5, color='green', s=100, zorder=5)
        start_drone = Circle((start_x + 0.5, start_y + 0.5), drone_radius,
                          fill=False, color='green', linestyle='-', linewidth=2, alpha=1.0)
        ax.add_patch(start_drone)

        # End position
        #end_x, end_y = path[-1]
        end_x, end_y = 6, 13
        end_point = ax.scatter(end_x + 0.5, end_y + 0.5, color='red', s=100, zorder=5)
        end_drone = Circle((end_x + 0.5, end_y + 0.5), drone_radius,
                          fill=False, color='red', linestyle='-', linewidth=2, alpha=1.0)
        ax.add_patch(end_drone)

        # Create proper legend items
        from matplotlib.patches import Patch
        original_obstacle_patch = Patch(facecolor='black', label='Original Obstacle')
        safety_buffer_patch = Patch(facecolor='lightgray', alpha=0.5, label='Safety Buffer')

        # Set axis properties and add legend
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_title('Drone Movement with Original and Inflated Obstacles')
        ax.invert_yaxis()  # Invert y-axis to match traditional grid coordinates

        # Add a proper legend with distinct colors
        ax.legend([path_line, original_obstacle_patch, safety_buffer_patch, start_drone, end_drone],
                 ['Path', 'Original Obstacle', 'Safety Buffer', 'Start Position', 'End Position'],
                 loc='upper left')

        plt.tight_layout()
        plt.show()

def save_path_to_json(path, corners, output_file="drone_path.json"):
    """
    Save the path to a JSON file for integration with other components.

    Args:
        path: List of (x, y) points representing the path
        output_file: Output JSON file path
    """
    if path is None:
        print("No path to save")
        return

    # Convert path to list of dictionaries with x, y coordinates
    path_dict = [{"x": point[0], "y": point[1]} for point in path]
    corners_dict = [{"x": point[0][0], "y": point[0][1]} for point in corners]

    # Create a dictionary with path information
    path_data = {
        "path": path_dict,
        "corners": corners_dict,
        "path_length": len(path),
        "timestamp": "2025-04-27T12:00:00"  # Example timestamp
    }

    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(path_data, f, indent=4)

    print(f"Path saved to {output_file}")

def angle_between_vectors(a, b):
    """Returns the angle in degrees between vectors a and b"""
    dot = a[0]*b[0] + a[1]*b[1]
    mag_a = math.hypot(*a)
    mag_b = math.hypot(*b)
    if mag_a == 0 or mag_b == 0:
        return 0
    cos_theta = dot / (mag_a * mag_b)
    # Clamp to avoid numerical issues
    cos_theta = max(min(cos_theta, 1), -1)
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

def find_corners(path, angle_threshold = 16):
    find_corners = []
    for i in range(1, len(path) - 1):
        p0, p1, p2 = path[i-1], path[i], path[i+1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        angle = angle_between_vectors(v1, v2)
        if np.abs(angle) > angle_threshold:
            find_corners.append((p1, round(angle, 2)))
    return find_corners

def main(image_path=None, drone_diameter=3, start_point=(0,0), end_point=(5,12)):
    """
    Main function to execute the drone path planning process.

    Args:
        image_path: Path to the binary image file (1024x1024 pixels)

        drone_diameter: Diameter of the drone in grid cells
        start_point: Optional (x, y) start point
        end_point: Optional (x, y) end point
    """
    # Step 1: Load and process the image - this is our ORIGINAL grid
    original_grid = load_and_process_image()
    if original_grid is None:
        return

    print(f"Original grid shape: {original_grid.shape}")

    # Step 2: Inflate obstacles based on drone diameter
    inflated_grid = inflate_obstacles(original_grid, drone_diameter)

    # Create a visualization that shows both original and inflated obstacles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot original grid
    ax1.imshow(original_grid, cmap='binary', interpolation='none')
    ax1.set_title('Original Grid')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot inflated grid
    ax2.imshow(inflated_grid, cmap='binary', interpolation='none')
    ax2.set_title(f'Inflated Grid (Drone Diameter = {drone_diameter})')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.show()

    # Explain the inflation concept
    print(f"Notice how obstacles are expanded by {drone_diameter/2} cells in all directions.")
    print("This ensures the drone's perimeter never touches obstacles while its center follows the path.")

    # Step 3: Find valid points and select/verify start and end points
    valid_points, start, end = find_valid_points(inflated_grid, start_point, end_point)

    if start is None or end is None:
        print("Could not determine valid start and end points.")
        return

    print(f"Selected start point: {start}")
    print(f"Selected end point: {end}")
    print(f"Found {len(valid_points)} valid points for drone movement")

    # Step 4: Create navigation graph
    G = create_navigation_graph(inflated_grid)

    # Step 5: Find shortest path
    path, distance = find_shortest_path(G, start, end)

    if path is None:
        print("No valid path found.")
        return

    corners = find_corners(path, angle_threshold=0)

    for point, angle in corners:
        print(f"Corner at {point} with angle {angle}°")

    # Step 6: Visualize the result
    print(f"Found path with {len(path)} steps and total distance {distance:.2f}")

    # Visualize with original and inflated obstacles clearly distinguished
    visualize_path_with_safety_buffers(original_grid, inflated_grid, drone_diameter, valid_points, path, start, end)

    # Step 7: Save path to JSON
    save_path_to_json(path, corners)

# Example usage
if __name__ == "__main__":
    # Create and solve a toy example with a drone diameter of 3 grid cells
    main(drone_diameter=2)
