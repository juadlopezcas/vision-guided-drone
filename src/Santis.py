from ultralytics import SAM
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import os
import math
class Camera:
    def take_photo_with_preview(camera_index, save_path):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Camera {camera_index} not found.")
            return

        ret, frame = cap.read()
        if ret:
            cv2.imshow("Captured Image", frame)  # Show image
            cv2.waitKey(2000)  # Display for 2 seconds before closing
            cv2.imwrite(save_path, cv2.resize(frame, (1024, 1024)))
            print(f"Photo saved as {save_path}")
        
        cap.release()
        cv2.destroyAllWindows() 

    def detect_cameras(max_cameras=5):
        available_cameras = []
        for i in range(max_cameras):  
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras

    def detect_and_capture(max_cameras=5, save_path="photo.jpg"):
        available_cameras = []
        for i in range(max_cameras):  
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if not available_cameras:
            print("No cameras detected.")
            return
        
        print("Available Cameras:", available_cameras)
        cam_index = int(input(f"Select camera index {available_cameras}: "))

        if cam_index not in available_cameras:
            print("Invalid camera selection.")
            return

        # Capture and show image
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print(f"Camera {cam_index} not found.")
            return

        ret, frame = cap.read()
        if ret:
            cv2.imshow("Captured Image", frame)
            cv2.waitKey(2000)  # Show image for 2 seconds
            cv2.imwrite(save_path, frame)
            print(f"Photo saved as {save_path}")
    
        cap.release()
        cv2.destroyAllWindows()


class segmentation:

    def initializeModel(model_name):
    # Load a SAM2.1 base model (you can choose from tiny, small, base, or large variants)
        model = SAM(model_name)
        model.eval()
    # (Optional) Display model info
        model.info()
        return model

   
    def runSegmentation(model, photo):
        results_bbox = model(photo)
        return results_bbox 
    

    def overlayMasks(results_bbox):
        import matplotlib.pyplot as plt
        import numpy as np
        import cv2

        # Get the first result from the output list
        result = results_bbox[0]

        # Extract the segmentation masks (they are likely PyTorch tensors on GPU)
        masks = result.masks.data  # This should be a list/array of tensors

        # Convert each mask to a NumPy array on the CPU
        masks = [mask.cpu().numpy() for mask in masks]

        # Visualize each mask individually
        for i, mask in enumerate(masks):
            plt.figure(figsize=(6, 6))
            plt.imshow(mask, cmap='gray')
            plt.title(f"Segmentation Mask {i+1}")
            plt.axis('off')
            plt.show()

        # Overlay all masks on the original image
        orig_image = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)
        overlay = orig_image.copy()

        # Define some colors for the masks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        alpha = 0.5  # transparency factor

        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title("Original Image with Segmentation Masks Overlay")
        plt.axis('off')
        plt.show()

class pixelpercm:
    def analyze_segmentation_mask(mask_path):
        # Load segmentation mask (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError("Error: Could not load the image. Check the file path.")

        # Force binary: 0 (background), 255 (object)
        binary_mask = np.where(mask == 255, 255, 0).astype(np.uint8)

        # Count pixels for each class
        background_pixels = np.sum(binary_mask == 0)
        object_pixels = np.sum(binary_mask == 255)

        # Estimate dimensions (assuming square)
        estimated_side = math.sqrt(object_pixels)

        # Print results
        print(f"Class 0 (Background): {background_pixels} pixels")
        print(f"Class 255 (Object): {object_pixels} pixels")
        print(f"Estimated object dimensions : {estimated_side:.2f} x {estimated_side:.2f} pixels")

        return {
            "background_pixels": background_pixels,
            "object_pixels": object_pixels,
            "estimated_side_length": estimated_side
        }
     
    def compute_cm_squared_per_pixel(real_width_cm, real_height_cm, pixel_width, pixel_height):
        # Compute total real-world area (cm²)
        real_area_cm2 = real_width_cm * real_height_cm

        # Compute total pixel area
        pixel_area = pixel_width * pixel_height

        # Compute cm² per pixel
        cm_squared_per_pixel = real_area_cm2 / pixel_area

        return cm_squared_per_pixel
    
class obstaclefinder:
    def analyze_grid_and_detect_obstacles(grid_image_path, obstacle_image_path, rows, cols, threshold=150):
        """
        Draws a labeled grid on the obstacle image and highlights obstacle boxes in red.
        Returns a list of (row, col) for each obstacle box.
        """
        # Load images
        obstacle_img_color = cv2.imread(obstacle_image_path)  # To draw on
        obstacle_img_gray = cv2.imread(obstacle_image_path, cv2.IMREAD_GRAYSCALE)
        h, w = obstacle_img_gray.shape
        box_h = h // rows
        box_w = w // cols

        obstacle_coords = []

        # Draw grid and labels over obstacle image
        for r in range(rows):
            y = r * box_h
            y_mid = y + box_h // 2
            cv2.putText(obstacle_img_color, str(r + 1), (5, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for c in range(cols):
                x = c * box_w
                x_mid = x + box_w // 2

                # Draw grid box
                cv2.rectangle(obstacle_img_color, (x, y), (x + box_w, y + box_h), (0, 255, 0), 1)

                if r == 0:
                    cv2.putText(obstacle_img_color, str(c + 1), (x_mid - 5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Check if this box has an obstacle
                roi = obstacle_img_gray[y:y + box_h, x:x + box_w]
                mean_intensity = np.mean(roi)

                if mean_intensity < threshold:
                    obstacle_coords.append((r + 1, c + 1))
                    cv2.rectangle(obstacle_img_color, (x, y), (x + box_w, y + box_h), (0, 0, 255), 2)

        # Show final image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(obstacle_img_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Obstacle Image with Grid and Highlights")
        plt.show()

        print("Boxes with obstacles (row, col):", obstacle_coords)
        return obstacle_coords




