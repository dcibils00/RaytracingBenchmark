import multiprocessing
import numpy as np
from PIL import Image, ImageTk
import time
import tkinter as tk
import random
import os
import psutil
import platform

# Ray tracing utility functions
def normalize(v):
    return v / np.linalg.norm(v)

def intersect_sphere(ray_origin, ray_dir, sphere_center, sphere_radius):
    oc = ray_origin - sphere_center
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = b * b - 4 * c
    if discriminant < 0:
        return False, None
    else:
        t = (-b - np.sqrt(discriminant)) / 2.0
        return True, t

def compute_lighting(point, normal, view_dir, light_sources):
    ambient_strength = 0.1
    diffuse_strength = 1.0
    specular_strength = 0.5
    shininess = 32

    total_lighting = ambient_strength * np.ones(3)

    for light in light_sources:
        light_dir = light['direction']
        light_color = light['color']

        diff = max(np.dot(normal, light_dir), 0.0)
        diffuse = diffuse_strength * diff * light_color

        reflect_dir = normalize(2 * np.dot(normal, light_dir) * normal - light_dir)
        spec = max(np.dot(view_dir, reflect_dir), 0.0)
        specular = specular_strength * (spec ** shininess) * light_color

        total_lighting += diffuse + specular

    return total_lighting

def raytrace_section(width, height, start_x, end_x, start_y, end_y, num_iterations, image_file):
    camera = np.array([0.0, 0.0, -1.0])

    # Define 6 spheres with different properties
    spheres = [
        {"center": np.array([-1.5, 0.0, 4]), "radius": 1, "color": np.array([1.0, 0.0, 0.0])},  # Red sphere
        {"center": np.array([0.0, -0.5, 3]), "radius": 0.75, "color": np.array([0.0, 1.0, 0.0])},  # Green sphere
        {"center": np.array([1.5, 0.5, 5]), "radius": 0.9, "color": np.array([0.0, 0.0, 1.0])},  # Blue sphere
        {"center": np.array([-2.0, -1.0, 3]), "radius": 0.7, "color": np.array([1.0, 1.0, 0.0])},  # Yellow sphere
        {"center": np.array([2.0, -0.5, 6]), "radius": 1.2, "color": np.array([0.5, 0.0, 0.5])},  # Purple sphere
        {"center": np.array([0.5, 1.0, 4.5]), "radius": 0.6, "color": np.array([0.0, 1.0, 1.0])},  # Cyan sphere
    ]

    # Lighting sources (two directional lights)
    light_sources = [
        {"direction": normalize(np.array([-1, -1, -1])), "color": np.array([1.0, 1.0, 1.0])},  # White directional light 1
        {"direction": normalize(np.array([1, 1, -0.5])), "color": np.array([0.5, 0.5, 1.0])},  # Blue directional light 2
    ]

    # Use numpy memmap to map the image file to memory
    image_data = np.memmap(image_file, dtype=np.uint8, mode='r+', shape=(height, width, 3))

    # Compute the pixels for this section
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            x = (2 * (j + 0.5) / width - 1) * width / height
            y = -(2 * (i + 0.5) / height - 1)

            ray_dir = normalize(np.array([x, y, 1]))
            ray_origin = camera
            color = np.array([0, 0, 0])

            closest_t = float('inf')
            hit_sphere = None

            for sphere in spheres:
                hit, t = intersect_sphere(ray_origin, ray_dir, sphere["center"], sphere["radius"])
                if hit and t < closest_t:
                    closest_t = t
                    hit_sphere = sphere

            if hit_sphere:
                hit_point = ray_origin + closest_t * ray_dir
                hit_normal = normalize(hit_point - hit_sphere["center"])
                view_dir = normalize(-ray_dir)
                lighting = compute_lighting(hit_point, hit_normal, view_dir, light_sources)
                color = np.clip(hit_sphere["color"] * lighting, 0, 1) * 255

            image_data[i, j] = color.astype(np.uint8)

    image_data.flush()  # Make sure data is written to the file

def process_sections(section_chunk):
    for section in section_chunk:
        raytrace_section(*section)

def benchmark_cpu_raytrace(width, height, num_processes, iterations_per_process, image_file):
    print(f"Starting ray tracing stress test with {num_processes} processes...")

    # Fixed grid size 30x30
    cell_size = 50
    grid_cols = width // cell_size
    grid_rows = height // cell_size

    # Prepare a list of all the sections to be processed
    sections = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            start_x = col * cell_size
            end_x = (col + 1) * cell_size if col != grid_cols - 1 else width
            start_y = row * cell_size
            end_y = (row + 1) * cell_size if row != grid_rows - 1 else height
            sections.append((width, height, start_x, end_x, start_y, end_y, iterations_per_process, image_file))

    # Now divide the work evenly between processes
    chunk_size = len(sections) // num_processes
    chunks = [sections[i:i + chunk_size] for i in range(0, len(sections), chunk_size)]

    # Start as many processes as there are CPU cores
    processes = []
    for chunk in chunks:
        p = multiprocessing.Process(target=process_sections, args=(chunk,))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

def update_image(canvas, image_file, width, height):
    # Use numpy memmap to read the image data from the file
    image_data = np.memmap(image_file, dtype=np.uint8, mode='r', shape=(height, width, 3))

    # Convert to PIL Image and display in Tkinter canvas
    pil_image = Image.fromarray(image_data)
    tk_image = ImageTk.PhotoImage(pil_image)

    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.image = tk_image
    canvas.update()

def get_system_info():
    cpu_info = psutil.cpu_freq()
    cpu_cores = multiprocessing.cpu_count()
    cpu_name = platform.processor()
    ram_info = psutil.virtual_memory().total / (1024 ** 3)  # Convert bytes to GB
    os_info = platform.system() + " " + platform.release()
    
    return {
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "cpu_freq": cpu_info.current if cpu_info else "N/A",
        "ram": ram_info,
        "os_info": os_info
    }

def progressive_render(root, canvas, width, height, cpu_cores, iterations_per_process):
    # Create a temporary file for memmap
    image_file = 'raytrace_image.dat'

    # Initialize the memmap file
    np.memmap(image_file, dtype=np.uint8, mode='w+', shape=(height, width, 3))

    # Start timing
    start_time = time.time()

    def on_render_complete():
        # Stop timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Get system information
        sys_info = get_system_info()

        # Display system info and time
        info_text = (
            f"Rendering complete in {elapsed_time:.2f} seconds\n"
            f"CPU: {sys_info['cpu_name']} ({sys_info['cpu_cores']} cores)\n"
            f"CPU Frequency: {sys_info['cpu_freq']} MHz\n"
            f"RAM: {sys_info['ram']:.2f} GB\n"
            f"OS: {sys_info['os_info']}"
        )
        canvas.create_text(10, height - 100, anchor=tk.NW, text=info_text, fill="white", font=("Arial", 10))

    # Start ray tracing in parallel
    process = multiprocessing.Process(target=benchmark_cpu_raytrace, args=(width, height, cpu_cores, iterations_per_process, image_file))
    process.start()

    def poll():
        if process.is_alive():
            update_image(canvas, image_file, width, height)
            root.after(100, poll)  # Poll every 100 milliseconds
        else:
            on_render_complete()

    poll()

if __name__ == "__main__":
    cpu_cores = multiprocessing.cpu_count()
    width, height = 800, 800  # Reduce the image size to avoid memory issues
    iterations_per_process = 1

    # Tkinter setup
    root = tk.Tk()
    root.title("Ray Tracing Benchmark")

    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()

    # Start the progressive rendering
    root.after(100, progressive_render, root, canvas, width, height, cpu_cores, iterations_per_process)
    
    root.mainloop()

    # Cleanup the memmap file after use
    if os.path.exists('raytrace_image.dat'):
        os.remove('raytrace_image.dat')
