import multiprocessing
import numpy as np
import pygame
import time

# Initialize Pygame and font module
pygame.init()
pygame.font.init()

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

    # Ambient lighting (constant across the scene)
    total_lighting = ambient_strength * np.ones(3)

    for light in light_sources:
        light_dir = light['direction']  # Directional light has a constant direction
        light_color = light['color']

        # Diffuse lighting (Lambertian reflection)
        diff = max(np.dot(normal, light_dir), 0.0)
        diffuse = diffuse_strength * diff * light_color

        # Specular lighting (Phong model)
        reflect_dir = normalize(2 * np.dot(normal, light_dir) * normal - light_dir)
        spec = max(np.dot(view_dir, reflect_dir), 0.0)
        specular = specular_strength * (spec ** shininess) * light_color

        # Sum up contributions from each light source
        total_lighting += diffuse + specular

    return total_lighting

def raytrace_section(width, height, start_row, end_row, num_iterations):
    # Scene setup: camera, spheres, and lights
    camera = np.array([0.0, 0.0, -1.0])

    spheres = [
        {"center": np.array([-1.5, 0.0, 4]), "radius": 1, "color": np.array([1.0, 0.0, 0.0])},  # Red sphere
        {"center": np.array([0.0, -0.5, 3]), "radius": 0.75, "color": np.array([0.0, 1.0, 0.0])},  # Green sphere
        {"center": np.array([1.5, 0.5, 5]), "radius": 0.9, "color": np.array([0.0, 0.0, 1.0])},  # Blue sphere
    ]

    # Lighting sources (two directional lights)
    light_sources = [
        {"direction": normalize(np.array([-1, -1, -1])), "color": np.array([1.0, 1.0, 1.0])},  # White directional light 1
        {"direction": normalize(np.array([1, 1, -0.5])), "color": np.array([0.5, 0.5, 1.0])},  # Blue directional light 2
    ]

    pixel_data = []

    # Ray tracing loop
    for _ in range(num_iterations):
        for i in range(start_row, end_row):
            row_pixels = []
            for j in range(width):
                # Normalize screen coordinates to [-1, 1]
                x = (2 * (j + 0.5) / width - 1) * width / height
                y = -(2 * (i + 0.5) / height - 1)

                # Ray direction
                ray_dir = normalize(np.array([x, y, 1]))
                ray_origin = camera

                # Default background color (black)
                color = np.array([0, 0, 0])

                # Check for intersections with spheres
                closest_t = float('inf')
                hit_sphere = None
                hit_point = None
                hit_normal = None

                for sphere in spheres:
                    hit, t = intersect_sphere(ray_origin, ray_dir, sphere["center"], sphere["radius"])
                    if hit and t < closest_t:
                        closest_t = t
                        hit_sphere = sphere
                        hit_point = ray_origin + t * ray_dir
                        hit_normal = normalize(hit_point - sphere["center"])

                # If a sphere was hit, calculate lighting
                if hit_sphere:
                    # Compute view direction
                    view_dir = normalize(-ray_dir)

                    # Compute lighting from both directional lights
                    lighting = compute_lighting(hit_point, hit_normal, view_dir, light_sources)
                    
                    # Apply lighting to the sphere's base color
                    color = np.clip(hit_sphere["color"] * lighting, 0, 1) * 255

                # Store pixel data (x, y, color)
                row_pixels.append((j, i, color.astype(int)))
            pixel_data.append(row_pixels)
    
    return pixel_data

def display_benchmark_info(screen, width, height, cpu_cores, elapsed_time, score):
    # Define font and text color
    font = pygame.font.SysFont('Arial', 18)
    text_color = (255, 255, 255)

    # Create the info strings
    info_lines = [
        f"RayTracing Benchmark",
        f"CPU Cores Used: {cpu_cores}",
        f"Time Taken: {elapsed_time:.2f} seconds",
        f"Benchmark Score: {score:.2f}",
    ]

    # Display each line of text
    for idx, line in enumerate(info_lines):
        text_surface = font.render(line, True, text_color)
        screen.blit(text_surface, (10, height - 120 + idx * 30))  # Display text near the bottom of the screen

    pygame.display.flip()

def display_raytracing_message(screen, width, height):
    """Display the 'Raytracing...' message before the rendering starts."""
    screen.fill((0, 0, 0))  # Clear the screen with black

    # Define font and message
    font = pygame.font.SysFont('Arial', 40)
    text_surface = font.render("Raytracing...", True, (255, 255, 255))

    # Position the message at the center of the screen
    screen.blit(text_surface, ((width - text_surface.get_width()) // 2, (height - text_surface.get_height()) // 2))

    pygame.display.flip()

def benchmark_cpu_raytrace(width, height, num_processes, iterations_per_process):
    print(f"Starting ray tracing stress test with {num_processes} processes...")

    # Initialize Pygame screen
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Ray Tracing Benchmark")

    # Display the 'Raytracing...' message before starting the rendering
    display_raytracing_message(screen, width, height)

    # Wait for a short moment to let the user see the "Raytracing..." message
    pygame.time.delay(2000)

    start_time = time.time()

    # Divide work between processes
    rows_per_process = height // num_processes
    process_args = []
    for i in range(num_processes):
        start_row = i * rows_per_process
        end_row = (i + 1) * rows_per_process if i != num_processes - 1 else height
        process_args.append((width, height, start_row, end_row, iterations_per_process))

    # Create a pool of workers to compute sections of the image
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(raytrace_section, process_args)

    # Flatten the result and update the screen
    for section in results:
        for row_pixels in section:
            for x, y, color in row_pixels:
                screen.set_at((x, y), color)

            # Update the display after every row and handle events
            pygame.display.flip()

            # Handle Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

    # Calculate benchmark score
    end_time = time.time()
    elapsed_time = end_time - start_time
    score = (width * height * num_processes * iterations_per_process) / elapsed_time

    # Display the benchmark info on screen
    display_benchmark_info(screen, width, height, num_processes, elapsed_time, score)

    print(f"Benchmark completed in {elapsed_time:.2f} seconds")
    print(f"CPU Benchmark Score: {score:.2f}")

if __name__ == "__main__":
    # Get the number of CPU cores
    cpu_cores = multiprocessing.cpu_count()

    # Image dimensions for ray tracing
    width, height = 1000, 1000  # Increased resolution to test progressive rendering

    # Number of iterations per process (higher values stress CPU more)
    iterations_per_process = 1

    # Run the benchmark
    benchmark_cpu_raytrace(width, height, cpu_cores, iterations_per_process)

    # Event loop to keep the window open until closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
