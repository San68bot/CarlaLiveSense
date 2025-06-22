import carla
import numpy as np
import cv2
import math
import queue
import os
import time
import random

# --- Configuration ---
MAP_NAME = 'Town03'
HOST = 'localhost'
PORT = 2000
TM_PORT = 8000
SENSOR_TICK = 0.05
CAM_W, CAM_H = 1280, 720
CAM_ANGLE_FOV = 90.0
LIDAR_RANGE = 100.0
LIDAR_POINTS_PER_SEC = 300000
LIDAR_ROT_FREQ = 20
VIDEO_FILENAME = 'carla_bev_dashboard_FHD_v2_20s.avi' # New filename
OUTPUT_DIR = 'output_video'
RECORD_SECONDS = 30

# <<< Updated Dashboard Configuration (1920x1080 with Taller Bottom) >>>
FUSED_W, FUSED_H = 1920, 720 # Top panel height reduced
SMALL_W, SMALL_H = 960, 360   # Bottom panel height increased
VIDEO_W = FUSED_W
VIDEO_H = FUSED_H + SMALL_H # 720 + 360 = 1080

# BEV Configuration
BEV_IMAGE_SIZE = 960
BEV_RANGE_M = 100
BEV_SCALE = BEV_IMAGE_SIZE / BEV_RANGE_M
BEV_POINT_RADIUS = 2

# --- Global Variables ---
K_CAM = None
image_queue = queue.Queue()
lidar_queue = queue.Queue()

# --- Functions ---

def get_cam_intrinsics(w, h, fov_deg):
    fov_rad = math.radians(fov_deg)
    focal_len = w / (2 * math.tan(fov_rad / 2))
    k_mat = np.array([
        [focal_len, 0, w / 2],
        [0, focal_len, h / 2],
        [0, 0, 1]
    ])
    return k_mat

def carla_transform_to_np(transform):
    return np.array(transform.get_matrix())

def project_pts_to_img(rgb_img, lidar_pts_2d, depths_m):
    vis_img = rgb_img.copy()
    img_h, img_w, _ = vis_img.shape

    if lidar_pts_2d is None or lidar_pts_2d.shape[0] == 0:
        return vis_img

    if len(depths_m) > 0:
        max_d = np.percentile(depths_m, 95)
        min_d = np.min(depths_m)
        if max_d <= min_d: max_d = min_d + 1.0
        norm_depths = (depths_m - min_d) / (max_d - min_d)
        norm_depths = np.clip(norm_depths, 0, 1)
    else:
        norm_depths = []

    for i, (u_coord, v_coord) in enumerate(lidar_pts_2d):
        u_coord_int, v_coord_int = int(u_coord), int(v_coord)
        if 0 <= u_coord_int < img_w and 0 <= v_coord_int < img_h:
            color_intensity_norm = norm_depths[i] if len(norm_depths) > 0 else 0.5
            color_lookup_val = np.array([[color_intensity_norm * 255]], dtype=np.uint8)
            bgr_color = cv2.applyColorMap(color_lookup_val, cv2.COLORMAP_JET)[0][0]
            cv2.circle(vis_img, (u_coord_int, v_coord_int), 2,
                       (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])), -1)
    return vis_img

def create_bev(lidar_pts, image_size, scale, point_radius):
    bev_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    min_z = -2.6
    max_z = 1.0
    height_range = max_z - min_z

    height_mask = (lidar_pts[:, 2] > min_z) & (lidar_pts[:, 2] < max_z)
    lidar_pts_filtered = lidar_pts[height_mask]

    if lidar_pts_filtered.shape[0] == 0:
        center_x = image_size // 2
        center_y = int(image_size * 0.8)
        ego_w, ego_h = int(2.0 * scale), int(4.0 * scale)
        ego_x, ego_y = center_x - ego_w // 2, center_y - ego_h // 2
        cv2.rectangle(bev_image, (ego_x, ego_y), (ego_x + ego_w, ego_y + ego_h), (0, 0, 255), -1)
        return bev_image

    x_coords = lidar_pts_filtered[:, 0]
    y_coords = lidar_pts_filtered[:, 1]
    z_coords = lidar_pts_filtered[:, 2]

    center_x = image_size // 2
    center_y = int(image_size * 0.8)

    px = (center_x + y_coords * scale).astype(int)
    py = (center_y - x_coords * scale).astype(int)

    bounds_mask = (px >= 0) & (px < image_size) & (py >= 0) & (py < image_size)

    px_final = px[bounds_mask]
    py_final = py[bounds_mask]
    z_final = z_coords[bounds_mask]

    norm_z = np.clip((z_final - min_z) / height_range, 0, 1)
    color_map_input = (norm_z * 255).astype(np.uint8)
    colors = cv2.applyColorMap(color_map_input, cv2.COLORMAP_JET).squeeze()

    if colors.ndim == 1:
        colors = np.array([colors])

    if colors.shape[0] == py_final.shape[0]:
        for i in range(len(px_final)):
            color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            cv2.circle(bev_image, (px_final[i], py_final[i]), point_radius, color, -1)
    else:
        print(f"Warning: Color ({colors.shape}) and Point ({py_final.shape}) mismatch!")

    ego_w, ego_h = int(2.0 * scale), int(4.0 * scale)
    ego_x, ego_y = center_x - ego_w // 2, center_y - ego_h // 2
    cv2.rectangle(bev_image, (ego_x, ego_y), (ego_x + ego_w, ego_y + ego_h), (0, 0, 255), -1)

    return bev_image

# --- CARLA Sensor Callbacks ---
def process_rgb_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    image_queue.put(array)

def process_lidar_scan(scan):
    points = np.frombuffer(scan.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_queue.put(points)

# --- Main Logic ---
def main():
    global K_CAM, BEV_SCALE
    K_CAM = get_cam_intrinsics(CAM_W, CAM_H, CAM_ANGLE_FOV)
    BEV_SCALE = BEV_IMAGE_SIZE / BEV_RANGE_M
    print(f"Camera Intrinsics (K_CAM):\n{K_CAM}\n")
    print(f"BEV Scale: {BEV_SCALE:.2f} pixels/meter")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(OUTPUT_DIR, VIDEO_FILENAME)
    video_fps = int(1 / SENSOR_TICK)
    max_frames = int(RECORD_SECONDS * video_fps)

    client = None
    world = None
    vehicle = None
    rgb_camera = None
    lidar_sensor = None
    video_writer = None
    original_settings = None
    traffic_manager = None

    try:
        client = carla.Client(HOST, PORT)
        client.set_timeout(10.0)
        world = client.load_world(MAP_NAME)
        original_settings = world.get_settings()
        print(f"Connected to CARLA on {HOST}:{PORT}, loaded map {MAP_NAME}.")

        traffic_manager = client.get_trafficmanager(TM_PORT)
        traffic_manager.set_synchronous_mode(True)
        print(f"Traffic Manager obtained on port {TM_PORT} and set to sync mode.")

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SENSOR_TICK
        world.apply_settings(settings)
        print(f"CARLA world set to synchronous mode with tick rate {SENSOR_TICK}s.")

        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Could not find any spawn points! Exiting.")
            return
        spawn_point = random.choice(spawn_points)
        print(f"Chose random spawn point: {spawn_point.location}")

        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned vehicle (ID: {vehicle.id}).")

        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.ignore_signs_percentage(vehicle, 100)
        print("Vehicle set to autopilot via Traffic Manager.")

        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(CAM_W))
        rgb_bp.set_attribute('image_size_y', str(CAM_H))
        rgb_bp.set_attribute('fov', str(CAM_ANGLE_FOV))
        rgb_bp.set_attribute('sensor_tick', str(SENSOR_TICK))
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        rgb_camera = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
        rgb_camera.listen(process_rgb_image)
        print(f"Spawned HD RGB camera (ID: {rgb_camera.id}).")

        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(LIDAR_RANGE))
        lidar_bp.set_attribute('points_per_second', str(LIDAR_POINTS_PER_SEC))
        lidar_bp.set_attribute('rotation_frequency', str(LIDAR_ROT_FREQ))
        lidar_bp.set_attribute('sensor_tick', str(SENSOR_TICK))
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        lidar_sensor.listen(process_lidar_scan)
        print(f"Spawned LiDAR sensor (ID: {lidar_sensor.id}).")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, video_fps, (VIDEO_W, VIDEO_H))
        print(f"Video will be saved to {video_path} ({max_frames} frames) with size {VIDEO_W}x{VIDEO_H}.")

        print("Waiting for sensors to stabilize...")
        world.tick()
        time.sleep(1.0)
        while not image_queue.empty(): image_queue.get()
        while not lidar_queue.empty(): lidar_queue.get()
        print("Starting recording loop...")

        frame_count = 0
        while frame_count < max_frames:
            world.tick()

            try:
                current_image = image_queue.get(timeout=1.0)
                current_lidar = lidar_queue.get(timeout=1.0)

                T_cam_world = carla_transform_to_np(rgb_camera.get_transform())
                T_lidar_world = carla_transform_to_np(lidar_sensor.get_transform())

                lidar_pts_raw = current_lidar[:, :3]
                lidar_pts_homo = np.hstack((lidar_pts_raw, np.ones((lidar_pts_raw.shape[0], 1))))

                T_world_cam = np.linalg.inv(T_cam_world)
                T_lidar_cam = T_world_cam @ T_lidar_world

                pts_cam_frame_homo = (T_lidar_cam @ lidar_pts_homo.T).T
                pts_cam_frame = pts_cam_frame_homo[:, :3]

                front_mask = pts_cam_frame[:, 0] > 0.1
                pts_cam_fwd = pts_cam_frame[front_mask]

                proj_pts_2d = None
                depths = []

                if pts_cam_fwd.shape[0] > 0:
                    pts_for_proj = np.zeros_like(pts_cam_fwd)
                    pts_for_proj[:, 0] = pts_cam_fwd[:, 1]
                    pts_for_proj[:, 1] = -pts_cam_fwd[:, 2]
                    pts_for_proj[:, 2] = pts_cam_fwd[:, 0]
                    proj_homo = (K_CAM @ pts_for_proj.T).T
                    valid_depth_mask = proj_homo[:, 2] > 1e-4
                    proj_homo = proj_homo[valid_depth_mask]
                    pts_for_proj = pts_for_proj[valid_depth_mask]
                    if proj_homo.shape[0] > 0:
                        proj_pts_2d = proj_homo[:, :2] / proj_homo[:, 2, np.newaxis]
                        depths = pts_for_proj[:, 2]
                    else:
                        proj_pts_2d = np.array([])
                        depths = np.array([])

                rgb_view = current_image.copy()
                fused_view = project_pts_to_img(current_image.copy(), proj_pts_2d, depths)
                bev_view = create_bev(lidar_pts_raw, BEV_IMAGE_SIZE, BEV_SCALE, BEV_POINT_RADIUS)

                fused_resized = cv2.resize(fused_view, (FUSED_W, FUSED_H))
                rgb_resized = cv2.resize(rgb_view, (SMALL_W, SMALL_H))
                bev_resized = cv2.resize(bev_view, (SMALL_W, SMALL_H))

                cv2.putText(fused_resized, 'Fused View (Camera + LiDAR)', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                cv2.putText(rgb_resized, 'RGB', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # <<< FIX: Updated Text >>>
                cv2.putText(bev_resized, "Bird's Eye View", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                bottom_row = np.hstack((rgb_resized, bev_resized))
                dashboard_image = np.vstack((fused_resized, bottom_row))

                video_writer.write(dashboard_image)
                display_w = 1280
                display_h = int(1280 * (VIDEO_H / VIDEO_W))
                dashboard_display = cv2.resize(dashboard_image, (display_w, display_h))
                cv2.imshow('CARLA BEV Dashboard', dashboard_display)

                frame_count += 1
                print(f"\rProcessed frame {frame_count}/{max_frames}", end="")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n'q' pressed, stopping early.")
                    break

            except queue.Empty:
                print("\nWarning: Sensor data queue was empty. Skipping frame.")
                continue

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n--- Cleaning up ---")
        if video_writer:
            video_writer.release()
            print("Video writer released.")
        if client and world:
            if traffic_manager:
                traffic_manager.set_synchronous_mode(False)
                print("Traffic Manager sync mode disabled.")
            if original_settings:
                world.apply_settings(original_settings)
                print("Restored original world settings.")

            actors_to_destroy = []
            actor_list = world.get_actors()
            for actor in actor_list:
                if actor.type_id.startswith('sensor.') or actor.type_id.startswith('vehicle.'):
                    actors_to_destroy.append(actor)

            print(f"Destroying {len(actors_to_destroy)} actors...")
            client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors_to_destroy], True)
            time.sleep(0.5)
            print("Actors destroyed.")

        cv2.destroyAllWindows()
        print("Cleanup complete. ✅")

if __name__ == "__main__":
    main()