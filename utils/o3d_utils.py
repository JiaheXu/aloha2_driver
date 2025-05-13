import cv2
import numpy as np
import open3d as o3d
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
import copy
from .math_utils import *
import trimesh

def get_depth(depth_image, mask):
    depth_mask = np.where(mask > 0, 1, 0)
    depth_image = depth_image * depth_mask
    return depth_image

def depth_to_pcl(depth_image, intrinsics):
    """
    Convert a depth image to a point cloud.
    """
    depth_image = depth_image / 1000
    h, w = depth_image.shape
    u0, v0 = intrinsics[0, 2], intrinsics[1, 2]
    fu, fv = intrinsics[0, 0], intrinsics[1, 1]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u_flat = u.flatten() - u0
    v_flat = v.flatten() - v0
    z_flat = depth_image.flatten()
    u_flat = u_flat[z_flat > 0]
    v_flat = v_flat[z_flat > 0]
    z_flat = z_flat[z_flat > 0]
    unporj_u = u_flat * z_flat / fu
    unporj_v = v_flat * z_flat / fv
    return np.vstack((unporj_u, unporj_v, z_flat)).T 

def write_ply(ply_path, points, colors= np.array([])):
        """Write mesh, point cloud, or oriented point cloud to ply file.
        Args:
            ply_path (str): Output ply path.
            points (float): Nx3 x,y,z locations for each point
            colors (uchar): Nx3 r,g,b color for each point
        We provide this function for you.
        """
        with open(ply_path, 'w') as f:
            # Write header.
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex {}\n'.format(len(points)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            if len(colors) != 0:
                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')
            f.write('end_header\n')
            # Write points.
            for i in range(len(points)):
                f.write('{0} {1} {2}'.format(
                    points[i][0],
                    points[i][1],
                    points[i][2]))
                if len(colors) != 0:
                    f.write(' {0} {1} {2}'.format(
                        int(colors[i][0]),
                        int(colors[i][1]),
                        int(colors[i][2])))

def mesh2points(mesh, num_points = 1000):
    """
    Convert a mesh to a list of points.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    points = points.view(np.ndarray)
    return points

def numpy_2_pcd(pcd_np, color = None ):

    # pcd_np = np.array(pcd_np)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    if(color is not None):
        color_np = np.zeros(pcd_np.shape)
        color_np[:,] = color
        pcd.colors = o3d.utility.Vector3dVector(color_np)
    return pcd

def pcd_2_numpy(pcd , scaling = False):

    pcd_np = np.asarray(pcd.points)
    if(scaling == True):
        pcd_np = np.concatenate( [pcd_np, np.ones( (pcd_np.shape[0],1) )], axis = 1)

    return pcd_np
    
def project_point( point_3d, color, image, extrinsic, intrinsic, radius = 5):
                
    point4d = np.append(point_3d, 1)
    new_point4d = np.matmul(extrinsic, point4d)
    point3d = new_point4d[:-1]
    zc = point3d[2]
    new_point3d = np.matmul(intrinsic, point3d)
    new_point3d = new_point3d/new_point3d[2]
    u = int(round(new_point3d[0]))
    v = int(round(new_point3d[1]))
    if(v<0 or v>= image.shape[0] or u<0 or u>= image.shape[1]):
        return image
    
    image[max(0, v-radius): min(v+radius, image.shape[0]), max(0, u-radius): min(u+radius, image.shape[1]) ] = color
    # print("updated")
    return image

def create_colored_point_cloud(color, depth, cam_intrinsic_np, far=1.0, near=0.1, num_points=10000, use_grid_sampling = True):
    assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])

    # Create meshgrid for pixel coordinates
    xmap = np.arange(color.shape[1])
    ymap = np.arange(color.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    # Calculate 3D coordinates
    points_z = depth / 1000.0
    # print("points_z: ", np.max(points_z), " ", np.min(points_z))
    cx = cam_intrinsic_np[0][2]
    cy = cam_intrinsic_np[2][2]    
    fx = cam_intrinsic_np[0][0]
    fy = cam_intrinsic_np[1][1]  

    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    cloud = cloud.reshape([-1, 3])
    
    # Clip points based on depth
    mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
    cloud = cloud[mask]
    color = color.reshape([-1, 3])
    color = color[mask]


    colored_cloud = np.hstack([cloud, color.astype(np.float32)])
    if use_grid_sampling:
        colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
    
    if num_points > colored_cloud.shape[0]:
        num_pad = num_points - colored_cloud.shape[0]
        pad_points = np.zeros((num_pad, 6))
        colored_cloud = np.concatenate([colored_cloud, pad_points], axis=0)
    else: 
        # Randomly sample points
        selected_idx = np.random.choice(colored_cloud.shape[0], num_points, replace=True)
        colored_cloud = colored_cloud[selected_idx]
    
    # shuffle
    np.random.shuffle(colored_cloud)
    return colored_cloud
    
def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]


def cropping(rgb, xyz, depth, bound_box, label = None, return_image = True):

    x = xyz[:,:,0]
    y = xyz[:,:,1]
    z = xyz[:,:,2]

    # print("bound_box: ", bound_box)
    valid_idx = np.where( (x>=bound_box[0][0]) & (x <=bound_box[0][1]) & (y>=bound_box[1][0]) & (y<=bound_box[1][1]) & (z>=bound_box[2][0]) & (z<=bound_box[2][1]) )

    if(return_image):

        cropped_rgb = np.zeros(rgb.shape)
        cropped_xyz = np.zeros(xyz.shape) 
        cropped_depth = np.zeros(depth.shape) 
        cropped_rgb[valid_idx] = rgb[valid_idx]
        cropped_xyz[valid_idx] = xyz[valid_idx]
        cropped_depth[valid_idx] = depth[valid_idx]
        return cropped_rgb, cropped_xyz, cropped_depth

    valid_xyz = xyz[valid_idx]
    valid_rgb = rgb[valid_idx]
    valid_label = None
    if(label is not None):
        valid_label = label[valid_idx]
            
    valid_pcd = o3d.geometry.PointCloud()
    valid_pcd.points = o3d.utility.Vector3dVector( valid_xyz)
    if(np.max(valid_rgb) > 1.0):
        valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb/255.0 )
    else:
        valid_pcd.colors = o3d.utility.Vector3dVector( valid_rgb )
    return valid_xyz, valid_rgb, valid_label, valid_pcd

def xyz_rgb_validation(rgb, xyz):
    # verify xyz and depth value
    valid_pcd = o3d.geometry.PointCloud()
    xyz = xyz.reshape(-1,3)
    rgb = (rgb/255.0).reshape(-1,3)
    valid_pcd.points = o3d.utility.Vector3dVector( xyz )
    valid_pcd.colors = o3d.utility.Vector3dVector( rgb )
    # visualize_pcd(valid_pcd)

def visualize_pcd_transform(pcd, left = None, right = None):
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    
    # left_mesh.scale(0.1, center=(left[0][3], left[1][3], left[2][3]))
    # right_mesh.scale(0.1, center=(right[0][3], right[1][3], right[2][3]))
    
    if left is not None:
        for trans in left:
            left_mesh = copy.deepcopy(mesh).transform(trans)
            vis.add_geometry(left_mesh)

    if right is not None:
        for trans in right:
            right_mesh = copy.deepcopy(mesh).transform(trans)
            vis.add_geometry(right_mesh)
    
    # view_ctl.set_up([-0.4, 0.0, 1.0])
    # view_ctl.set_front([-4.02516493e-01, 3.62146675e-01, 8.40731978e-01])
    # view_ctl.set_lookat([0.0 ,0.0 ,0.0])
    
    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    # view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()
    
def visualize_pcd(pcd, traj_lists = None, curr_pose = None, drawlines = False):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    # if(use_arrow):
    #     mesh = o3d.geometry.TriangleMesh.create_arrow( cylinder_radius=0.01, cone_radius=0.01, cylinder_height=0.005, cone_height=0.01, resolution=20, cylinder_split=4, cone_split=1 )
    # print("curr_pose: ", curr_pose)
    if(traj_lists is not None):

        for traj_idx ,traj in enumerate( traj_lists, 0 ):
            points = [ [0,0,0] ]
            lines = []
            colors = []
            if(curr_pose is not None):
                points.append( curr_pose[traj_idx][0:3,3] )
                lines.append( [ len(points) - 1 , len(points) - 2])
                colors.append( [1,0,0] )
            for node_idx, point in enumerate( traj , 0 ):
                new_mesh = copy.deepcopy(mesh).transform(point)
                vis.add_geometry(new_mesh)
                if drawlines:
                    points.append(point[0:3,3])
                    lines.append( [ len(points) - 1 , len(points) - 2])
                    colors.append( [1,0,0] )


            
            if drawlines:
                # lines.append( [ 0, len(points)])
                # colors.append( [1,0,0] )
                # print("points: ", len(points))
                # print("lines: ", lines)
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(points),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)
                # o3d.visualization.draw_geometries([line_set])

    if(curr_pose is not None):
        for pose in curr_pose:
            curr_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            curr_mesh.scale(0.05, center=(0., 0., 0.) )
            curr_mesh = curr_mesh.transform(pose)
            vis.add_geometry(curr_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()
def visualize_pcd_delta_transform(pcd, start_ts = [], delta_transforms=[]):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.2, center=(0., 0., 0.) )
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.))

    new_mesh = copy.deepcopy(mesh).transform( get_transform(start_t[0:3], start_t[3:7]) )
    vis.add_geometry(new_mesh)
    for start_t, delta_transform in zip( start_ts, delta_transforms):
        init_transform = get_transform( start_t )
        for delta_t in delta_transform:
            last_trans = get_transform( delta_t ) @ init_transform
            new_mesh = copy.deepcopy(mesh).transform(last_trans)
            vis.add_geometry(new_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()

# def visualize_pcds(pcds):

#     coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window()
#     coor_frame.scale(0.2, center=(0., 0., 0.) )
#     vis.add_geometry(coor_frame)
#     vis.get_render_option().background_color = np.asarray([255, 255, 255])

#     view_ctl = vis.get_view_control()

    

#     # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#     # mesh.scale(0.1, center=(0., 0., 0.))

#     # new_mesh = copy.deepcopy(mesh).transform( get_transform(start_t[0:3], start_t[3:7]) )
#     # vis.add_geometry(new_mesh)
#     for pcd in pcds:
#         vis.add_geometry(pcd)

#     view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
#     view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
#     view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
#     vis.run()
#     vis.destroy_window()



def visualize_pcds(pcds, curr_pose = None, drawlines = False):

    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    coor_frame.scale(0.1, center=(0., 0., 0.))
    vis.add_geometry(coor_frame)
    vis.get_render_option().background_color = np.asarray([255, 255, 255])

    view_ctl = vis.get_view_control()

    for pcd in pcds:
        if(pcd is not None):
            vis.add_geometry(pcd)

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(0.1, center=(0., 0., 0.) )
    if(curr_pose is not None):
        for pose in curr_pose:
            curr_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            curr_mesh.scale(0.05, center=(0., 0., 0.) )
            curr_mesh = curr_mesh.transform(pose)
            vis.add_geometry(curr_mesh)

    view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_front((-0.3, 0.0, 0.2))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0.0, 0.0, 0.3))  # set the original point as the center point of the window
    vis.run()
    vis.destroy_window()
