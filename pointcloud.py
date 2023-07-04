import flow_IO
import flow_utils
import numpy as np
import open3d as o3d
from PIL import Image
import os


SPRING_BASELINE = 0.065


def get_depth(disp1, intrinsics, baseline=SPRING_BASELINE):
    """
    get depth from reference frame disparity and camera intrinsics
    """
    return intrinsics[0] * baseline / disp1


def pointcloud_visualization(disp1, img, intrinsics):
    """
    visualize Spring data as point cloud
    """

    disp1 = disp1[::2,::2]

    depth = get_depth(disp1, intrinsics)

    points3D = flow_utils.inv_project(depth, intrinsics)

    points3D = points3D.reshape((-1,3))
    img = img.reshape((-1,3))

    valid = ~np.isnan(points3D[:,0]) & ~np.isnan(points3D[:,1]) & ~np.isnan(points3D[:,2]) & ~np.isinf(points3D[:,0]) & ~np.isinf(points3D[:,1]) & ~np.isinf(points3D[:,2])
    points3D = points3D[valid]
    img = img[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    pcd.colors = o3d.utility.Vector3dVector(img)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Spring', width=1920, height=1080)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    opt.point_size = 2

    # set default camera
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    int_matrix = camera_params.intrinsic.intrinsic_matrix.copy()
    int_matrix[0,0] = intrinsics[0]
    int_matrix[1,1] = intrinsics[1]
    camera_params.intrinsic.intrinsic_matrix = int_matrix
    camera_params.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":

    seq = "0004"
    frame = 44

    spring_path = os.getenv("SPRING_DIR", "/data/spring")
    print("looking for Spring dataset in", spring_path)

    disp1_path = os.path.join(spring_path, "train", seq, "disp1_left", f"disp1_left_{frame:04d}.dsp5")
    intrinsics_path = os.path.join(spring_path, "train", seq, "cam_data", "intrinsics.txt")
    img_path = os.path.join(spring_path, "train", seq, "frame_left", f"frame_left_{frame:04d}.png")

    disp1 = flow_IO.readDispFile(disp1_path)
    intrinsics = np.loadtxt(intrinsics_path)[frame - 1]
    img = np.asarray(Image.open(img_path)) / 255.0

    pointcloud_visualization(disp1, img, intrinsics)
