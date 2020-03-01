from argparse import ArgumentParser
import open3d as o3d
import numpy as np

# from https://github.com/intel-isl/Open3D/blob/master/examples/Python/Advanced/load_save_viewpoint.py
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--scan', type=int, required=True,
                        help='the scan to visualize')

    parser.add_argument('--use_viewpoint', default=False, action='store_true',
                        help='use precalculated viewpoint')
    parser.add_argument('--save_viewpoint', default=False, action='store_true',
                        help='save this viewpoint')

    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(f"results/points/scan{args.scan}.ply")
    vis = o3d.visualization.Visualizer()

    vis.create_window()
    ctr = vis.get_view_control()

    opt = vis.get_render_option()
    opt.background_color = np.array([128/255, 128/255, 255/255])
    if args.use_viewpoint:
        param = o3d.io.read_pinhole_camera_parameters(f"results/viewpoint.json")
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.run()
    if args.save_viewpoint:
        vis.add_geometry(pcd)
        vis.run()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(f"results/viewpoint.json", param)
    vis.destroy_window()