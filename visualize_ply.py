from argparse import ArgumentParser
import open3d as o3d

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--scan', type=int, required=True,
                        help='the scan to visualize')

    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(f"results/points/scan{args.scan}.ply")
    o3d.visualization.draw_geometries([pcd])