import os
import polars as pl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="res/sparse/model")
parser.add_argument("--dest", type=str, default="res/sparse/model")
if __name__ == "__main__":
    args = parser.parse_args()
    root = args.root
    dest = args.dest

    images_file = os.path.join(root, "images.txt")
    points_file = os.path.join(root, "points3D.txt")

    # ==> Format points3D.txt
    cols = ["POINT3D_ID", "X", "Y", "Z", "R", "G", "B", "ERROR", "TRACK[]_as_(IMAGE_ID,POINT2D_IDX)"]

    with open(points_file, "r") as f:
        for _ in range(3):
            f.readline() # skip header
        lines = f.readlines()

    def parse_line(line):
        parts = line.strip().split()

        point3d_id = int(parts[0])
        x, y, z = map(float, parts[1:4])
        r, g, b = map(int, parts[4:7])
        error = float(parts[7])
        
        # Remaining values are in (IMAGE_ID, POINT2D_IDX) pairs
        track = [(int(parts[i]), int(parts[i + 1])) for i in range(8, len(parts), 2)]
        
        return (point3d_id, x, y, z, r, g, b, error, track)

    # Parse all lines
    parsed_data = [parse_line(line) for line in lines]

    # Create Polars DataFrame
    df = pl.DataFrame(
        parsed_data,
        schema=cols
    )

    save_path = os.path.join(dest, "points3D.parquet")
    df.write_parquet(save_path)
    print(f"points3D successfully formated and saved to {save_path}")


    # ==> Format images.txt
    cols = "IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME", "POINTS2D[]_as_(X,Y,POINT3D_ID)"

    with open(images_file, "r") as f:
        for _ in range(4):
            f.readline()  # Skip header
        lines = f.readlines()

    # Ensure we process pairs of lines
    assert len(lines) % 2 == 0, "File format is incorrect: lines should be in pairs."

    def parse_pair(meta_line, points_line):
        meta_parts = meta_line.strip().split()
        points_parts = points_line.strip().split()

        # Extract metadata from the first line
        image_id = int(meta_parts[0])
        qw, qx, qy, qz = map(float, meta_parts[1:5])
        tx, ty, tz = map(float, meta_parts[5:8])
        camera_id = int(meta_parts[8])
        name = meta_parts[9]  # The filename (string)

        # Extract points_2D from the second line, ensuring proper typing
        points_2D = [
            (float(points_parts[i]), float(points_parts[i + 1]), float(points_parts[i + 2]))
            for i in range(0, len(points_parts), 3)
        ]

        return (image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name, points_2D)

    parsed_data = [parse_pair(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]

    df = pl.DataFrame(
        parsed_data,
        schema=cols
    )

    save_path = os.path.join(dest, "images.parquet")
    df.write_parquet(save_path)
    print(f"images file successfully formated and saved to {save_path}")