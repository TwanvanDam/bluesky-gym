"""Merge one or more Mollweide GeoTIFF tiles and reproject to a target CRS.

# Merge two tiles and reproject to EPSG:3035 at 1 km resolution
python -m scripts.merge_population_maps \\
    scripts/population_maps/GHS_POP_*_R3_C19.tif \\
    scripts/population_maps/GHS_POP_*_R3_C18.tif \\
    --output scripts/population_maps/europe_3035_1km.tif
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", metavar="FILE",
                   help="Input GeoTIFF file(s). Glob patterns are expanded.")
    p.add_argument("--output", "-o", required=True, metavar="FILE",
                   help="Output GeoTIFF path.")
    p.add_argument("--crs", default="EPSG:3035", metavar="CRS",
                   help="Output CRS (default: EPSG:3035).")
    p.add_argument("--resolution", type=float, default=None, metavar="METERS",
                   help="Output pixel size in metres. Defaults to source resolution.")
    p.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="Clip output to this bounding box in the OUTPUT CRS.")
    p.add_argument("--resampling", default="cubic_spline",
                   choices=[r.name for r in Resampling],
                   help="Resampling method (default: cubic_spline).")
    return p.parse_args()


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            paths.extend(Path(p) for p in matched)
        else:
            paths.append(Path(pattern))
    return paths


def main():
    args = parse_args()

    input_paths = expand_inputs(args.inputs)
    for p in input_paths:
        if not p.exists():
            print(f"ERROR: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Input files ({len(input_paths)}):")
    for p in input_paths:
        print(f"  {p}")

    # merge tiles in their native CRS
    datasets = [rasterio.open(p) for p in input_paths]

    src_crs = datasets[0].crs
    nodata = datasets[0].nodata
    src_dtype = datasets[0].dtypes[0]

    print(f"\nMerging {len(datasets)} tile(s) (source CRS: {src_crs}, nodata: {nodata}) ...")
    merged_data, merged_transform = merge(datasets, nodata=nodata)
    for ds in datasets:
        ds.close()

    src_height, src_width = merged_data.shape[1], merged_data.shape[2]
    print(f"  Merged shape: {src_width} x {src_height} px")

    # calculate output transform in the target CRS
    dst_crs = CRS.from_string(args.crs)
    resolution = args.resolution  # None → calculate_default_transform picks it

    transform_kwargs = dict(
        src_crs=src_crs,
        dst_crs=dst_crs,
        width=src_width,
        height=src_height,
        left=merged_transform.c,
        bottom=merged_transform.f + merged_transform.e * src_height,
        right=merged_transform.c + merged_transform.a * src_width,
        top=merged_transform.f,
    )
    if resolution is not None:
        transform_kwargs["resolution"] = resolution

    dst_transform, dst_width, dst_height = calculate_default_transform(**transform_kwargs)

    # Apply optional bounds clip
    if args.bounds:
        xmin, ymin, xmax, ymax = args.bounds
        from rasterio.transform import from_bounds
        dst_transform = from_bounds(xmin, ymin, xmax, ymax, dst_width, dst_height)
        # Recompute dimensions to match resolution
        if resolution is not None:
            dst_width = int((xmax - xmin) / resolution)
            dst_height = int((ymax - ymin) / resolution)
            dst_transform = from_bounds(xmin, ymin, xmax, ymax, dst_width, dst_height)

    print(f"\nReprojecting to {dst_crs} ...")
    print(f"  Output shape: {dst_width} x {dst_height} px")
    print(f"  Resampling:   {args.resampling}")

    # reproject
    out_nodata = -9999.0
    dst_data = np.full((1, dst_height, dst_width), nodata, dtype=src_dtype)

    reproject(
        source=merged_data,
        destination=dst_data,
        src_transform=merged_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=getattr(Resampling, args.resampling),
    )

    # set nodata values
    nodata_mask = dst_data[0] == nodata
    dst_data[0] = np.clip(dst_data[0], 0, None)
    dst_data[0][nodata_mask] = out_nodata

    # save as tiled, compressed GeoTIFF
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {output_path} ...")
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=dst_height,
        width=dst_width,
        count=1,
        dtype=src_dtype,
        crs=dst_crs,
        transform=dst_transform,
        nodata=out_nodata,
        compress="lzw",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(dst_data)

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. Output: {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
