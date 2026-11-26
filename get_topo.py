"""
Topography for San Antonio (Chile) -> dfs2, optionally merged with mesh bathymetry.

Source: Copernicus GLO-30 DEM (30 m, free, no login), served from AWS open data.
Heights are metres above the EGM2008 geoid; the sea surface is set to 0 in the DEM.

Requirements:
    pip install mikeio rasterio pyproj numpy

Examples:
    # Topography only, 30 m grid in UTM 19S over the default San Antonio box
    python topo_to_dfs2.py topo_san_antonio.dfs2 --dx 30

    # Custom box (lon_min lat_min lon_max lat_max)
    python topo_to_dfs2.py topo.dfs2 --dx 50 --bbox -71.75 -33.70 -71.50 -33.50

    # Topography + bathymetry: grid over the mesh extent, sea from mesh, land from DEM
    python topo_to_dfs2.py topobathy.dfs2 --dx 30 --mesh san_antonio.mesh

    # XYZ output (x y z columns, UTM 19S) instead of dfs2, land points only
    python topo_to_dfs2.py topo_san_antonio.xyz --dx 30 --land-only

    # Use a DEM you already have (any GeoTIFF) instead of downloading
    python topo_to_dfs2.py topo.dfs2 --dx 30 --dem my_dem.tif
"""
import argparse
import importlib.util
import math
import os
import sys
import urllib.request


def _set_geo_data_paths():
    """Tell GDAL/PROJ where their data files are when the conda environment
    is not activated (e.g. running from VS Code or Spyder). Fixes:
      - Warning: Cannot find gdalvrt.xsd (GDAL_DATA is not defined)
      - ERROR: PROJ: proj_create_from_database: Cannot find proj.db
    Must run before rasterio is imported."""
    spec = importlib.util.find_spec("rasterio")
    pkg = os.path.dirname(spec.origin) if spec and spec.origin else ""
    prefixes = [sys.prefix, os.environ.get("CONDA_PREFIX", "")]

    gdal_dirs = [os.path.join(pkg, "gdal_data")]
    proj_dirs = [os.path.join(pkg, "proj_data")]
    for pre in filter(None, prefixes):
        gdal_dirs += [os.path.join(pre, "Library", "share", "gdal"), os.path.join(pre, "share", "gdal")]
        proj_dirs += [os.path.join(pre, "Library", "share", "proj"), os.path.join(pre, "share", "proj")]

    if not os.path.isfile(os.path.join(os.environ.get("GDAL_DATA", ""), "gdalvrt.xsd")):
        for d in gdal_dirs:
            if os.path.isfile(os.path.join(d, "gdalvrt.xsd")):
                os.environ["GDAL_DATA"] = d
                break

    proj_env = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB") or ""
    if not os.path.isfile(os.path.join(proj_env, "proj.db")):
        for d in proj_dirs:
            if os.path.isfile(os.path.join(d, "proj.db")):
                os.environ["PROJ_DATA"] = d
                os.environ["PROJ_LIB"] = d  # older PROJ versions
                break


_set_geo_data_paths()

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.transform import from_origin
from pyproj import CRS

import mikeio

# Port of San Antonio and surroundings (lon_min, lat_min, lon_max, lat_max)
DEFAULT_BBOX = (-71.72, -33.68, -71.52, -33.50)
COP_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


def download_copernicus(bbox, folder="dem_tiles"):
    """Download the 1x1 degree Copernicus GLO-30 tiles covering bbox (lon/lat)."""
    os.makedirs(folder, exist_ok=True)
    lon0, lat0, lon1, lat1 = bbox
    files = []
    for lat in range(math.floor(lat0), math.floor(lat1) + 1):
        for lon in range(math.floor(lon0), math.floor(lon1) + 1):
            # Tile name uses the SW corner of the tile
            url = COP_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                 ew="E" if lon >= 0 else "W", lon=abs(lon))
            fn = os.path.join(folder, os.path.basename(url))
            if not os.path.exists(fn):
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, fn)
            files.append(fn)
    return files


def target_grid(args, crs):
    """Return (x0, y0, nx, ny, dx, dy) for cell centres in the target CRS."""
    dx = args.dx
    dy = args.dy or args.dx
    if args.mesh:
        g = mikeio.Mesh(args.mesh).geometry.get_overset_grid(dx=dx, dy=dy)
        return g.x[0], g.y[0], g.nx, g.ny, g.dx, g.dy
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", crs, *args.bbox)
    nx = int(np.ceil((xmax - xmin) / dx))
    ny = int(np.ceil((ymax - ymin) / dy))
    return xmin + dx / 2, ymin + dy / 2, nx, ny, dx, dy


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dfs2_file", nargs="?", default="topo_san_antonio.xyz",
                   help="Output file: .xyz for x y z text, anything else for dfs2 "
                        "(default: topo_san_antonio.xyz)")
    p.add_argument("--dx", type=float, default=30.0, help="Grid spacing in metres (default 30)")
    p.add_argument("--dy", type=float, default=None)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--epsg", type=int, default=32719,
                   help="Output projection (default 32719 = WGS84 / UTM 19S). "
                        "With --mesh this must match the mesh projection.")
    p.add_argument("--mesh", help="MIKE .mesh: use its extent and fill sea cells with its bathymetry")
    p.add_argument("--dem", help="Local DEM GeoTIFF to use instead of downloading Copernicus")
    p.add_argument("--land-only", action="store_true",
                   help="XYZ only: drop points with z <= 0 (the DEM's flat sea)")
    args = p.parse_args()

    crs = CRS.from_epsg(args.epsg)
    x0, y0, nx, ny, dx, dy = target_grid(args, crs)
    print(f"Grid: nx={nx}, ny={ny}, dx={dx}, dy={dy}, EPSG:{args.epsg}")

    # --- Read DEM (mosaic of tiles if needed) ---
    if args.dem:
        files = [args.dem]
    else:
        bbox = args.bbox
        if args.mesh:  # make sure the tiles cover the mesh extent
            bbox = transform_bounds(crs, "EPSG:4326", x0 - dx, y0 - dy,
                                    x0 + nx * dx, y0 + ny * dy)
        files = download_copernicus(bbox)
    srcs = [rasterio.open(f) for f in files]
    dem, dem_transform = merge(srcs)
    dem_crs = srcs[0].crs
    dem_nodata = srcs[0].nodata
    for s in srcs:
        s.close()

    # --- Reproject onto the target grid (rasterio rows run north -> south) ---
    dst = np.full((ny, nx), np.nan, dtype="float32")
    dst_transform = from_origin(x0 - dx / 2, y0 - dy / 2 + ny * dy, dx, dy)
    reproject(
        source=dem[0].astype("float32"), destination=dst,
        src_transform=dem_transform, src_crs=dem_crs, src_nodata=dem_nodata,
        dst_transform=dst_transform, dst_crs=crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    z = np.flipud(dst)  # MIKE grids: row 0 is the southern edge

    # --- Optionally merge bathymetry from the mesh ---
    if args.mesh:
        msh = mikeio.Mesh(args.mesh)
        geom = msh.geometry
        da = mikeio.DataArray(data=geom.element_coordinates[:, 2], geometry=geom,
                              item=mikeio.ItemInfo("z", mikeio.EUMType.Bathymetry))
        grid = mikeio.Grid2D(x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny,
                             projection=geom.projection)
        bathy = da.interp_like(grid, n_nearest=5).to_numpy().squeeze()
        inside = ~np.isnan(bathy)
        # Inside the mesh domain use the mesh; elsewhere use the DEM
        z = np.where(inside, bathy, z)
        print(f"Merged: {inside.sum()} cells from mesh, {(~inside).sum()} from DEM")
        projection = geom.projection
    else:
        projection = crs.to_wkt(version="WKT1_ESRI")

    # --- Write XYZ (x y z per line, cell centres, empty cells skipped) ---
    if args.dfs2_file.lower().endswith(".xyz"):
        X, Y = np.meshgrid(x0 + dx * np.arange(nx), y0 + dy * np.arange(ny))
        keep = ~np.isnan(z) & (z != 0)  # always drop empty cells and zero values
        if args.land_only:
            keep &= z > 0
        np.savetxt(args.dfs2_file, np.column_stack([X[keep], Y[keep], z[keep]]),
                   fmt="%.2f %.2f %.3f")
        print(f"Wrote {args.dfs2_file}  ({keep.sum()} points, "
              f"z range {np.nanmin(z[keep]):.1f} to {np.nanmax(z[keep]):.1f} m)")
        return

    # --- Write dfs2 ---
    grid = mikeio.Grid2D(x0=x0, y0=y0, dx=dx, dy=dy, nx=nx, ny=ny, projection=projection)
    da_out = mikeio.DataArray(
        data=z, geometry=grid,
        item=mikeio.ItemInfo("Bathymetry", mikeio.EUMType.Bathymetry, mikeio.EUMUnit.meter),
    )
    da_out.to_dfs(args.dfs2_file)
    print(f"Wrote {args.dfs2_file}  (z range {np.nanmin(z):.1f} to {np.nanmax(z):.1f} m)")


if __name__ == "__main__":
    main()