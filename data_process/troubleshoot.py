import xarray as xr

# The ARCO-ERA5 Zarr store URL
url = "gs://weatherbench2/datasets/era5/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
#weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr" 
# #weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr" 
# #weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"

print(f"Connecting to: {url} ...")

try:
    # Open dataset lazily (chunks={} prevents downloading data)
    ds = xr.open_dataset(url, engine="zarr", chunks={}, storage_options={'token': 'anon'})

    print(ds)
    
    print("\n--- Available Variables ---")
    for var_name in sorted(ds.data_vars):
        print(f"  {var_name}")

    print("\n--- Coordinates ---")
    for coord_name in sorted(ds.coords):
        print(f"  {coord_name}")

except Exception as e:
    print(f"\nError accessing dataset: {e}")