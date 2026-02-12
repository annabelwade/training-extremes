from cmocean import cm
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cmocean import cm
import os
import re
import numpy as np
from matplotlib import cm as mpl_cm
import math
from datetime import datetime, timedelta

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter

import warnings, sys
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def crop_spatial_bounds(ds, bounding_box):
    """Helper to crop dataset to bounding box."""
    ds_cropped = ds.where(
        (ds['lat'] >= bounding_box['latitude_min']) & (ds['lat'] <= bounding_box['latitude_max']) &
        (ds['lon'] >= bounding_box['longitude_min']) & (ds['lon'] <= bounding_box['longitude_max']),
        drop=True
    )
    return ds_cropped

def calculate_mse(forecast, truth, latitude_weighting=True):
    """Helper to calculate MSE or Latitude-Weighted MSE."""
    if latitude_weighting:
        squared_diff = (forecast - truth) ** 2
        weights = np.cos(np.deg2rad(forecast.lat))
        weights /= weights.mean() # normalize weights by their mean
        return float((squared_diff*weights).mean())
    else:
        squared_diff = (forecast - truth) ** 2
        return float(squared_diff.mean())

def setup_figure_layout(num_plots):
    """Helper to handle the rows/columns math and figure initialization."""
    if num_plots > 12:
        print('plotting case 1')
        rows = num_plots // 6 - 1
        cols = num_plots // rows + (num_plots % rows > 0)
        print('num rows:', rows, 'num cols:', cols, 'num plots:', num_plots)
        fig_height = rows * 3
    elif num_plots > 9:
        print('plotting case 2')
        rows = 2
        cols = math.ceil(num_plots / 2)
        fig_height = 6 
    else:
        print('plotting case 3')
        rows = 1
        cols = num_plots
        fig_height = 6 if num_plots <= 4 else 3
    
    fig, ax = plt.subplots(
        rows, cols, figsize=(20, fig_height), 
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    
    if num_plots > 1:
        ax_list = ax.flatten()
    else:
        ax_list = [ax]
        
    return fig, ax_list, ax

def get_forecast_and_truth_data(experiment_number, ckpt_num, leadtime, var, bounding_box, 
            valid_timestep_str="2022_12_27T00"):
    """Helper to load forecast and truth data for a given checkpoint."""
    valid_datetime = datetime.fromisoformat(valid_timestep_str)
    start_datetime = valid_datetime - timedelta(days=leadtime) #datetime.fromisoformat(init_timestep_str)
    inference_name = start_datetime.strftime("%Y_%m_%dT%H")+'_nsteps'+str(leadtime*4)

    if experiment_number == 1:
        valid_time_ind=2  # will be retiring this experiment soon and won't need this check
    else:
        valid_time_ind=0 

    output_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/Experiment{experiment_number}/{valid_timestep_str.replace('-', '_')[:10]}/Checkpoint{ckpt_num}_{inference_name}.nc"
    ds = xr.open_dataset(output_fp)
    ds_cropped = crop_spatial_bounds(ds, bounding_box)
    forecast_data = ds_cropped[var].isel(valid_time=valid_time_ind)

    # if var is IVT, use a different file path for truth
    # TODO

    truth_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/init_files/Initialize_{inference_name}.nc"
    truth_ds = xr.open_dataset(truth_fp)
    truth_ds = truth_ds.isel(time=1).sel(variable=var)
    truth_ds_cropped = crop_spatial_bounds(truth_ds, bounding_box)
    truth_ds_cropped = truth_ds_cropped.to_dataarray().squeeze()

    print(f'LOADED VALIDTIME {valid_timestep_str} VAR {var} LEADTIME {leadtime} ckpt {ckpt_num} FORECAST:', output_fp)

    return forecast_data, truth_ds_cropped

def get_largest_path_from_contour(cs, one_contour_level=True):
    """
    Takes in cs, a QuadContourSet (output of ax.contour - contour line), finds the largest 
    closed polygon by area, and returns its (x, y) coordinates.
    """
    max_area = 0
    best_poly = None
    
    # cs.allsegs is a list of levels -> list of polygons -> (N, 2) arrays
        # level0segs = [polygon0, polygon1, ...]
        # polygon0 = [[x0, y0], [x1, y1], ...]
    # Since we only contour one level ask for one level, we look at index 0
    if one_contour_level:
        if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
            for poly in cs.allsegs[0]:
                # Shoelace formula for area
                x, y = poly[:, 0], poly[:, 1] # the x,y values are lat,lon values

                # use shapely to calculate area
                area = Polygon(zip(x, y)).area
                if area > max_area:
                    max_area = area
                    best_poly = poly
    else:
        # error handling for multiple contour levels not implemented
        raise NotImplementedError("Multiple contour levels not supported in get_largest_path_from_contour.")
                
    if best_poly is not None:
        return best_poly[:, 0], best_poly[:, 1]
    return None, None

def get_IoU_of_contours(model_x, model_y, truth_x, truth_y, ):
    """Helper to calculate Intersection over Union (IoU) of two contours."""

    model_poly = Polygon(zip(model_x, model_y))
    truth_poly = Polygon(zip(truth_x, truth_y))

    if not model_poly.is_valid or not truth_poly.is_valid:
        print("Invalid polygon for IoU calculation.")
        return 0.0

    intersection = model_poly.intersection(truth_poly).area
    union = model_poly.union(truth_poly).area

    if union == 0:
        print("Union area is zero for IoU calculation.")
        return 0.0

    iou = intersection / union

    # #### DEBUGGING
    # # CONFIRMED THAT SHOELACE FORMULA AND SHAPELY AREA ARE GIVING THE SAME AREA
    # x,y = model_x, model_y
    # area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    # print(f'Model Area Shoelace: {area}, Shapely: {model_poly.area}')
    # print(iou)
    # # Make two intermediate plots to debug intersection and union areas 
    # fig_iou, ax_iou = plt.subplots(1,2, figsize=(10,5))
    # ax_iou[0].set_title('Intersection Area')
    # x_i, y_i = model_poly.intersection(truth_poly).exterior.xy
    # ax_iou[0].fill(x_i, y_i, alpha=0.5, fc='blue', ec='black')
    # ax_iou[0].text(0.5, 0.9, f'Area: {intersection:.2f}', transform=ax_iou[0].transAxes, )
    # ax_iou[1].set_title('Union Area')
    # x_u, y_u = model_poly.union(truth_poly).exterior.xy
    # ax_iou[1].fill(x_u, y_u, alpha=0.5, fc='green', ec='black')
    # ax_iou[1].text(0.5, 0.9, f'Area: {union:.2f}', transform=ax_iou[1].transAxes)

    # plt.show()
    # ####
    
    return iou

def get_amplitude_of_contour(data_array, contour_x, contour_y, latitude_weighting):
    """
    Helper to sum all the values of that variable within the contour.
    data_array: xarray DataArray with 'lat' and 'lon' dimensions
    contour_x: x-coordinates (lon) of the contour polygon's vertices
    contour_y: y-coordinates (lat) of the contour polygon's vertices
    latitude_weighting: whether to apply latitude weighting before summing

    In the case of 'tcwv', we are calling this metric "volume"
    o/w this is "amplitude".
    """

    contour_poly = Polygon(zip(contour_x, contour_y))
    if not contour_poly.is_valid:
        print("Invalid polygon for amplitude calculation.")
        return 0.0

    total_amplitude = 0.0
    if latitude_weighting:
        weights = np.cos(np.deg2rad(data_array['lat'].values))
        weights /= weights.mean()  # normalize weights by their mean
    for lat_i, lat in enumerate(data_array['lat'].values):
        for lon in data_array['lon'].values:
            point = Point(lon, lat)
            if contour_poly.contains(point):
                if latitude_weighting:
                    weight = weights[lat_i]
                    value = data_array.sel(lat=lat, lon=lon).values * weight
                else:
                    value = data_array.sel(lat=lat, lon=lon).values
                total_amplitude += value

    return float(total_amplitude)

def get_intensity_of_contour(data_array, contour_x, contour_y, latitude_weighting):
    """
    Helper to calculate the average intensity (mean value) of the variable within the contour.
    First, utilizes the get_amplitude_of_contour function to get the total amplitude,
    then divides by the number of grid points within the contour to get the mean intensity.

    data_array: xarray DataArray with 'lat' and 'lon' dimensions
    contour_x: x-coordinates (lon) of the contour polygon's vertices
    contour_y: y-coordinates (lat) of the contour polygon's vertices
    """
    contour_poly = Polygon(zip(contour_x, contour_y))
    if not contour_poly.is_valid:
        print("Invalid polygon for intensity calculation.")
        return 0.0

    total_amplitude = get_amplitude_of_contour(data_array, contour_x, contour_y, latitude_weighting)

    # Count number of grid points within the contour
    point_count = 0
    for lat in data_array['lat'].values:
        for lon in data_array['lon'].values:
            point = Point(lon, lat)
            if contour_poly.contains(point):
                point_count += 1

    if point_count == 0:
        print("No grid points found within the contour for intensity calculation.")
        return 0.0

    average_intensity = total_amplitude / point_count
    return float(average_intensity)

def render_panel(ax, background_data, title, cmap, vmin, vmax, extent, var, ckpts=[], ckpt_num=None,
                  contour_val=None, contour_data=None, truth_contour_data=None, is_truth=False,
                  mse_value=None, contour_metrics=True, poly_ax=None):
    """Helper to plot a single panel (contourf + optional contours + formatting)."""

    # Define metrics to include in boxes   
    metrics_to_incl = ['mse', 'iou', 'err_amplitude', 'err_volume', 'err_intensity'] # for controlling which metrics to visualize
    
    # 1. Main Background Plot
    ax.contourf(
        background_data['lon'], background_data['lat'], background_data,
        transform=ccrs.PlateCarree(), 
        cmap=cmap, vmin=vmin, vmax=vmax, levels=25
    )

    # Initialize metric containers
    mx, my, tx, ty = None, None, None, None
    metrics_list = [] # List of dicts: {'key': 'mse', 'val': 123, 'str': 'MSE: 123'}
    
    # 2. Red Contour (Feature Tracking)
    if contour_val is not None and is_truth is False: # only plot red contour for model
        c_data = contour_data if contour_data is not None else background_data
        cs = ax.contour(
            c_data['lon'], c_data['lat'], c_data,
            levels=[contour_val], colors='red', transform=ccrs.PlateCarree(), label='model',
        )

        if contour_metrics:
            # Use invisible contourf (filled) to capture the boundary-aware geometry
            # This ensures the polygon follows the box edges instead of cutting corners
            cs_fill = ax.contourf(
                c_data['lon'], c_data['lat'], c_data,
                levels=[contour_val, np.inf], alpha=0, transform=ccrs.PlateCarree()
            )
            mx, my = get_largest_path_from_contour(cs_fill)

    # 3. Black Contour (Truth Overlay - optional)
    if truth_contour_data is not None and contour_val is not None:
        cs_truth = ax.contour(
            truth_contour_data['lon'], truth_contour_data['lat'], truth_contour_data, linestyles='dashdot',
            levels=[contour_val], colors='black', transform=ccrs.PlateCarree(), alpha=0.8, label='truth',
        )

        if contour_metrics:
            cs_truth_fill = ax.contourf(
                truth_contour_data['lon'], truth_contour_data['lat'], truth_contour_data,
                levels=[contour_val, np.inf], alpha=0, transform=ccrs.PlateCarree()
            )
            tx, ty = get_largest_path_from_contour(cs_truth_fill)

    # 4. Calculate Metrics
    if contour_metrics and contour_val is not None:
        
        # A. Truth Panel Metrics (Amplitude only)
        if is_truth:
            if tx is not None:
                # Amplitude
                amp_truth = get_amplitude_of_contour(background_data, tx, ty, latitude_weighting=True)
                if var == 'tcwv':
                    metrics_list.append({'key': 'amp', 'val': amp_truth, 'str': f"vol: {amp_truth:.1f}"})
                    print(f"Truth total volume within contour: {amp_truth:.1f}")
                else:
                    metrics_list.append({'key': 'amp', 'val': amp_truth, 'str': f"amp: {amp_truth:.1f}"})
                    print(f"Truth total amplitude within contour: {amp_truth:.1f}")

                # Intensity
                intensity_truth = get_intensity_of_contour(background_data, tx, ty, latitude_weighting=True)
                metrics_list.append({'key': 'intensity', 'val': intensity_truth, 'str': f"int: {intensity_truth:.2f}"})
                print(f"Truth average intensity within contour: {intensity_truth:.2f}")
                
                # --- Plot Polygon on second figure for Truth ---
                if poly_ax is not None:
                    poly_ax.set_title(title, fontsize=10)
                    poly_ax.set_extent(extent, crs=ccrs.PlateCarree())
                    poly_ax.coastlines()
                    poly_ax.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
                    # Plot just the truth polygon in black/grey
                    poly_ax.fill(tx, ty, alpha=0.5, fc='gray', ec='black', transform=ccrs.PlateCarree())
        
        # B. Forecast Panel Metrics (IoU, MSE, Amp Diff)
        else:
            # MSE
            if mse_value is not None and 'mse' in metrics_to_incl:
                metrics_list.append({'key': 'mse', 'val': mse_value, 'str': f"MSE: {mse_value:.2f}"})
            
            # IoU, Amplitude Difference, Intensity Difference
            if mx is not None and tx is not None:
                if 'iou' in metrics_to_incl:
                    # IoU
                    iou = get_IoU_of_contours(mx, my, tx, ty)
                    metrics_list.append({'key': 'iou', 'val': iou, 'str': f"IoU: {iou:.2f}"})
                
                if 'err_amplitude' in metrics_to_incl or 'err_volume' in metrics_to_incl:
                    # Amplitude Difference (Model Amp - Truth Amp)
                    amp_model = get_amplitude_of_contour(c_data, mx, my, latitude_weighting=True)
                    if var == 'tcwv':
                        print(f"ckpt_num: {ckpt_num}, Model pred total volume: {amp_model:.1f}")
                    else:
                        print(f"ckpt_num: {ckpt_num}, Model pred total amplitude: {amp_model:.1f}")
                    
                    # # Amp raw (optional to display, usually good for reference)
                    # metrics_list.append({'key': 'amp_raw', 'val': amp_model, 'str': f"Amp: {amp_model:.1f}"})

                    amp_truth = get_amplitude_of_contour(truth_contour_data, tx, ty, latitude_weighting=True)
                    percent_error = ((amp_model - amp_truth) / amp_truth * 100) 
                    if var == 'tcwv':
                        print(f"ckpt_num: {ckpt_num}, Truth total volume: {amp_truth:.1f}, % err volume: {percent_error:.1f}%")
                        metrics_list.append({'key': 'err_volume', 'val': percent_error, 'str': f"% err vol: {percent_error:.1f}"})
                    else:
                        metrics_list.append({'key': 'err_amplitude', 'val': percent_error, 'str': f"% err amp: {percent_error:.1f}"})
                
                if 'err_intensity' in metrics_to_incl:
                    # Intensity Difference (Model Intensity - Truth Intensity)
                    intensity_model = get_intensity_of_contour(c_data, mx, my, latitude_weighting=True)
                    intensity_truth = get_intensity_of_contour(truth_contour_data, tx, ty, latitude_weighting=True)
                    percent_error_intensity = ((intensity_model - intensity_truth) / intensity_truth * 100)
                    print(f"ckpt_num: {ckpt_num}, Model pred intensity: {intensity_model:.2f}, Truth intensity: {intensity_truth:.2f}, % err intensity: {percent_error_intensity:.1f}%")
                    metrics_list.append({'key': 'err_intensity', 'val': percent_error_intensity, 'str': f"% err int: {percent_error_intensity:.1f}"})
                    
                # --- Plot Polygons on second figure for Forecast ---
                if poly_ax is not None:
                    poly_ax.set_title(title, fontsize=10)
                    poly_ax.set_extent(extent, crs=ccrs.PlateCarree())
                    poly_ax.coastlines()
                    poly_ax.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
                    
                    model_poly = Polygon(zip(mx, my))
                    truth_poly = Polygon(zip(tx, ty))
                    
                    if model_poly.is_valid and truth_poly.is_valid:
                        # truth
                        x_u, y_u = truth_poly.exterior.xy
                        poly_ax.fill(x_u, y_u, alpha=0.3, fc='grey', ec='black', transform=ccrs.PlateCarree())
                        x_u, y_u = model_poly.exterior.xy
                        poly_ax.fill(x_u, y_u, alpha=0.3, fc='green', ec='lime', transform=ccrs.PlateCarree())
                                
                        poly_ax.text(0.97, 0.03, f'IoU: {iou:.2f}', transform=poly_ax.transAxes, fontsize=12,
                                     bbox=dict(boxstyle='round', fc="w", ec="k", alpha=0.8),
                                    ha='right', va='bottom')

    # 5. Standard Formatting
    ax.set_title(title, fontsize=14)
    ax.coastlines()
    ax.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 6. Legend
    if contour_val is not None and ckpt_num == ckpts[-1] and not is_truth:  # only add legend to last ckpt panel
        legend_elements = [
            Line2D([0], [0], color='red', lw=1.5, label='model'),
            Line2D([0], [0], color='black', lw=1.5, linestyle='dashdot', label='truth')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    # 7. Metrics Text Box (Render individual lines)
    rendered_text_objs = {}
    if len(metrics_list) > 0:
        # Start from bottom and work up
        y_pos = 0.03
        y_step = 0.1
        for metric in reversed(metrics_list): # reverse iteration so MSE is at the top of the stack
            t = ax.text(0.97, y_pos, metric['str'], 
                    transform=ax.transAxes, 
                    color='black', fontsize=11, 
                    bbox=dict(boxstyle='round', fc="w", ec="k", alpha=0.9),
                    ha='right', va='bottom')
            
            # Store the text object and value for global comparison
            rendered_text_objs[metric['key']] = {'val': metric['val'], 'text_obj': t}
            y_pos += y_step

    return rendered_text_objs

def plot_Nckpts_1leadtime(leadtime = 5, ckpts=[1,30,70,90], experiment_number=1, 
                    valid_timestep_str="2022_12_27T00", init_timestep_str="2022_12_22T00", 
                    error=False, var ='tcwv', bounding_box={}, plot_truth=False,
                    cmap=cm.haline, error_cmap = mpl_cm.BrBG, cbar_min=None, cbar_max=None,
                    contour_percentile=None, contour_val=None, white_negative_values=False,
                    plot_truth_contour=False, save_fig=True, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures',
                    title=None,
                    ):
    
    ### SETUP ###
    # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
    if not (f'exp{experiment_number}' in figs_dir):
        figs_dir = f'{figs_dir}/exp{experiment_number}'
        # create the directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
    
    extent = [bounding_box['longitude_min'], bounding_box['longitude_max'], bounding_box['latitude_min'], bounding_box['latitude_max']]
    
    # Determine panels to plot
    panels = [] 
    if plot_truth:
        panels.append('Truth')
    panels.extend(ckpts)
    num_plots = len(panels)

    fig, ax_list, ax = setup_figure_layout(num_plots)
    fig_poly, ax_list_poly, _ = setup_figure_layout(num_plots) #  second figure for Polygons ---
    
    # Load Truth Data
    truth_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/init_files/Initialize_{init_timestep_str}_nsteps{leadtime*4}.nc"
    
    # grab the valid time string from the truth file for the title
    truth_ds = xr.open_dataset(truth_fp)
    print('LOADED INIT TIME', init_timestep_str, 'VAR', var, 'LEADTIME', leadtime, 'TRUTH:', truth_fp)
    print(truth_ds['time'])
    truth_time_ind = 1 # the truth for the valid time is at index 1
    truth_valid_time_str = str(truth_ds['time'].values[truth_time_ind]) 
    truth_ds = truth_ds.isel(time=truth_time_ind).sel(variable=var)
    truth_ds = crop_spatial_bounds(truth_ds, bounding_box)

    if experiment_number == 1:
        # valid_time_ind=2
        # throw an error since experiment 1 is deprecated
        raise ValueError("Experiment 1 is deprecated (valid_time_ind=2 not 0). Please use experiment_number=2 or higher.") 
    else:
        valid_time_ind=0 # the model forecast for the valid time is at index 0

    if error:
        # calculate difference
        error_dict = {}
        for ckpt_num in ckpts:
            output_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/Experiment{experiment_number}/{valid_timestep_str[:10]}/Checkpoint{ckpt_num}_{init_timestep_str}_nsteps{leadtime*4}.nc"
            ds = xr.open_dataset(output_fp)
            print(f'LOADED VALIDTIME {valid_timestep_str} VAR {var} LEADTIME {leadtime} ckpt {ckpt_num} INIT_TIME {init_timestep_str} FORECAST:', output_fp)
            ds_cropped = crop_spatial_bounds(ds, bounding_box)
            print()
            diff = ds_cropped[var].isel(valid_time=valid_time_ind) - truth_ds
            error_dict[ckpt_num] = diff

    # Find cbar limits across all panels first
    use_provided_cbar_lims = False
    if cbar_max is None and cbar_min is None:
        cbar_min, cbar_max = np.inf, -np.inf
    else: 
        use_provided_cbar_lims = True
    if error:
        error_cbar_min, error_cbar_max = np.inf, -np.inf
    ds_cropped_list = {} 

    truth_ds = truth_ds.to_dataarray().squeeze() 

    # Calculate contour value from truth distribution
    if not contour_percentile is None:
        contour_val = np.percentile(truth_ds.values, contour_percentile)
    
    # 1. Check truth limits if plotting truth
    if plot_truth and not use_provided_cbar_lims:
        if white_negative_values:
            truth_ds = truth_ds.where(truth_ds >= 0)
        data = truth_ds.values
        cbar_min = min(cbar_min, np.nanmin(data))
        cbar_max = max(cbar_max, np.nanmax(data))
        
    # 2. Check checkpoint limits
    for ckpt_num in ckpts:
        output_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/Experiment{experiment_number}/{valid_timestep_str[:10]}/Checkpoint{ckpt_num}_{init_timestep_str}_nsteps{leadtime*4}.nc"
        ds = xr.open_dataset(output_fp)
        ds_cropped = crop_spatial_bounds(ds, bounding_box)

        print(f'LOADED VALIDTIME {valid_timestep_str} VAR {var} LEADTIME {leadtime} ckpt {ckpt_num} INIT_TIME {init_timestep_str} FORECAST:', output_fp)
        print('ds.valid_time.values', ds.valid_time.values)

        if white_negative_values and not error:
            ds_cropped[var] = ds_cropped[var].where(ds_cropped[var] >= 0)
        ds_cropped_list[ckpt_num] = ds_cropped
        if error:
            data = error_dict[ckpt_num].to_dataarray().squeeze()
            error_cbar_min = min(error_cbar_min, np.nanmin(data))
            error_cbar_max = max(error_cbar_max, np.nanmax(data))
        elif not use_provided_cbar_lims and not error:
            data = ds_cropped[var].isel(valid_time=valid_time_ind).values  
            cbar_min = min(cbar_min, np.nanmin(data))
            cbar_max = max(cbar_max, np.nanmax(data))
    
    if error:
        abs_max = max(abs(error_cbar_min), abs(error_cbar_max))
        error_cbar_min = -abs_max
        error_cbar_max = abs_max

    # Global tracker for all metrics across panels
    # Format: [{'key': 'mse', 'val': 0.5, 'text_obj': Text}, ...]
    all_metrics_tracker = []

    ### PLOTTING LOOP ###    
    for i, item in enumerate(panels):
        ax_i = ax_list[i]; poly_ax_i = ax_list_poly[i] # Get corresponding poly axis
        
        # --- HELPER CALL 2: Render Panel Content ---
        if item == 'Truth':
            # Truth panel returns None or few metrics
            _ = render_panel(
                ax=ax_i,
                background_data=truth_ds,
                title=f"truth {var} at {truth_valid_time_str[:10]}",
                cmap=cmap, vmin=cbar_min, vmax=cbar_max, extent=extent,
                contour_val=contour_val,
                contour_data=None, # Uses background (truth) for red contour
                truth_contour_data=truth_ds, is_truth=True,
                mse_value=None, # No MSE for truth,
                ckpts=ckpts,
                poly_ax=poly_ax_i,
                var=var,
            )
        else:
            ckpt_num = item
            # Calculate MSE on raw forecast vs truth regardless of plotting mode
            raw_forecast = ds_cropped_list[ckpt_num][var].isel(valid_time=valid_time_ind)
            mse_val = calculate_mse(raw_forecast, truth_ds)

            if error:
                # In error mode
                panel_metrics = render_panel(
                    ax=ax_i,
                    background_data=error_dict[ckpt_num].to_dataarray().squeeze(),
                    title=f"checkpoint {ckpt_num}",
                    cmap=error_cmap, vmin=error_cbar_min, vmax=error_cbar_max, extent=extent,
                    contour_val=contour_val,
                    contour_data=raw_forecast, 
                    truth_contour_data=truth_ds if plot_truth_contour else None,
                    ckpt_num=ckpt_num,
                    mse_value=mse_val,
                    ckpts=ckpts,
                    poly_ax=poly_ax_i,
                    var=var,
                )
            else:
                # In normal mode
                panel_metrics = render_panel(
                    ax=ax_i,
                    background_data=raw_forecast,
                    title=f"checkpoint {ckpt_num}",
                    cmap=cmap, vmin=cbar_min, vmax=cbar_max, extent=extent,
                    contour_val=contour_val,
                    contour_data=None, # Uses background (forecast) for red contour
                    truth_contour_data=truth_ds if plot_truth_contour else None,
                    ckpt_num=ckpt_num,
                    mse_value=mse_val,
                    ckpts=ckpts,
                    poly_ax=poly_ax_i,
                    var=var,
                )
            
            # Add these to our global list
            # panel_metrics is a dict: {'mse': {'val': X, 'text_obj': T}, 'iou': ...}
            if panel_metrics:
                all_metrics_tracker.append(panel_metrics)

    # --- POST-PROCESSING: Find best metrics and Color Boxes ---
    if all_metrics_tracker:
        print('Identifying best metrics across all ckpt panels...')
        # Flatten the structure to easily find min/max
        # Collect all values
        mse_vals = [m['mse']['val'] for m in all_metrics_tracker if 'mse' in m]
        iou_vals = [m['iou']['val'] for m in all_metrics_tracker if 'iou' in m]
        if var == 'tcwv':
            err_amp_vals = [m['err_volume']['val'] for m in all_metrics_tracker if 'err_volume' in m]
        else:
            err_amp_vals = [m['err_amplitude']['val'] for m in all_metrics_tracker if 'err_amplitude' in m]
        err_intensity_vals = [m['err_intensity']['val'] for m in all_metrics_tracker if 'err_intensity' in m]

        best_mse = min(mse_vals) if mse_vals else None
        best_iou = max(iou_vals) if iou_vals else None
        best_err_amp = min(err_amp_vals, key=abs) if err_amp_vals else None
        best_err_intensity = min(err_intensity_vals, key=abs) if err_intensity_vals else None

        # Iterate through all panels and highlight the specific text boxes
        for panel in all_metrics_tracker:
            # MSE
            if best_mse is not None and 'mse' in panel:
                if panel['mse']['val'] == best_mse:
                    panel['mse']['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))
            # IoU
            if best_iou is not None and 'iou' in panel:
                if panel['iou']['val'] == best_iou:
                    panel['iou']['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))
            # Amplitude Error
            if best_err_amp is not None:
                err_amp_key = 'err_volume' if var == 'tcwv' else 'err_amplitude'
                if err_amp_key in panel:
                    if abs(panel[err_amp_key]['val']) == abs(best_err_amp):
                        panel[err_amp_key]['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))
            # Intensity Error
            if best_err_intensity is not None and 'err_intensity' in panel:
                if abs(panel['err_intensity']['val']) == abs(best_err_intensity):
                    panel['err_intensity']['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))


    # Turn off unused axes
    for j in range(num_plots, len(ax_list)):
        ax_list[j].axis('off')
        ax_list_poly[j].axis('off') # Also turn off unused poly axes
    
    # Create Colorbar
    if error:
        bounds = np.linspace(error_cbar_min, error_cbar_max, 25)
        cmap_to_use = error_cmap
    else:
        bounds = np.linspace(cbar_min, cbar_max, 25)
        cmap_to_use = cmap
    
    norm = BoundaryNorm(bounds, cmap_to_use.N)
    mappable = ScalarMappable(norm=norm, cmap=cmap_to_use)
    mappable.set_array([])
    
    if isinstance(ax, np.ndarray) and ax.ndim == 2 and ax.shape[0] > 2:
        # location='bottom' puts it under plots. 
        # aspect=40 makes it thinner (standard is 20). 
        # pad=0.04 ensures separation from plots.
        cbar = fig.colorbar(mappable, ax=ax, location='bottom',
         shrink=0.6, pad=0.04, aspect=40
        )
    else:
        cbar = fig.colorbar(mappable, ax=ax, location='right', shrink=0.8, pad=0.02)
        
    if not contour_percentile is None and not error:
        cbar.ax.hlines(contour_val, 0, 1, colors='red', linewidths=2)

    # --- SAVE AND TITLE MAIN FIGURE ---
    ckpts_str = '_'.join(map(str, ckpts))
    if error:
        cbar.set_label(f"{var} prediction - truth")
        if not title is None:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"{var} prediction - truth at lead time {leadtime} days (exp {experiment_number})", fontsize=16)
        fp = f'{figs_dir}/{var}_leadtime{leadtime}_error_ckpts{ckpts_str}_validtime{valid_timestep_str[:10]}.png'
    else:
        cbar.set_label(f"{var}")
        if not title is None:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"{var} at lead time {leadtime} days (exp {experiment_number})", fontsize=16)
        fp = f'{figs_dir}/{var}_leadtime{leadtime}_ckpts{ckpts_str}_validtime{valid_timestep_str[:10]}.png'
    
    # --- TITLE AND SAVE POLYGON FIGURE ---
    fig_poly.suptitle(f"{var} AR region at lead time {leadtime} days", fontsize=16)
    fp_poly = fp.replace('.png', '_polygons.png')

    if save_fig:
        print('Saving figure to:', fp)
        plt.figure(fig.number) # set active
        plt.savefig(fp, dpi=500)
        
        print('Saving polygon figure to:', fp_poly)
        plt.figure(fig_poly.number) # set active
        plt.savefig(fp_poly, dpi=500)
    
    plt.show()
    return cbar_min, cbar_max, contour_val

def calculate_mse_for_ckpts_and_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box,
                latitude_weighting=True):
    mses = {
        leadtime: [] for leadtime in leadtimes
    }

    for leadtime in leadtimes:
        for ckpt in ckpts:
            # get forecast and truth for this ckpt
            forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
                valid_timestep_str=valid_timestep_str,  
                var=var, bounding_box=bounding_box)
            mse = calculate_mse(forecast, truth, latitude_weighting=latitude_weighting,)
            mses[leadtime].append(mse)
    
    return mses

def plot_mse_ckpt_leadtimes(ckpts, var, experiment_number, highlight_min = True, mses=None,
                figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', bounding_box={},
                normalize=True, latitude_weighting=True, valid_timestep_str='2022-12-27', 
                leadtimes=[3,5,7], presentation_style=False):

    # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
    if not (f'exp{experiment_number}' in figs_dir):
        figs_dir = f'{figs_dir}/exp{experiment_number}'
        # create the directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)

    if mses is None:
        mses = calculate_mse_for_ckpts_and_leadtimes(ckpts=ckpts, leadtimes=leadtimes, var=var, 
        experiment_number=experiment_number, valid_timestep_str=valid_timestep_str, bounding_box=bounding_box,
        latitude_weighting=True)

    # Normalize by the value of the mse_w at the last checkpoint
    if normalize:
        for leadtime in mses:
            last_mse = mses[leadtime][-1] # final ckpt mse
            mses[leadtime] = [mse / last_mse for mse in mses[leadtime]]

    # Make a plot of MSE vs ckpt for different lead times
    fig,ax = plt.subplots(figsize=(6,4))

    cmap = cm.matter
    colors = {
        3: cmap(0.2),
        5: cmap(0.5),
        7: cmap(0.8)
    }
    for leadtime in leadtimes:
        print('plotting leadtime', leadtime, 'mses[::2]:', mses[leadtime][::2],'color', colors[leadtime])
        mses_select = mses[leadtime]
        
        if presentation_style:
            # 1) & 3) Implement gaussian smoothing, plot original in low alpha, smooth lines thicker
            # Change legend elements to 'n-day forecast' (only labelled for the smooth lines)
            ax.plot(ckpts, mses_select, color=colors[leadtime], alpha=0.3, linewidth=1,
            #  marker='o', markersize=2
            )
            # Add smoothed line on top
            mses_smooth = gaussian_filter(mses_select, sigma=2)
            ax.plot(ckpts, mses_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
        else:
            ax.plot(ckpts, mses_select, label=f'leadtime={leadtime} days', color=colors[leadtime], 
            alpha=0.825, linewidth=2, 
            # marker='o', markersize=2
            )

    if highlight_min and not presentation_style: # for now, not highlighting the minimum if plotting in presentation format
        # find the minimum mse for each leadtime and highlight it with a star
        for leadtime in leadtimes:
            mses_select = mses[leadtime]
            min_idx = np.argmin(mses_select)
            # if there are points within 0.5 stdev from the minimum, highlight them too
            stdev = np.std(mses_select)
            for i, mse in enumerate(mses_select):
                threshold = mses_select[min_idx] + 0.1 * stdev
                if mse <= threshold:
                    ax.plot(ckpts[i], mse, marker='*', color=colors[leadtime], markersize=8, alpha=0.8)
                    # ax.text(ckpts[i]-0.03, mse,'🌟',fontsize=10) 
            ax.plot(ckpts[min_idx], mses_select[min_idx], marker='*', color=colors[leadtime], markersize=8, alpha=0.8)
            # ax.text(ckpts[min_idx]-0.03, mses_select[min_idx],'★',fontsize=12)

    ax.set_xlabel('Training Checkpoint')
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )
    
    if presentation_style:
        # 4) Change title for presentation style
        ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Error throughout Training')
    elif latitude_weighting:
        # ax.set_title(f'latitude-weighted MSE of {var} for experiment {experiment_number} event')
        # include the event date in the title
        ax.set_title(f'Latitude-weighted MSE of {var} event on {valid_timestep_str[:10]} (exp {experiment_number})')
    else:
        ax.set_title(f'MSE of {var} for experiment {experiment_number} event')
        
    ax.legend()
    # plt.yscale('log')
    if presentation_style:
        # 2) Change y axis to 'Mean Square Error' with (normalized) if normalized
        ax.set_ylabel("Mean Squared Error (normalized)") if normalize else ax.set_ylabel("Mean Squared Error")
        if normalize:
            # draw a horizontal line at y=1 for presentation style as well
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    elif normalize:
        # draw a horizontal line at y=1
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        if latitude_weighting:
            # set the ylabel to show that it's normalized latitude-weighted mse by symbolizing mse_w / mse_w_final 
            ax.set_ylabel(r"$\frac{\text{mse}_w}{\text{mse}_{w,\text{final}}}$",
            # rotate so that the fraction line is horizontal
            rotation=0, labelpad=20
            )
        else:
            ax.set_ylabel(r"$\frac{\text{mse}}{\text{mse}_{\text{final}}}$",
            # rotate so that the fraction line is horizontal
            rotation=0, labelpad=20
            )
    else:
        if latitude_weighting:
            ax.set_ylabel(r"$\text{mse}_w$")
        else:
            ax.set_ylabel(r"$\text{mse}$")
            
    plt.tight_layout()
    
    # Determine filename suffix based on arguments
    suffix = f"_leadtimes_{'_'.join(map(str, leadtimes))}"
    suffix += 'normalized' if normalize else ''
    suffix += "_presentation.png" if presentation_style else ".png"
    
    if latitude_weighting:
        plt.savefig(f'{figs_dir}/{var}_latitude_weighted_mse_vs_ckpt{suffix}', dpi=300)
    else:
        plt.savefig(f'{figs_dir}/{var}_mse_vs_ckpt{suffix}', dpi=300)
    plt.show()

def plot_iou_ckpt_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
        contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/viz/figures', highlight_max=True,
        presentation_style=False):

    # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
    if not (f'exp{experiment_number}' in figs_dir):
        figs_dir = f'{figs_dir}/exp{experiment_number}'
        # create the directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        
    ious = {
        leadtime: [] for leadtime in leadtimes
    }

    for leadtime in leadtimes:
        for ckpt in ckpts:
            # get forecast and truth for this ckpt
            forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
                valid_timestep_str=valid_timestep_str,  
                var=var, bounding_box=bounding_box)
            
            # calculate contour value from truth distribution
            contour_val = np.percentile(truth.values, contour_percentile)

            # use plt.contourf with levels=[val, np.inf] and alpha=0 to get the contour paths without plotting them, then extract the largest path for both forecast and truth contours to calculate IoU
            cs_forecast = plt.contourf(
                forecast['lon'], forecast['lat'], forecast,
                levels=[contour_val, np.inf], alpha=0
            )
            mx, my = get_largest_path_from_contour(cs_forecast)

            cs_truth = plt.contourf(
                truth['lon'], truth['lat'], truth,
                levels=[contour_val, np.inf], alpha=0
            )
            tx, ty = get_largest_path_from_contour(cs_truth)

            iou = get_IoU_of_contours(mx, my, tx, ty)
            ious[leadtime].append(iou)

            plt.clf()
    
    # Make a plot of IoU vs ckpt for different lead times
    fig,ax = plt.subplots(figsize=(6,4))

    cmap = cm.matter
    colors = {
        3: cmap(0.2),
        5: cmap(0.5),
        7: cmap(0.8)
    }
    for leadtime in [3,5,7]:
        ious_select = ious[leadtime]
        if presentation_style:
            # Plot original lines in low alpha and thinner
            ax.plot(ckpts, ious_select, color=colors[leadtime], alpha=0.3, linewidth=1, )
            # Add smoothed line on top
            ious_smooth = gaussian_filter(ious_select, sigma=2)
            ax.plot(ckpts, ious_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
        else:
            ax.plot(ckpts, ious_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

    if highlight_max:
        # find the maximum iou for each leadtime and highlight it with a star
        for leadtime in [3,5,7]:
            ious_select = np.array(ious[leadtime])
            max_idx = np.argmax(ious_select)
            # if there are points within .1 stdev from the maximum, highlight them too
            stdev = np.std(ious_select)
            for i, iou in enumerate(ious_select):
                threshold = ious_select[max_idx] - 0.1 * stdev
                if iou >= threshold:
                    ax.plot(ckpts[i], iou, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
            ax.plot(ckpts[max_idx], ious_select[max_idx], marker='*', color=colors[leadtime], markersize=8)

    ax.set_xlabel('Checkpoint')
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )

    if presentation_style:
        ax.set_xlabel('Training Checkpoint')
        ax.set_ylabel("Event Location Accuracy (IoU)")
        ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Location\nAccuracy throughout Training')
        plt.ylim(0,0.9)
    else:
        ax.set_ylabel("IoU")
        ax.set_title(f'{var} contour IoU on {valid_timestep_str[:10]} (exp {experiment_number})')
        plt.ylim(0,1)

    ax.legend()
    plt.tight_layout()

    # Determine filename suffix based on presentation_style
    suffix = "_presentation.png" if presentation_style else ".png"

    plt.savefig(f'{figs_dir}/{var}_iou_vs_ckpt_leadtimes{suffix}', dpi=300)
    plt.show()

def plot_err_amp_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
        contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/viz/figures', highlight_min=True,
        absolute_figure=True, presentation_style=False):

    # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
    if not (f'exp{experiment_number}' in figs_dir):
        figs_dir = f'{figs_dir}/exp{experiment_number}'
        # create the directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
    
    err_amps = {
        leadtime: [] for leadtime in leadtimes
    }

    for leadtime in leadtimes:
        for ckpt in ckpts:
            # get forecast and truth for this ckpt
            forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
                valid_timestep_str=valid_timestep_str,  
                var=var, bounding_box=bounding_box)
            
            # calculate contour value from truth distribution
            contour_val = np.percentile(truth.values, contour_percentile)

            cs_forecast = plt.contourf(
                forecast['lon'], forecast['lat'], forecast,
                levels=[contour_val, np.inf], alpha=0
            )
            mx, my = get_largest_path_from_contour(cs_forecast)

            cs_truth = plt.contourf(
                truth['lon'], truth['lat'], truth,
                levels=[contour_val, np.inf], alpha=0
            )
            tx, ty = get_largest_path_from_contour(cs_truth)

            # calculate amplitude difference
            amp_model = get_amplitude_of_contour(forecast, mx, my, latitude_weighting=True)
            amp_truth = get_amplitude_of_contour(truth, tx, ty, latitude_weighting=True)
            percent_error = ((amp_model - amp_truth) / amp_truth * 100) 
            err_amps[leadtime].append(percent_error)

            # clear figure to avoid overlapping contours
            plt.clf()
    
    # Make a plot of % Amp Error vs ckpt for different lead times
    fig, ax = plt.subplots(figsize=(6,4))
    
    # Setup second figure for Absolute Error if requested
    if absolute_figure:
        fig_abs, ax_abs = plt.subplots(figsize=(6,4))
    else:
        fig_abs, ax_abs = None, None

    cmap = cm.matter
    colors = {
        3: cmap(0.2),
        5: cmap(0.5),
        7: cmap(0.8)
    }
    for leadtime in [3,5,7]:
        err_amps_select = err_amps[leadtime]
        
        if presentation_style:
            # 1. Main Plot: Raw (low alpha) + Smooth (thick)
            ax.plot(ckpts, err_amps_select, color=colors[leadtime], alpha=0.3, linewidth=1,)
            err_amps_smooth = gaussian_filter(err_amps_select, sigma=2)
            ax.plot(ckpts, err_amps_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
            
            # 2. Absolute Plot: Abs(Raw) (low alpha) + Abs(Smooth) (thick)
            if absolute_figure:
                abs_errs = np.abs(err_amps_select)
                ax_abs.plot(ckpts, abs_errs, color=colors[leadtime], alpha=0.3, linewidth=1)
                ax_abs.plot(ckpts, np.abs(err_amps_smooth), label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
        
        else:
            # Original Plotting Logic
            ax.plot(ckpts, err_amps_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2)
            if absolute_figure:
                abs_errs = np.abs(err_amps_select)
                ax_abs.plot(ckpts, abs_errs, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2)

    if highlight_min:
        # find the minimum absolute error for each leadtime and highlight it with a star
        for leadtime in [3,5,7]:
            err_amps_select = np.array(err_amps[leadtime])
            # Use absolute error to find the 'best' performance (closest to 0)
            abs_errs = np.abs(err_amps_select)
            min_idx = np.argmin(abs_errs)
            
            # if there are points within 0.05 stdev from the minimum (absolute), highlight them too
            stdev = np.std(abs_errs)
            for i, val in enumerate(err_amps_select):
                threshold = abs_errs[min_idx] + 0.1 * stdev
                if abs(val) <= threshold:
                    # Highlight on original axis
                    ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
                    # Highlight on absolute axis
                    if absolute_figure:
                        ax_abs.plot(ckpts[i], abs(val), marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
            
            # Highlight min on original axis
            ax.plot(ckpts[min_idx], err_amps_select[min_idx], marker='*', color=colors[leadtime], markersize=8)
            # Highlight min on absolute axis
            if absolute_figure:
                ax_abs.plot(ckpts[min_idx], abs(err_amps_select[min_idx]), marker='*', color=colors[leadtime], markersize=8)

    # --- Formatting Original Plot ---
    ax.axhline(0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5) # Add horizontal line at 0
    ax.set_xlabel('Training Checkpoint' if presentation_style else 'Checkpoint')
    
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )
    
    if presentation_style:
        ax.set_ylabel("% Error in Volume ")
        ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Volume\nError throughout Training')
    elif var == 'tcwv':
        ax.set_ylabel("% Volume Error")
        ax.set_title(f'% Volume Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
    else:
        ax.set_ylabel("% Amplitude Error")
        ax.set_title(f'% Amplitude Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
        
    ax.legend()
    # plt.ylim(-100,100)
    
    # Determine filename suffix based on presentation_style
    suffix = "_presentation.png" if presentation_style else ".png"
    
    # Save original explicitly using fig object methods to avoid state confusion
    fig.tight_layout()
    fig.savefig(f'{figs_dir}/{var}_err_amp_vs_ckpt_leadtimes{suffix}', dpi=300)
    
    # --- Formatting Absolute Plot ---
    if absolute_figure:
        ax_abs.set_xlabel('Training Checkpoint' if presentation_style else 'Checkpoint')
        ax_abs.set_xticks(ckpts)
        ax_abs.set_xticklabels(xticklabels)
        
        if presentation_style:
            ax_abs.set_ylabel("Absolute % Error in Water Vapour Volume")
            ax_abs.set_title(f'{valid_timestep_str[:10]} Atmospheric River Volume\nAbsolute Error throughout Training')
        else:
            ax_abs.set_ylabel("Absolute % Amplitude Error")
            ax_abs.set_title(f'Absolute % Amplitude Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
            
        ax_abs.legend()
        
        # Save absolute explicitly using fig_abs object methods
        fig_abs.tight_layout()
        fig_abs.savefig(f'{figs_dir}/{var}_abs_err_amp_vs_ckpt_leadtimes{suffix}', dpi=300)
    
    plt.show()

def plot_err_intensity_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
        contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/viz/figures', highlight_min=True,
        absolute_figure=True, presentation_style=False):

    # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
    if not (f'exp{experiment_number}' in figs_dir):
        figs_dir = f'{figs_dir}/exp{experiment_number}'
        # create the directory if it doesn't exist
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
    
    err_intensities = {
        leadtime: [] for leadtime in leadtimes
    }

    for leadtime in leadtimes:
        for ckpt in ckpts:
            # get forecast and truth for this ckpt
            forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
                valid_timestep_str=valid_timestep_str,  
                var=var, bounding_box=bounding_box)
            
            # calculate contour value from truth distribution
            contour_val = np.percentile(truth.values, contour_percentile)

            cs_forecast = plt.contourf(
                forecast['lon'], forecast['lat'], forecast,
                levels=[contour_val, np.inf], alpha=0
            )
            mx, my = get_largest_path_from_contour(cs_forecast)

            cs_truth = plt.contourf(
                truth['lon'], truth['lat'], truth,
                levels=[contour_val, np.inf], alpha=0
            )
            tx, ty = get_largest_path_from_contour(cs_truth)

            intensity_model = get_intensity_of_contour(forecast, mx, my, latitude_weighting=True)
            intensity_truth = get_intensity_of_contour(truth, tx, ty, latitude_weighting=True)
            percent_error_intensity = ((intensity_model - intensity_truth) / intensity_truth * 100) 
            err_intensities[leadtime].append(percent_error_intensity)

            plt.clf()
    
    # Make a plot of % Intensity Error vs ckpt for different lead times
    fig, ax = plt.subplots(figsize=(6,4))
    # Setup second figure for Absolute Error if requested
    if absolute_figure:
        fig_abs, ax_abs = plt.subplots(figsize=(6,4))
    else:
        fig_abs, ax_abs = None, None
    
    cmap = cm.matter
    colors = {
        3: cmap(0.2),
        5: cmap(0.5),
        7: cmap(0.8)
    }
    for leadtime in [3,5,7]:
        err_intensities_select = err_intensities[leadtime]
        
        if presentation_style:
            # 1. Main Plot
            ax.plot(ckpts, err_intensities_select, color=colors[leadtime], alpha=0.3, linewidth=1)
            err_intensities_smooth = gaussian_filter(err_intensities_select, sigma=2)
            ax.plot(ckpts, err_intensities_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
            
            # 2. Absolute Plot
            if absolute_figure:
                abs_errs = np.abs(err_intensities_select)
                ax_abs.plot(ckpts, abs_errs, color=colors[leadtime], alpha=0.3, linewidth=1)
                ax_abs.plot(ckpts, np.abs(err_intensities_smooth), label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
        else:
            # Original Plotting Logic
            ax.plot(ckpts, err_intensities_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2)
            if absolute_figure:
                abs_errs = np.abs(err_intensities_select)
                ax_abs.plot(ckpts, abs_errs, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2)

    if highlight_min:
        # find the minimum absolute error for each leadtime and highlight it with a star
        for leadtime in [3,5,7]:
            err_intensities_select = np.array(err_intensities[leadtime])
            # Use absolute error to find the 'best' performance (closest to 0)
            abs_errs = np.abs(err_intensities_select)
            min_idx = np.argmin(abs_errs)
            
            # if there are points within 0.05 stdev from the minimum (absolute), highlight them too
            stdev = np.std(abs_errs)
            for i, val in enumerate(err_intensities_select):
                threshold = abs_errs[min_idx] + 0.1 * stdev
                if abs(val) <= threshold:
                    # Highlight on original axis
                    ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
                    # Highlight on absolute axis
                    if absolute_figure:
                        ax_abs.plot(ckpts[i], abs(val), marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
            
            # Highlight min on original axis
            ax.plot(ckpts[min_idx], err_intensities_select[min_idx], marker='*', color=colors[leadtime], markersize=8)
            # Highlight min on absolute axis
            if absolute_figure:
                ax_abs.plot(ckpts[min_idx], abs(err_intensities_select[min_idx]), marker='*', color=colors[leadtime], markersize=8)
    
    # --- Formatting Original Plot ---
    ax.axhline(0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5) # Add horizontal line at 0
    ax.set_xlabel('Training Checkpoint' if presentation_style else 'Checkpoint')
    
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )
    
    if presentation_style:
        ax.set_ylabel("% Error in Intensity")
        ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Intensity\nError throughout Training')
    else:
        ax.set_ylabel("% Intensity Error")
        ax.set_title(f'% Intensity Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
    
    ax.legend()
    # plt.ylim(-100,100)
    
    # Determine filename suffix based on presentation_style
    suffix = "_presentation.png" if presentation_style else ".png"

    # Save original explicitly using fig object methods
    fig.tight_layout()
    fig.savefig(f'{figs_dir}/{var}_err_intensity_vs_ckpt_leadtimes{suffix}', dpi=300)
    
    # --- Formatting Absolute Plot ---
    if absolute_figure:
        ax_abs.set_xlabel('Training Checkpoint' if presentation_style else 'Checkpoint')
        ax_abs.set_xticks(ckpts)
        ax_abs.set_xticklabels(xticklabels, )
        
        if presentation_style:
            ax_abs.set_ylabel("Absolute % Error in Intensity")
            ax_abs.set_title(f'{valid_timestep_str[:10]} Atmospheric River Intensity\nAbsolute Error throughout Training')
        else:
            ax_abs.set_ylabel("Absolute % Intensity Error")
            ax_abs.set_title(f'% Intensity Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
            
        ax_abs.legend()
        
        # Save absolute explicitly using fig_abs object methods
        fig_abs.tight_layout()
        fig_abs.savefig(f'{figs_dir}/{var}_abs_err_intensity_vs_ckpt_leadtimes{suffix}', dpi=300)
    
    plt.show()

# old functions:
# def plot_iou_ckpt_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
#         contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_max=True,
#     ):

#     # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
#     if not (f'exp{experiment_number}' in figs_dir):
#         figs_dir = f'{figs_dir}/exp{experiment_number}'
#         # create the directory if it doesn't exist
#         if not os.path.exists(figs_dir):
#             os.makedirs(figs_dir)
        
#     ious = {
#         leadtime: [] for leadtime in leadtimes
#     }

#     for leadtime in leadtimes:
#         for ckpt in ckpts:
#             # get forecast and truth for this ckpt
#             forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
#                 valid_timestep_str=valid_timestep_str,  
#                 var=var, bounding_box=bounding_box)
            
#             # calculate contour value from truth distribution
#             contour_val = np.percentile(truth.values, contour_percentile)

#             # --- FIX START ---
#             # Changed plt.contour to plt.contourf with levels=[val, np.inf] and alpha=0
#             cs_forecast = plt.contourf(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contourf(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)
#             # --- FIX END ---

#             iou = get_IoU_of_contours(mx, my, tx, ty)
#             ious[leadtime].append(iou)

#             plt.clf()
    
#     # Make a plot of IoU vs ckpt for different lead times
#     fig,ax = plt.subplots(figsize=(6,4))

#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         ious_select = ious[leadtime]
#         ax.plot(ckpts, ious_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

#     if highlight_max:
#         # find the maximum iou for each leadtime and highlight it with a star
#         for leadtime in [3,5,7]:
#             ious_select = np.array(ious[leadtime])
#             max_idx = np.argmax(ious_select)
#             # if there are points within .1 stdev from the maximum, highlight them too
#             stdev = np.std(ious_select)
#             for i, iou in enumerate(ious_select):
#                 threshold = ious_select[max_idx] - 0.1 * stdev
#                 if iou >= threshold:
#                     ax.plot(ckpts[i], iou, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#             ax.plot(ckpts[max_idx], ious_select[max_idx], marker='*', color=colors[leadtime], markersize=8)

#     ax.set_xlabel('Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("IoU")
#     ax.set_title(f'{var} contour IoU on {valid_timestep_str[:10]} (exp {experiment_number})')
#     ax.legend()
#     plt.ylim(0,1)
#     plt.tight_layout()

#     plt.savefig(f'{figs_dir}/{var}_iou_vs_ckpt_leadtimes.png', dpi=300)
#     plt.show()

# def plot_err_amp_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
#         contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=True,
#         absolute_figure=True):

#     # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
#     if not (f'exp{experiment_number}' in figs_dir):
#         figs_dir = f'{figs_dir}/exp{experiment_number}'
#         # create the directory if it doesn't exist
#         if not os.path.exists(figs_dir):
#             os.makedirs(figs_dir)
    
#     err_amps = {
#         leadtime: [] for leadtime in leadtimes
#     }

#     for leadtime in leadtimes:
#         for ckpt in ckpts:
#             # get forecast and truth for this ckpt
#             forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
#                 valid_timestep_str=valid_timestep_str,  
#                 var=var, bounding_box=bounding_box)
            
#             # calculate contour value from truth distribution
#             contour_val = np.percentile(truth.values, contour_percentile)

#             # --- FIX START ---
#             # Changed plt.contour to plt.contourf with levels=[val, np.inf] and alpha=0
#             cs_forecast = plt.contourf(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contourf(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)
#             # --- FIX END ---

#             # calculate amplitude difference
#             amp_model = get_amplitude_of_contour(forecast, mx, my, latitude_weighting=True)
#             amp_truth = get_amplitude_of_contour(truth, tx, ty, latitude_weighting=True)
#             percent_error = ((amp_model - amp_truth) / amp_truth * 100) 
#             err_amps[leadtime].append(percent_error)

#             # clear figure to avoid overlapping contours
#             plt.clf()
    
#     # Make a plot of % Amp Error vs ckpt for different lead times
#     fig, ax = plt.subplots(figsize=(6,4))
    
#     # Setup second figure for Absolute Error if requested
#     if absolute_figure:
#         fig_abs, ax_abs = plt.subplots(figsize=(6,4))
#     else:
#         fig_abs, ax_abs = None, None

#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         err_amps_select = err_amps[leadtime]
        
#         # Plot on original axis
#         ax.plot(ckpts, err_amps_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)
        
#         # Plot on absolute axis
#         if absolute_figure:
#             abs_errs = np.abs(err_amps_select)
#             ax_abs.plot(ckpts, abs_errs, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

#     if highlight_min:
#         # find the minimum absolute error for each leadtime and highlight it with a star
#         for leadtime in [3,5,7]:
#             err_amps_select = np.array(err_amps[leadtime])
#             # Use absolute error to find the 'best' performance (closest to 0)
#             abs_errs = np.abs(err_amps_select)
#             min_idx = np.argmin(abs_errs)
            
#             # if there are points within 0.05 stdev from the minimum (absolute), highlight them too
#             stdev = np.std(abs_errs)
#             for i, val in enumerate(err_amps_select):
#                 threshold = abs_errs[min_idx] + 0.1 * stdev
#                 if abs(val) <= threshold:
#                     # Highlight on original axis
#                     ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#                     # Highlight on absolute axis
#                     if absolute_figure:
#                         ax_abs.plot(ckpts[i], abs(val), marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
            
#             # Highlight min on original axis
#             ax.plot(ckpts[min_idx], err_amps_select[min_idx], marker='*', color=colors[leadtime], markersize=8)
#             # Highlight min on absolute axis
#             if absolute_figure:
#                 ax_abs.plot(ckpts[min_idx], abs(err_amps_select[min_idx]), marker='*', color=colors[leadtime], markersize=8)

#     # --- Formatting Original Plot ---
#     ax.axhline(0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5) # Add horizontal line at 0
#     ax.set_xlabel('Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     if var == 'tcwv':
#         ax.set_ylabel("% Volume Error")
#         ax.set_title(f'% Volume Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
#     else:
#         ax.set_ylabel("% Amplitude Error")
#         ax.set_title(f'% Amplitude Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
#     ax.legend()
#     # plt.ylim(-100,100)
#     plt.tight_layout()
#     # Save original
#     plt.figure(fig.number)
#     plt.savefig(f'{figs_dir}/{var}_err_amp_vs_ckpt_leadtimes.png', dpi=300)
    
#     # --- Formatting Absolute Plot ---
#     if absolute_figure:
#         ax_abs.set_xlabel('Checkpoint')
#         ax_abs.set_xticks(ckpts)
#         ax_abs.set_xticklabels(xticklabels)
#         ax_abs.set_ylabel("Absolute % Amplitude Error")
#         ax_abs.set_title(f'Absolute % Amplitude Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
#         ax_abs.legend()
#         plt.figure(fig_abs.number) # set active figure to abs
#         plt.tight_layout()
#         plt.savefig(f'{figs_dir}/{var}_abs_err_amp_vs_ckpt_leadtimes.png', dpi=300)
    
#     plt.show()

# def plot_err_intensity_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
#         contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=True,
#         absolute_figure=True,):

#     # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
#     if not (f'exp{experiment_number}' in figs_dir):
#         figs_dir = f'{figs_dir}/exp{experiment_number}'
#         # create the directory if it doesn't exist
#         if not os.path.exists(figs_dir):
#             os.makedirs(figs_dir)
    
#     err_intensities = {
#         leadtime: [] for leadtime in leadtimes
#     }

#     for leadtime in leadtimes:
#         for ckpt in ckpts:
#             # get forecast and truth for this ckpt
#             forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
#                 valid_timestep_str=valid_timestep_str,  
#                 var=var, bounding_box=bounding_box)
            
#             # calculate contour value from truth distribution
#             contour_val = np.percentile(truth.values, contour_percentile)

#             # --- FIX START ---
#             # Changed plt.contour to plt.contourf with levels=[val, np.inf] and alpha=0
#             cs_forecast = plt.contourf(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contourf(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)
#             # --- FIX END ---

#             # calculate intensity difference
#             intensity_model = get_intensity_of_contour(forecast, mx, my, latitude_weighting=True)
#             intensity_truth = get_intensity_of_contour(truth, tx, ty, latitude_weighting=True)
#             percent_error_intensity = ((intensity_model - intensity_truth) / intensity_truth * 100) 
#             err_intensities[leadtime].append(percent_error_intensity)

#             # clear figure to avoid overlapping contours
#             plt.clf()
    
#     # Make a plot of % Intensity Error vs ckpt for different lead times
#     fig, ax = plt.subplots(figsize=(6,4))
#     # Setup second figure for Absolute Error if requested
#     if absolute_figure:
#         fig_abs, ax_abs = plt.subplots(figsize=(6,4))
#     else:
#         fig_abs, ax_abs = None, None
    
#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         err_intensities_select = err_intensities[leadtime]
        
#         # Plot on original axis
#         ax.plot(ckpts, err_intensities_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)
        
#         # Plot on absolute axis
#         if absolute_figure:
#             abs_errs = np.abs(err_intensities_select)
#             ax_abs.plot(ckpts, abs_errs, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)
#     if highlight_min:
#         # find the minimum absolute error for each leadtime and highlight it with a star
#         for leadtime in [3,5,7]:
#             err_intensities_select = np.array(err_intensities[leadtime])
#             # Use absolute error to find the 'best' performance (closest to 0)
#             abs_errs = np.abs(err_intensities_select)
#             min_idx = np.argmin(abs_errs)
            
#             # if there are points within 0.05 stdev from the minimum (absolute), highlight them too
#             stdev = np.std(abs_errs)
#             for i, val in enumerate(err_intensities_select):
#                 threshold = abs_errs[min_idx] + 0.1 * stdev
#                 if abs(val) <= threshold:
#                     # Highlight on original axis
#                     ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#                     # Highlight on absolute axis
#                     if absolute_figure:
#                         ax_abs.plot(ckpts[i], abs(val), marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
            
#             # Highlight min on original axis
#             ax.plot(ckpts[min_idx], err_intensities_select[min_idx], marker='*', color=colors[leadtime], markersize=8)
#             # Highlight min on absolute axis
#             if absolute_figure:
#                 ax_abs.plot(ckpts[min_idx], abs(err_intensities_select[min_idx]), marker='*', color=colors[leadtime], markersize=8)
#     # --- Formatting Original Plot ---
#     ax.axhline(0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5) # Add horizontal line at 0
#     ax.set_xlabel('Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("% Intensity Error")
#     ax.set_title(f'% Intensity Error of {var} contour on {valid_timestep_str[:10]} (exp {experiment_number})')
#     ax.legend()
#     # plt.ylim(-100,100)
#     plt.tight_layout()
#     # Save original
#     plt.figure(fig.number)
#     plt.savefig(f'{figs_dir}/{var}_err_intensity_vs_ckpt_leadtimes.png', dpi=300)
#     plt.show()

# # Altered functions for presentation 
# def plot_presentation_iou_ckpt_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
#         contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_max=False,
#     ):
#     """
#     Same as plot_iou_ckpt_leadtimes but with formatting changes for presentation 
#     - smooth over lines with a moving average (similar to event_metrics_smooth = event_metrics.apply(gaussian_filter, sigma=3) ) and make original curves more transparent in the background
#     - smoothed lines are thicker
#     - change title and labels in legend to be more presentation-friendly, i.e. "3 day forecast" instead of "leadtime=3 days" and "atmospheric river location" instead of "IoU"
#     """
#     # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
#     if not (f'exp{experiment_number}' in figs_dir):
#         figs_dir = f'{figs_dir}/exp{experiment_number}'
#         # create the directory if it doesn't exist
#         if not os.path.exists(figs_dir):
#             os.makedirs(figs_dir)
#     ious = {
#         leadtime: [] for leadtime in leadtimes
#     }

#     for leadtime in leadtimes:
#         for ckpt in ckpts:
#             # get forecast and truth for this ckpt
#             forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
#                 valid_timestep_str=valid_timestep_str,  
#                 var=var, bounding_box=bounding_box)
            
#             # calculate contour value from truth distribution
#             contour_val = np.percentile(truth.values, contour_percentile)

#             # Changed plt.contour to plt.contourf with levels=[val, np.inf] and alpha=0
#             cs_forecast = plt.contourf(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contourf(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)

#             iou = get_IoU_of_contours(mx, my, tx, ty)
#             ious[leadtime].append(iou)

#             plt.clf()
#     # Make a plot of IoU vs ckpt for different lead times
#     fig,ax = plt.subplots(figsize=(6,4))
#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         ious_select = ious[leadtime]
#         ax.plot(ckpts, ious_select,
#         #  label=f'{leadtime}-day forecast', 
#          color=colors[leadtime], alpha=0.3, linewidth=1, marker='o', markersize=2)
#         # Add smoothed line on top
#         ious_smooth = gaussian_filter(ious_select, sigma=2)
#         ax.plot(ckpts, ious_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
#     if highlight_max:
#         # find the maximum iou for each leadtime and highlight it with a star
#         for leadtime in [3,5,7]:
#             ious_select = np.array(ious[leadtime])
#             max_idx = np.argmax(ious_select)
#             # if there are points within .1 stdev from the maximum, highlight them too
#             stdev = np.std(ious_select)
#             for i, iou in enumerate(ious_select):
#                 threshold = ious_select[max_idx] - 0.1 * stdev
#                 if iou >= threshold:
#                     ax.plot(ckpts[i], iou, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#             ax.plot(ckpts[max_idx], ious_select[max_idx], marker='*', color=colors[leadtime], markersize=8)
#     ax.set_xlabel('Training Checkpoint') #'Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("Event Location Accuracy (IoU)")
#     # set max of y axis to be 0.9
#     plt.ylim(0,0.9)
#     ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Location\nAccuracy throughout Training')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(f'{figs_dir}/{var}_iou_vs_ckpt_leadtimes_presentation.png', dpi=300)
#     plt.show()

# def plot_presentation_err_amp_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
#         contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=False,
#         absolute_figure=True):
#     """
#     Same as plot_err_amp_ckpts_leadtimes but with formatting changes for presentation 
#     - smooth over lines with a moving average (similar to event_metrics_smooth = event_metrics.apply(gaussian_filter, sigma=3) ) and make original curves more transparent in the background
#     - smoothed lines are thicker
#     - change title and labels in legend to be more presentation-friendly, i.e. "3 day forecast" instead of "leadtime=3 days" and "atmospheric river amplitude error" instead of "% Amplitude Error"
#     """
#     # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
#     if not (f'exp{experiment_number}' in figs_dir):
#         figs_dir = f'{figs_dir}/exp{experiment_number}'
#         # create the directory if it doesn't exist
#         if not os.path.exists(figs_dir):
#             os.makedirs(figs_dir)
    
#     err_amps = {
#         leadtime: [] for leadtime in leadtimes
#     }

#     for leadtime in leadtimes:
#         for ckpt in ckpts:
#             # get forecast and truth for this ckpt
#             forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
#                 valid_timestep_str=valid_timestep_str,  
#                 var=var, bounding_box=bounding_box)
            
#             # calculate contour value from truth distribution
#             contour_val = np.percentile(truth.values, contour_percentile)

#             # Changed plt.contour to plt.contourf with levels=[val, np.inf] and alpha=0
#             cs_forecast = plt.contourf(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contourf(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)
#             # calculate amplitude difference
#             amp_model = get_amplitude_of_contour(forecast, mx, my, latitude_weighting=True)
#             amp_truth = get_amplitude_of_contour(truth, tx, ty, latitude_weighting=True)
#             percent_error = ((amp_model - amp_truth) / amp_truth * 100)
#             err_amps[leadtime].append(percent_error)
#             # clear figure to avoid overlapping contours
#             plt.clf()
#     # Make a plot of % Amp Error vs ckpt for different lead times
#     fig, ax = plt.subplots(figsize=(6,4))
#     # Setup second figure for Absolute Error if requested
#     if absolute_figure:
#         fig_abs, ax_abs = plt.subplots(figsize=(6,4))
#     else:
#         fig_abs, ax_abs = None, None
#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }

#     for leadtime in [3,5,7]:
#         err_amps_select = err_amps[leadtime]
#         # Plot on original axis
#         ax.plot(ckpts, err_amps_select, 
#         # label=f'{leadtime}-day forecast', 
#         color=colors[leadtime], alpha=0.3, linewidth=1, marker='o', markersize=2)
#         # Add smoothed line on top
#         err_amps_smooth = gaussian_filter(err_amps_select, sigma=2)
#         ax.plot(ckpts, err_amps_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
#         # Plot on absolute axis
#         if absolute_figure:
#             abs_errs = np.abs(err_amps_select)
#             ax_abs.plot(ckpts, abs_errs, 
#             # label=f'{leadtime}-day forecast', 
#             color=colors[leadtime], alpha=0.3, linewidth=1, marker='o', markersize=2)
#             ax_abs.plot(ckpts, np.abs(err_amps_smooth), label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
#     if highlight_min:
#         # find the minimum absolute error for each leadtime and highlight it with a star
#         for leadtime in [3,5,7]:
#             err_amps_select = np.array(err_amps[leadtime])
#             # Use absolute error to find the 'best' performance (closest to 0)
#             abs_errs = np.abs(err_amps_select)
#             min_idx = np.argmin(abs_errs)
#             # if there are points within 0.05 stdev from the minimum (absolute), highlight them too
#             stdev = np.std(abs_errs)
#             for i, val in enumerate(err_amps_select):
#                 threshold = abs_errs[min_idx] + 0.1 * stdev
#                 if abs(val) <= threshold:
#                     # Highlight on original axis
#                     ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#                     # Highlight on absolute axis
#                     if absolute_figure:
#                         ax_abs.plot(ckpts[i], abs(val), marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#             # Highlight min on original axis
#             ax.plot(ckpts[min_idx], err_amps_select[min_idx], marker='*', color=colors[leadtime], markersize=8)
#             # Highlight min on absolute axis
#             if absolute_figure:
#                 ax_abs.plot(ckpts[min_idx], abs(err_amps_select[min_idx]), marker='*', color=colors[leadtime], markersize=8)
    
#     # --- Formatting Original Plot ---
#     ax.axhline(0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5) # Add horizontal line at 0
#     ax.set_xlabel('Training Checkpoint') #'Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("% Error in Volume ")
#     ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Volume\nError throughout Training')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(f'{figs_dir}/{var}_err_amp_vs_ckpt_leadtimes_presentation.png', dpi=300)

#     # --- Formatting Absolute Plot ---
#     if absolute_figure:
#         ax_abs.set_xlabel('Training Checkpoint') #'Checkpoint')
#         ax_abs.set_xticks(ckpts)
#         ax_abs.set_xticklabels(xticklabels, )
#         ax_abs.set_ylabel("Absolute % Error in Water Vapour Volume")
#         ax_abs.set_title(f'{valid_timestep_str[:10]} Atmospheric River Volume\nAbsolute Error throughout Training')
#         ax_abs.legend()
#         plt.tight_layout()
#         plt.savefig(f'{figs_dir}/{var}_abs_err_amp_vs_ckpt_leadtimes_presentation.png', dpi=300)
#     plt.show()

# def plot_presentation_err_intensity_ckpt_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
#         contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=False,
#         absolute_figure=True,):
#     """
#     Same as plot_err_intensity_ckpts_leadtimes but with formatting changes for presentation 
#     - smooth over lines with a moving average (similar to event_metrics_smooth = event_metrics.apply(gaussian_filter, sigma=3) ) and make original curves more transparent in the background
#     - smoothed lines are thicker
#     - change title and labels in legend to be more presentation-friendly, i.e. "3 day forecast" instead of "leadtime=3 days" and "atmospheric river intensity error" instead of "% Intensity Error"
#     """
#     # If the subdirectory of the figs_dir is not exp{experiment_number}, add that
#     if not (f'exp{experiment_number}' in figs_dir):
#         figs_dir = f'{figs_dir}/exp{experiment_number}'
#         # create the directory if it doesn't exist
#         if not os.path.exists(figs_dir):
#             os.makedirs(figs_dir)
    
#     err_intensities = {
#         leadtime: [] for leadtime in leadtimes
#     }

#     for leadtime in leadtimes:
#         for ckpt in ckpts:
#             # get forecast and truth for this ckpt
#             forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
#                 valid_timestep_str=valid_timestep_str,  
#                 var=var, bounding_box=bounding_box)
            
#             # calculate contour value from truth distribution
#             contour_val = np.percentile(truth.values, contour_percentile)

#             # Changed plt.contour to plt.contourf with levels=[val, np.inf] and alpha=0
#             cs_forecast = plt.contourf(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contourf(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val, np.inf], alpha=0
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)
#             # calculate intensity difference
#             intensity_model = get_intensity_of_contour(forecast, mx, my, latitude_weighting=True)
#             intensity_truth = get_intensity_of_contour(truth, tx, ty, latitude_weighting=True)
#             percent_error_intensity = ((intensity_model - intensity_truth) / intensity_truth * 100)
#             err_intensities[leadtime].append(percent_error_intensity)
#             # clear figure to avoid overlapping contours
#             plt.clf()
#     # Make a plot of % Intensity Error vs ckpt for different lead times
#     fig, ax = plt.subplots(figsize=(6,4))
#     # Setup second figure for Absolute Error if requested
#     if absolute_figure:
#         fig_abs, ax_abs = plt.subplots(figsize=(6,4))
#     else:
#         fig_abs, ax_abs = None, None
#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         err_intensities_select = err_intensities[leadtime]
#         # Plot on original axis
#         ax.plot(ckpts, err_intensities_select, 
#         # label=f'{leadtime}-day forecast', 
#         color=colors[leadtime], alpha=0.3, linewidth=1, marker='o', markersize=2)
#         # Add smoothed line on top
#         err_intensities_smooth = gaussian_filter(err_intensities_select, sigma=2)
#         ax.plot(ckpts, err_intensities_smooth, label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
#         # Plot on absolute axis
#         if absolute_figure:
#             abs_errs = np.abs(err_intensities_select)
#             ax_abs.plot(ckpts, abs_errs, 
#             # label=f'{leadtime}-day forecast',
#              color=colors[leadtime], alpha=0.3, linewidth=1, marker='o', markersize=2)
#             ax_abs.plot(ckpts, np.abs(err_intensities_smooth), label=f'{leadtime}-day forecast', color=colors[leadtime], alpha=0.9, linewidth=3)
#     if highlight_min:
#         # find the minimum absolute error for each leadtime and highlight it with a star
#         for leadtime in [3,5,7]:
#             err_intensities_select = np.array(err_intensities[leadtime])
#             # Use absolute error to find the 'best' performance (closest to 0)
#             abs_errs = np.abs(err_intensities_select)
#             min_idx = np.argmin(abs_errs)
#             # if there are points within 0.05 stdev from the minimum (absolute), highlight them too
#             stdev = np.std(abs_errs)
#             for i, val in enumerate(err_intensities_select):
#                 threshold = abs_errs[min_idx] + 0.1 * stdev
#                 if abs(val) <= threshold:
#                     # Highlight on original axis
#                     ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#                     # Highlight on absolute axis
#                     if absolute_figure:
#                         ax_abs.plot(ckpts[i], abs(val), marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#             # Highlight min on original axis
#             ax.plot(ckpts[min_idx], err_intensities_select[min_idx], marker='*', color=colors[leadtime], markersize=8)
#             # Highlight min on absolute axis
#             if absolute_figure:
#                 ax_abs.plot(ckpts[min_idx], abs(err_intensities_select[min_idx]), marker='*', color=colors[leadtime], markersize=8)
#     # --- Formatting Original Plot ---
#     ax.axhline(0, linestyle='--', color='gray', alpha=0.7, linewidth=1.5) # Add horizontal line at 0
#     ax.set_xlabel('Training Checkpoint') #'Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 5 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("% Error in Intensity")
#     ax.set_title(f'{valid_timestep_str[:10]} Atmospheric River Intensity\nError throughout Training')
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(f'{figs_dir}/{var}_err_intensity_vs_ckpt_leadtimes_presentation.png', dpi=300)
#     # --- Formatting Absolute Plot ---
#     if absolute_figure:
#         ax_abs.set_xlabel('Training Checkpoint') #'Checkpoint')
#         ax_abs.set_xticks(ckpts)
#         ax_abs.set_xticklabels(xticklabels, )
#         ax_abs.set_ylabel("Absolute % Error in Intensity")
#         ax_abs.set_title(f'{valid_timestep_str[:10]} Atmospheric River Intensity\nAbsolute Error throughout Training')
#         ax_abs.legend()
#         plt.tight_layout()
#         plt.savefig(f'{figs_dir}/{var}_abs_err_intensity_vs_ckpt_leadtimes_presentation.png', dpi=300)
    
#     plt.show()
