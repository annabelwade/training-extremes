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

import warnings
warnings.filterwarnings('ignore')

def crop_spatial_bounds(ds, bounding_box):
    """Helper to crop dataset to bounding box."""
    ds_cropped = ds.where(
        (ds['lat'] >= bounding_box['latitude_min']) & (ds['lat'] <= bounding_box['latitude_max']) &
        (ds['lon'] >= bounding_box['longitude_min']) & (ds['lon'] <= bounding_box['longitude_max']),
        drop=True
    )
    return ds_cropped

def calculate_weighted_mse(forecast, truth):
    """Helper to calculate Latitude-Weighted MSE."""
    squared_diff = (forecast - truth) ** 2
    weights = np.cos(np.deg2rad(forecast.lat))
    weights /= weights.mean() # normalize weights by their mean
    return float((squared_diff*weights).mean())

def setup_figure_layout(num_plots):
    """Helper to handle the rows/columns math and figure initialization."""
    if num_plots > 12:
        rows = num_plots // 5
        cols = num_plots // rows + (num_plots % rows > 0)
        fig_height = rows * 3.5
    elif num_plots > 9:
        rows = 2
        cols = math.ceil(num_plots / 2)
        fig_height = 6 
    else:
        rows = 1
        cols = num_plots
        fig_height = 6 if num_plots <= 4 else 3
    
    fig, ax = plt.subplots(
        rows, cols, figsize=(18, fig_height), 
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

    truth_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/init_files/Initialize_{inference_name}.nc"
    truth_ds = xr.open_dataset(truth_fp)
    truth_ds = truth_ds.isel(time=1).sel(variable=var)
    truth_ds_cropped = crop_spatial_bounds(truth_ds, bounding_box)
    truth_ds_cropped = truth_ds_cropped.to_dataarray().squeeze()

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

def get_amplitude_of_contour(data_array, contour_x, contour_y):
    """
    Helper to sum all the values of that variable within the contour.
    data_array: xarray DataArray with 'lat' and 'lon' dimensions
    contour_x: x-coordinates (lon) of the contour polygon's vertices
    contour_y: y-coordinates (lat) of the contour polygon's vertices
    """

    contour_poly = Polygon(zip(contour_x, contour_y))
    if not contour_poly.is_valid:
        print("Invalid polygon for amplitude calculation.")
        return 0.0

    total_amplitude = 0.0
    for lat in data_array['lat'].values:
        for lon in data_array['lon'].values:
            point = Point(lon, lat)
            if contour_poly.contains(point):
                value = data_array.sel(lat=lat, lon=lon).values
                total_amplitude += value

    return float(total_amplitude)

def render_panel(ax, background_data, title, cmap, vmin, vmax, extent, ckpts=[], ckpt_num=None,
                  contour_val=None, contour_data=None, truth_contour_data=None, is_truth=False,
                  mse_value=None, contour_metrics=True, poly_ax=None):
    """Helper to plot a single panel (contourf + optional contours + formatting)."""
    
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
            # get largest contour path for metrics
            mx, my = get_largest_path_from_contour(cs)

    # 3. Black Contour (Truth Overlay - optional)
    if truth_contour_data is not None and contour_val is not None:
        cs_truth = ax.contour(
            truth_contour_data['lon'], truth_contour_data['lat'], truth_contour_data, linestyles='dashdot',
            levels=[contour_val], colors='black', transform=ccrs.PlateCarree(), alpha=0.8, label='truth',
        )

        if contour_metrics:
            tx, ty = get_largest_path_from_contour(cs_truth)

    # 4. Calculate Metrics
    if contour_metrics and contour_val is not None:
        
        # A. Truth Panel Metrics (Amplitude only)
        if is_truth:
            if tx is not None:
                amp_truth = get_amplitude_of_contour(background_data, tx, ty)
                metrics_list.append({'key': 'amp', 'val': amp_truth, 'str': f"Amp: {amp_truth:.1f}"})
                print(f"Truth total amplitude within contour: {amp_truth:.1f}")
                
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
            if mse_value is not None:
                metrics_list.append({'key': 'mse', 'val': mse_value, 'str': f"MSE: {mse_value:.2f}"})
            
            if mx is not None and tx is not None:
                # IoU
                iou = get_IoU_of_contours(mx, my, tx, ty)
                metrics_list.append({'key': 'iou', 'val': iou, 'str': f"IoU: {iou:.2f}"})
                
                # Amplitude Difference (Model Amp - Truth Amp)
                amp_model = get_amplitude_of_contour(c_data, mx, my)
                print(f"ckpt_num: {ckpt_num}, Model pred total amplitude: {amp_model:.1f}")
                
                # # Amp raw (optional to display, usually good for reference)
                # metrics_list.append({'key': 'amp_raw', 'val': amp_model, 'str': f"Amp: {amp_model:.1f}"})

                amp_truth = get_amplitude_of_contour(truth_contour_data, tx, ty)
                percent_error = ((amp_model - amp_truth) / amp_truth * 100) 
                metrics_list.append({'key': 'err', 'val': percent_error, 'str': f"% err amp: {percent_error:.1f}"})
                
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
        
        # We want the order to match typical reading (MSE top, Err bottom) or stack up?
        # The list was appended MSE -> IoU -> Err. 
        # To display nicely: let's reverse iteration so MSE is at the top of the stack
        for metric in reversed(metrics_list):
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
                    plot_truth_contour=False, save_fig=True,
                    ):
    
    ### SETUP ###
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
    truth_valid_time_str = str(truth_ds['time'].values[1]) 
    truth_ds = truth_ds.isel(time=1).sel(variable=var)
    truth_ds = crop_spatial_bounds(truth_ds, bounding_box)

    if experiment_number == 1:
        valid_time_ind=2 
    else:
        valid_time_ind=0 

    if error:
        # calculate difference
        error_dict = {}
        for ckpt_num in ckpts:
            output_fp = f"/projectnb/eb-general/wade/sfno/inference_runs/sandbox/Experiment{experiment_number}/{valid_timestep_str[:10]}/Checkpoint{ckpt_num}_{init_timestep_str}_nsteps{leadtime*4}.nc"
            ds = xr.open_dataset(output_fp)
            ds_cropped = crop_spatial_bounds(ds, bounding_box)
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
            # Truth panel returns None or minimal metrics, usually we don't box truth metrics
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
                poly_ax=poly_ax_i
            )
        else:
            ckpt_num = item
            # Calculate MSE on raw forecast vs truth regardless of plotting mode
            raw_forecast = ds_cropped_list[ckpt_num][var].isel(valid_time=valid_time_ind)
            mse_val = calculate_weighted_mse(raw_forecast, truth_ds)

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
                    poly_ax=poly_ax_i
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
                    poly_ax=poly_ax_i
                )
            
            # Add these to our global list
            # panel_metrics is a dict: {'mse': {'val': X, 'text_obj': T}, 'iou': ...}
            if panel_metrics:
                all_metrics_tracker.append(panel_metrics)

    # --- POST-PROCESSING: Find best metrics and Color Boxes ---
    if all_metrics_tracker:
        # Flatten the structure to easily find min/max
        # Collect all values
        mse_vals = [m['mse']['val'] for m in all_metrics_tracker if 'mse' in m]
        iou_vals = [m['iou']['val'] for m in all_metrics_tracker if 'iou' in m]
        err_vals = [abs(m['err']['val']) for m in all_metrics_tracker if 'err' in m]

        best_mse = min(mse_vals) if mse_vals else None
        best_iou = max(iou_vals) if iou_vals else None
        best_err_abs = min(err_vals) if err_vals else None

        # Iterate through all panels and highlight the specific text boxes
        for panel in all_metrics_tracker:
            
            # Check MSE
            if 'mse' in panel and best_mse is not None:
                if panel['mse']['val'] == best_mse:
                    panel['mse']['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))
            
            # Check IoU
            if 'iou' in panel and best_iou is not None:
                if panel['iou']['val'] == best_iou:
                    panel['iou']['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))

            # Check % Err (using abs comparison)
            if 'err' in panel and best_err_abs is not None:
                if abs(panel['err']['val']) == best_err_abs:
                    panel['err']['text_obj'].set_bbox(dict(boxstyle='round', fc="w", ec="lime", alpha=0.9, lw=2.5))


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
    
    cbar = fig.colorbar(mappable, ax=ax, location='right', shrink=0.8, pad=0.02)
    if not contour_percentile is None and not error:
        cbar.ax.hlines(contour_val, 0, 1, colors='red', linewidths=2)

    # --- SAVE AND TITLE MAIN FIGURE ---
    ckpts_str = '_'.join(map(str, ckpts))
    if error:
        cbar.set_label(f"{var} prediction - truth")
        fig.suptitle(f"{var} prediction - truth at lead time {leadtime} days (exp {experiment_number})", fontsize=16)
        fp = f'figures/{var}_leadtime{leadtime}_error_ckpts{ckpts_str}_validtime{valid_timestep_str[:10]}_exp{experiment_number}.png'
    else:
        cbar.set_label(f"{var}")
        fig.suptitle(f"{var} at lead time {leadtime} days (exp {experiment_number})", fontsize=16)
        fp = f'figures/{var}_leadtime{leadtime}_ckpts{ckpts_str}_validtime{valid_timestep_str[:10]}_exp{experiment_number}.png'
    
    # --- TITLE AND SAVE POLYGON FIGURE ---
    fig_poly.suptitle(f"{var} largest contours at lead time {leadtime} days", fontsize=16)
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

def calculate_mse_for_ckpts_and_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box):
    mses = {
        leadtime: [] for leadtime in leadtimes
    }

    for leadtime in leadtimes:
        for ckpt in ckpts:
            # get forecast and truth for this ckpt
            forecast, truth = get_forecast_and_truth_data(experiment_number=experiment_number, ckpt_num=ckpt, leadtime=leadtime, 
                valid_timestep_str=valid_timestep_str,  
                var=var, bounding_box=bounding_box)
            mse = calculate_weighted_mse(forecast, truth)
            mses[leadtime].append(mse)
    
    return mses

def mse_ckpt_leadtimes(ckpts, mses, var, experiment_number, highlight_min = True, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures'):
    # Make a plot of MSE vs ckpt for different lead times
    fig,ax = plt.subplots(figsize=(6,4))

    cmap = cm.matter
    colors = {
        3: cmap(0.2),
        5: cmap(0.5),
        7: cmap(0.8)
    }
    for leadtime in [3,5,7]:
        mses_select = mses[leadtime]
        ax.plot(ckpts, mses_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

    if highlight_min:
        # find the minimum mse for each leadtime and highlight it with a star
        for leadtime in [3,5,7]:
            mses_select = mses[leadtime]
            min_idx = np.argmin(mses_select)
            # if there are points within 0.5 stdev from the minimum, highlight them too
            stdev = np.std(mses_select)
            for i, mse in enumerate(mses_select):
                threshold = mses_select[min_idx] + 0.05 * stdev
                if mse <= threshold:
                    ax.plot(ckpts[i], mse, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
                    # ax.text(ckpts[i]-0.03, mse,'🌟',fontsize=10) 
            ax.plot(ckpts[min_idx], mses_select[min_idx], marker='*', color=colors[leadtime], markersize=8,)
            # ax.text(ckpts[min_idx]-0.03, mses_select[min_idx],'★',fontsize=12)

    ax.set_xlabel('Checkpoint')
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 3 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )
    ax.set_ylabel(r"$\text{mse}_w$")
    ax.set_title(f'latitude-weighted MSE of {var} for experiment {experiment_number} event')
    ax.legend()
    plt.yscale('log')
    plt.tight_layout()
    output_dir = figs_dir + f'/exp{experiment_number}/'
    plt.savefig(f'{output_dir}/{var}_mse_vs_ckpt_leadtimes.png', dpi=300)
    plt.show()

def iou_ckpt_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_max=True):
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

            # get largest contours
            cs_forecast = plt.contour(
                forecast['lon'], forecast['lat'], forecast,
                levels=[contour_val]
            )
            mx, my = get_largest_path_from_contour(cs_forecast)

            cs_truth = plt.contour(
                truth['lon'], truth['lat'], truth,
                levels=[contour_val]
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
        ax.plot(ckpts, ious_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

    if highlight_max:
        # find the maximum iou for each leadtime and highlight it with a star
        for leadtime in [3,5,7]:
            ious_select = np.array(ious[leadtime])
            max_idx = np.argmax(ious_select)
            # if there are points within 0.05 stdev from the maximum, highlight them too
            stdev = np.std(ious_select)
            for i, iou in enumerate(ious_select):
                threshold = ious_select[max_idx] - 0.05 * stdev
                if iou >= threshold:
                    ax.plot(ckpts[i], iou, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
            ax.plot(ckpts[max_idx], ious_select[max_idx], marker='*', color=colors[leadtime], markersize=8)

    ax.set_xlabel('Checkpoint')
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 3 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )
    ax.set_ylabel("IoU")
    ax.set_title(f'IoU of {var} contours for experiment {experiment_number} event')
    ax.legend()
    plt.ylim(0,1)
    plt.tight_layout()
    output_dir = figs_dir + f'/exp{experiment_number}/'
    plt.savefig(f'{output_dir}/{var}_iou_vs_ckpt_leadtimes.png', dpi=300)
    plt.show()

def err_amp_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=True,
absolute_figure=True):
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

            # get largest contours
            cs_forecast = plt.contour(
                forecast['lon'], forecast['lat'], forecast,
                levels=[contour_val]
            )
            mx, my = get_largest_path_from_contour(cs_forecast)

            cs_truth = plt.contour(
                truth['lon'], truth['lat'], truth,
                levels=[contour_val]
            )
            tx, ty = get_largest_path_from_contour(cs_truth)

            # calculate amplitude difference
            amp_model = get_amplitude_of_contour(forecast, mx, my)
            amp_truth = get_amplitude_of_contour(truth, tx, ty)
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
        
        # Plot on original axis
        ax.plot(ckpts, err_amps_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)
        
        # Plot on absolute axis
        if absolute_figure:
            abs_errs = np.abs(err_amps_select)
            ax_abs.plot(ckpts, abs_errs, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

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
                threshold = abs_errs[min_idx] + 0.05 * stdev
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
    ax.set_xlabel('Checkpoint')
    # make the xlabel have finer ticks for ckpts
    ax.set_xticks(ckpts)
    xticklabels = [str(ckpt) if i % 3 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
    ax.set_xticklabels(xticklabels, )
    ax.set_ylabel("% Amplitude Error")
    ax.set_title(f'% Amplitude Error of {var} contours for experiment {experiment_number} event')
    ax.legend()
    # plt.ylim(-100,100)
    plt.tight_layout()
    output_dir = figs_dir + f'/exp{experiment_number}/'
    
    # Save original
    plt.figure(fig.number)
    plt.savefig(f'{output_dir}/{var}_err_amp_vs_ckpt_leadtimes.png', dpi=300)
    
    # --- Formatting Absolute Plot ---
    if absolute_figure:
        ax_abs.set_xlabel('Checkpoint')
        ax_abs.set_xticks(ckpts)
        ax_abs.set_xticklabels(xticklabels)
        ax_abs.set_ylabel("Absolute % Amplitude Error")
        ax_abs.set_title(f'Absolute % Amplitude Error of {var} contours for experiment {experiment_number} event')
        ax_abs.legend()
        plt.figure(fig_abs.number) # set active figure to abs
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{var}_abs_err_amp_vs_ckpt_leadtimes.png', dpi=300)
    
    plt.show()

# def err_amp_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
# contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=True,
# absolute_figure=True):
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

#             # get largest contours
#             cs_forecast = plt.contour(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val]
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contour(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val]
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)

#             # calculate amplitude difference
#             amp_model = get_amplitude_of_contour(forecast, mx, my)
#             amp_truth = get_amplitude_of_contour(truth, tx, ty)
#             percent_error = ((amp_model - amp_truth) / amp_truth * 100) 
#             err_amps[leadtime].append(percent_error)

#             # clear figure to avoid overlapping contours
#             plt.clf()
    
#     # Make a plot of % Amp Error vs ckpt for different lead times
#     fig,ax = plt.subplots(figsize=(6,4))

#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         err_amps_select = err_amps[leadtime]
#         ax.plot(ckpts, err_amps_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

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
#                 threshold = abs_errs[min_idx] + 0.05 * stdev
#                 if abs(val) <= threshold:
#                     ax.plot(ckpts[i], val, marker='*', color=colors[leadtime], markersize=8, alpha=0.75)
#             ax.plot(ckpts[min_idx], err_amps_select[min_idx], marker='*', color=colors[leadtime], markersize=8)

#     ax.set_xlabel('Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 3 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("% Amplitude Error")
#     ax.set_title(f'% Amplitude Error of {var} contours for experiment {experiment_number} event')
#     ax.legend()
#     # plt.ylim(-100,100)
#     plt.tight_layout()
#     output_dir = figs_dir + f'/exp{experiment_number}/'
#     plt.savefig(f'{output_dir}/{var}_err_amp_vs_ckpt_leadtimes.png', dpi=300)
#     plt.show()

# def iou_ckpt_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
# contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_max=True):
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

#             # get largest contours
#             cs_forecast = plt.contour(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val]
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contour(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val]
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
#         ax.plot(ckpts, ious_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

#     ax.set_xlabel('Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 3 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("IoU")
#     ax.set_title(f'IoU of {var} contours for experiment {experiment_number} event')
#     ax.legend()
#     plt.ylim(0,1)
#     plt.tight_layout()
#     output_dir = figs_dir + f'/exp{experiment_number}/'
#     plt.savefig(f'{output_dir}/{var}_iou_vs_ckpt_leadtimes.png', dpi=300)
#     plt.show()

# def err_amp_ckpts_leadtimes(ckpts, leadtimes, var, experiment_number, valid_timestep_str, bounding_box, 
# contour_percentile=80, figs_dir = '/projectnb/eb-general/wade/sfno/inference/figures', highlight_min=True):
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

#             # get largest contours
#             cs_forecast = plt.contour(
#                 forecast['lon'], forecast['lat'], forecast,
#                 levels=[contour_val]
#             )
#             mx, my = get_largest_path_from_contour(cs_forecast)

#             cs_truth = plt.contour(
#                 truth['lon'], truth['lat'], truth,
#                 levels=[contour_val]
#             )
#             tx, ty = get_largest_path_from_contour(cs_truth)

#             # calculate amplitude difference
#             amp_model = get_amplitude_of_contour(forecast, mx, my)
#             amp_truth = get_amplitude_of_contour(truth, tx, ty)
#             percent_error = ((amp_model - amp_truth) / amp_truth * 100) 
#             err_amps[leadtime].append(percent_error)

#             # clear figure to avoid overlapping contours
#             plt.clf()
    
#     # Make a plot of % Amp Error vs ckpt for different lead times
#     fig,ax = plt.subplots(figsize=(6,4))

#     cmap = cm.matter
#     colors = {
#         3: cmap(0.2),
#         5: cmap(0.5),
#         7: cmap(0.8)
#     }
#     for leadtime in [3,5,7]:
#         err_amps_select = err_amps[leadtime]
#         ax.plot(ckpts, err_amps_select, label=f'leadtime={leadtime} days', color=colors[leadtime], alpha=0.9, linewidth=2, marker='o', markersize=2)

#     ax.set_xlabel('Checkpoint')
#     # make the xlabel have finer ticks for ckpts
#     ax.set_xticks(ckpts)
#     xticklabels = [str(ckpt) if i % 3 == 0 else ''  for i, ckpt in enumerate(ckpts) ]
#     ax.set_xticklabels(xticklabels, )
#     ax.set_ylabel("% Amplitude Error")
#     ax.set_title(f'% Amplitude Error of {var} contours for experiment {experiment_number} event')
#     ax.legend()
#     plt.ylim(-100,100)
#     plt.tight_layout()
#     output_dir = figs_dir + f'/exp{experiment_number}/'
#     plt.savefig(f'{output_dir}/{var}_err_amp_vs_ckpt_leadtimes.png', dpi=300)
#     plt.show()

