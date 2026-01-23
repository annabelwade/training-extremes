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

from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

import warnings
warnings.filterwarnings('ignore')

def calculate_weighted_mse(forecast, truth):
    """Helper to calculate Latitude-Weighted MSE."""
    squared_diff = (forecast - truth) ** 2
    weights = np.cos(np.deg2rad(forecast.lat))
    weights /= weights.mean() # normalize weights by their mean
    return float((squared_diff*weights).mean())

def setup_figure_layout(num_plots):
    """Helper to handle the rows/columns math and figure initialization."""
    if num_plots > 9:
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

def render_panel(ax, background_data, title, cmap, vmin, vmax, extent, ckpt_num=None,
                  contour_val=None, contour_data=None, truth_contour_data=None, is_truth=False,
                  mse_value=None,):
    """Helper to plot a single panel (contourf + optional contours + formatting)."""
    
    # 1. Main Background Plot
    ax.contourf(
        background_data['lon'], background_data['lat'], background_data,
        transform=ccrs.PlateCarree(), 
        cmap=cmap, vmin=vmin, vmax=vmax, levels=25
    )
    
    # 2. Red Contour (Feature Tracking)
    if contour_val is not None and is_truth is False: # only plot red contour for model
        c_data = contour_data if contour_data is not None else background_data
        ax.contour(
            c_data['lon'], c_data['lat'], c_data,
            levels=[contour_val], colors='red', transform=ccrs.PlateCarree(), label='model',
        )

    # 3. Black Contour (Truth Overlay - optional)
    if truth_contour_data is not None and contour_val is not None:
        ax.contour(
            truth_contour_data['lon'], truth_contour_data['lat'], truth_contour_data, linestyles='dashdot',
            levels=[contour_val], colors='black', transform=ccrs.PlateCarree(), alpha=0.8, label='truth',
        )

    # 4. Standard Formatting
    ax.set_title(title, fontsize=14)
    ax.coastlines()
    ax.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # 5. Legend for Contours
    if contour_val is not None and ckpt_num == ckpts[-1]: # only add legend to last ckpt panel
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='model'),
            Line2D([0], [0], color='black', lw=2, linestyle='dashdot', label='truth')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    # 6. Weighted MSE Text
    if mse_value is not None:
        mse_w_str = r"$\text{mse}_w$"
        ax.text(0.97, 0.03, f"{mse_w_str}: {mse_value:.2f}", 
                transform=ax.transAxes, 
                color='black', fontsize=12, 
                bbox=dict(boxstyle='round', fc="w", ec="k"),
                ha='right', va='bottom')

def crop_spatial_bounds(ds, bounding_box):
    """Helper to crop dataset to bounding box."""
    ds_cropped = ds.where(
        (ds['lat'] >= bounding_box['latitude_min']) & (ds['lat'] <= bounding_box['latitude_max']) &
        (ds['lon'] >= bounding_box['longitude_min']) & (ds['lon'] <= bounding_box['longitude_max']),
        drop=True
    )
    return ds_cropped

def plot_Nckpts_1leadtime(leadtime = 5, ckpts=[1,30,70,90], experiment_number=1, 
                    valid_timestep_str="2022_12_2T00", init_timestep_str="2022_12_22T00", 
                    error=False, var ='tcwv', bounding_box={}, plot_truth=False,
                    cmap=cm.haline, error_cmap = mpl_cm.BrBG, cbar_min=None, cbar_max=None,
                    contour_percentile=None, contour_val=None, white_negative_values=False,
                    plot_truth_contour=False,
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

    ### PLOTTING LOOP ###    
    for i, item in enumerate(panels):
        ax_i = ax_list[i]
        
        # --- HELPER CALL 2: Render Panel Content ---
        if item == 'Truth':
            render_panel(
                ax=ax_i,
                background_data=truth_ds,
                title=f"truth {var} at {truth_valid_time_str[:10]}",
                cmap=cmap, vmin=cbar_min, vmax=cbar_max, extent=extent,
                contour_val=contour_val,
                contour_data=None, # Uses background (truth) for red contour
                truth_contour_data=truth_ds, is_truth=True,
                mse_value=None, # No MSE for truth
            )
        else:
            ckpt_num = item
            # Calculate MSE on raw forecast vs truth regardless of plotting mode
            raw_forecast = ds_cropped_list[ckpt_num][var].isel(valid_time=valid_time_ind)
            mse_val = calculate_weighted_mse(raw_forecast, truth_ds)

            if error:
                # In error mode: Background is error dict, Contour is raw forecast
                render_panel(
                    ax=ax_i,
                    background_data=error_dict[ckpt_num].to_dataarray().squeeze(),
                    title=f"checkpoint {ckpt_num}",
                    cmap=error_cmap, vmin=error_cbar_min, vmax=error_cbar_max, extent=extent,
                    contour_val=contour_val,
                    contour_data=raw_forecast, 
                    truth_contour_data=truth_ds if plot_truth_contour else None,
                    ckpt_num=ckpt_num,
                    mse_value=mse_val,
                )
            else:
                # In normal mode: Background is raw forecast, Contour is raw forecast
                render_panel(
                    ax=ax_i,
                    background_data=raw_forecast,
                    title=f"checkpoint {ckpt_num}",
                    cmap=cmap, vmin=cbar_min, vmax=cbar_max, extent=extent,
                    contour_val=contour_val,
                    contour_data=None, # Uses background (forecast) for red contour
                    truth_contour_data=truth_ds if plot_truth_contour else None,
                    ckpt_num=ckpt_num,
                    mse_value=mse_val,
                )

    # Turn off unused axes
    for j in range(num_plots, len(ax_list)):
        ax_list[j].axis('off')
    
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
    
    cbar = fig.colorbar(mappable, ax=ax, location='right', shrink=0.9, pad=0.02)
    if not contour_percentile is None and not error:
        cbar.ax.hlines(contour_val, 0, 1, colors='red', linewidths=2)

    if error:
        cbar.set_label(f"{var} prediction - truth")
        fig.suptitle(f"{var} prediction - truth at lead time {leadtime} days (exp {experiment_number})", fontsize=16)
        ckpts_str = '_'.join(map(str, ckpts))
        fp = f'figures/{var}_leadtime{leadtime}_error_ckpts{ckpts_str}_validtime{valid_timestep_str[:10]}_exp{experiment_number}.png'
    else:
        cbar.set_label(f"{var}")
        fig.suptitle(f"{var} at lead time {leadtime} days (exp {experiment_number})", fontsize=16)
        ckpts_str = '_'.join(map(str, ckpts))
        fp = f'figures/{var}_leadtime{leadtime}_ckpts{ckpts_str}_validtime{valid_timestep_str[:10]}_exp{experiment_number}.png'
    print('Saving figure to:', fp)
    plt.savefig(fp, dpi=500)
    plt.show()
    return cbar_min, cbar_max, contour_val
