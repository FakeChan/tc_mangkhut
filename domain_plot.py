import f90nml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os

def plot_wrf_domains(namelist_path,output_path):
    """
    解析 namelist.wps 文件并绘制 WRF 模式的嵌套域地图。

    Args:
        namelist_path (str): namelist.wps 文件的路径。
    """
    if not os.path.exists(namelist_path):
        print(f"错误: 未找到文件 '{namelist_path}'")
        return

    # 1. 使用 f90nml 读取和解析 namelist 文件
    try:
        nml = f90nml.read(namelist_path)
        geogrid_nml = nml['geogrid']
        share_nml = nml['share']
    except Exception as e:
        print(f"解析 namelist 文件时出错: {e}")
        return

    # 2. 提取地图投影所需的参数
    # 从文件中读取投影信息 
    map_proj = geogrid_nml['map_proj'].strip().lower()
    ref_lat = geogrid_nml['ref_lat']
    ref_lon = geogrid_nml['ref_lon']
    truelat1 = geogrid_nml['truelat1']
    # 如果 truelat2 不存在或为空, WRF 会默认使用 truelat1 的值 
    truelat2 = geogrid_nml.get('truelat2', truelat1)
    stand_lon = geogrid_nml.get('stand_lon', ref_lon)

    # 3. 创建 Cartopy 地图投影
    if map_proj == 'lambert':
        projection = ccrs.LambertConformal(
            central_longitude=stand_lon,
            central_latitude=ref_lat,
            standard_parallels=(truelat1, truelat2)
        )
    else:
        raise ValueError(f"不支持的地图投影: {map_proj}。请修改脚本以支持其他投影。")
    
    # 地理坐标系，用于添加经纬度标签
    geodetic_proj = ccrs.PlateCarree()

    # 4. 初始化绘图
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_title("WRF Domain Configuration", fontsize=16)

    # 5. 提取并计算每个域的边界
    # namelist.wps 中 max_dom=1 但参数为2个域，因此我们以参数列表长度为准 [cite: 1, 2]
    e_we_list = np.atleast_1d(geogrid_nml['e_we'])
    e_sn_list = np.atleast_1d(geogrid_nml['e_sn'])
    max_dom = len(e_we_list)
    
    dx_parent = geogrid_nml['dx'] # 父域的dx 
    dy_parent = geogrid_nml['dy'] # 父域的dy 
    ratio_list = np.atleast_1d(geogrid_nml.get('parent_grid_ratio', 1))
    i_start_list = np.atleast_1d(geogrid_nml.get('i_parent_start', 1))
    j_start_list = np.atleast_1d(geogrid_nml.get('j_parent_start', 1))
    
    domain_corners = []
    
    # 循环处理每个域
    for i in range(max_dom):
        e_we = e_we_list[i]
        e_sn = e_sn_list[i]

        if i == 0:  # 父域 (d01)
            dx = dx_parent
            dy = dy_parent
            # 父域的中心位于投影坐标系的原点 (0,0)
            width = (e_we - 1) * dx
            height = (e_sn - 1) * dy
            ll_x = -width / 2.0
            ll_y = -height / 2.0
            parent_ll_x = ll_x  # 保存父域左下角坐标，用于计算子域位置
            parent_ll_y = ll_y
        else:  # 嵌套域 (d02, d03, ...)
            ratio = ratio_list[i]
            i_start = i_start_list[i]
            j_start = j_start_list[i]
            
            # 计算嵌套域的分辨率
            dx = dx_parent / np.prod(ratio_list[1:i+1])
            dy = dy_parent / np.prod(ratio_list[1:i+1])

            # 基于父域计算嵌套域的左下角坐标
            offset_x = (i_start - 1) * (dx * ratio)
            offset_y = (j_start - 1) * (dy * ratio)
            ll_x = parent_ll_x + offset_x
            ll_y = parent_ll_y + offset_y
            
            width = (e_we - 1) * dx
            height = (e_sn - 1) * dy

        # 记录所有域的角点，以便后续自动设置地图范围
        domain_corners.append([ll_x, ll_y])
        domain_corners.append([ll_x + width, ll_y + height])
        
        # 绘制矩形框
        domain_color = 'blue' if i == 0 else 'red'
        ax.add_patch(mpatches.Rectangle(
            (ll_x, ll_y), width, height,
            fill=None,
            edgecolor=domain_color,
            linewidth=2,
            label=f'd0{i+1}'
        ))
        # 添加域名标签
        ax.text(ll_x + width * 0.02, ll_y + height * 0.02, f'd0{i+1}',
                color=domain_color, fontsize=12, weight='bold', va='bottom', ha='left')

    # 6. 美化地图
    # 自动计算并设置地图的显示范围
    corners = np.array(domain_corners)
    min_x, min_y = corners.min(axis=0)
    max_x, max_y = corners.max(axis=0)
    buffer_x = (max_x - min_x) * 0.1
    buffer_y = (max_y - min_y) * 0.1
    ax.set_extent([min_x - buffer_x, max_x + buffer_x, 
                   min_y - buffer_y, max_y + buffer_y], crs=projection)

    # 添加地理元素
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='black')
    ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.coastlines(resolution='50m')
    
    # 添加网格线和经纬度标签
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')


# --- 脚本执行入口 ---
if __name__ == "__main__":
    output_path='/share/home/lililei1/kcfu/tc_mangkhut/plot_scripts/figs/domain_plot.png'
    plot_wrf_domains('/share/home/lililei1/kcfu/models/real_WRF_WPS/V4.1/WPS-4.1/namelist.wps',output_path)