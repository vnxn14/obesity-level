import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


obesity_order = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III',
    'Overweight_Level_I',
    'Overweight_Level_II'
]


# Shared Function 1 --- Excel-Style 3D Column Chart
def excel_style_3d_column_chart(
    categories,
    values,
    title,
    x_label='',
    y_label='Frequency',
    figsize=(11, 7),
    bar_width=0.45
):

    # Excel-like colors
    front_color = '#4472C4'
    side_color = '#2F5597'
    top_color = '#5B9BD5'

    # Background colors
    chart_bg = '#FFFFFF'
    plot_bg = '#FFFFFF'

    # Gridline colors
    major_grid_color = '#D9D9D9'
    minor_grid_color = '#EDEDED'

    text_color = '#404040'
    depth_x = 0.12
    depth_y = max(values) * 0.04 if max(values) > 0 else 5
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(chart_bg)
    ax.set_facecolor(plot_bg)
    x = np.arange(len(categories))
    max_value = max(values)


    # y-axis settings
    y_max = np.ceil(max_value * 1.20 / 10) * 10

    if y_max <= 100:
        major_step = 10
    elif y_max <= 300:
        major_step = 20
    elif y_max <= 600:
        major_step = 50
    else:
        major_step = 100

    minor_step = major_step / 6
    major_yticks = np.arange(0, y_max + major_step, major_step)
    minor_yticks = np.arange(0, y_max + minor_step, minor_step)
    plot_left = -0.55
    plot_right = len(categories) - 0.45


    # Primary minor horizontal gridlines
    for y in minor_yticks:
        if not np.any(np.isclose(y, major_yticks)):

            ax.plot(
                [plot_left, plot_right],
                [y, y],
                color=minor_grid_color,
                linewidth=0.6,
                zorder=0
            )

            ax.plot(
                [plot_left, plot_left + depth_x],
                [y, y + depth_y],
                color=minor_grid_color,
                linewidth=0.6,
                zorder=0
            )


    # Primary major horizontal gridlines
    for y in major_yticks:
        ax.plot(
            [plot_left, plot_right],
            [y, y],
            color=major_grid_color,
            linewidth=0.9,
            zorder=0
        )

        ax.plot(
            [plot_left, plot_left + depth_x],
            [y, y + depth_y],
            color=major_grid_color,
            linewidth=0.9,
            zorder=0
        )


    # Primary major vertical gridlines
    for i in range(len(categories) + 1):
        xpos = i - 0.5
        ax.plot(
            [xpos, xpos],
            [0, y_max],
            color=major_grid_color,
            linewidth=0.8,
            zorder=0
        )


    # Top Perspective Line
    ax.plot(
        [plot_left + depth_x, plot_right + depth_x],
        [y_max + depth_y, y_max + depth_y],
        color=major_grid_color,
        linewidth=0.8,
        zorder=0
    )
 
 
    # Draw 3D Bars
    for i, value in enumerate(values):
        
        left = x[i] - bar_width / 2
        right = x[i] + bar_width / 2


        front = Rectangle((left, 0), bar_width,value, facecolor=front_color, edgecolor='none', zorder=3)
        ax.add_patch(front)


        side = Polygon(
            [
                (right, 0),
                (right + depth_x, depth_y),
                (right + depth_x, value + depth_y),
                (right, value)
            ],
            closed=True,
            facecolor=side_color,
            edgecolor='none',
            zorder=2
        )
        ax.add_patch(side)


        top = Polygon(
            [
                (left, value),
                (right, value),
                (right + depth_x, value + depth_y),
                (left + depth_x, value + depth_y)
            ],
            closed=True,
            facecolor=top_color,
            edgecolor='none',
            zorder=4
        )
        ax.add_patch(top)


        # Data Labels
        ax.text(
            x[i] + depth_x / 2,
            value + depth_y + y_max * 0.015,
            f'{int(value)}',
            ha='center',
            va='bottom',
            fontsize=10,
            color=text_color,
            zorder=5
        )


    #  Title
    ax.set_title(title,fontsize=20,color='#595959',pad=20)


    # Axis Labels
    ax.set_xlabel( x_label, fontsize=11, color=text_color, labelpad=15)
    ax.set_ylabel(y_label, fontsize=11, color=text_color, labelpad=10)


    # Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=text_color)

    ax.set_yticks(major_yticks)
    ax.set_yticklabels([str(int(y)) for y in major_yticks], fontsize=10, color=text_color)


    # Limits
    ax.set_xlim(-0.7, len(categories) - 0.3 + depth_x)
    ax.set_ylim(0, y_max + depth_y * 2)


    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='both', length=0)
    ax.grid(False)
    plt.tight_layout()
    return fig, ax


# Shared Function 2 --- Excel-Style 3D Histogram
def excel_style_3d_histogram_same_width(
    bin_labels,
    values,
    title,
    x_label='',
    y_label='Frequency',
    figsize=(11, 7),
    bar_width=0.45
):
    
    # Excel-like colors
    front_color = '#4472C4'
    side_color = '#2F5597'
    top_color = '#5B9BD5'

    # Background colors
    chart_bg = '#FFFFFF'
    plot_bg = '#FFFFFF'

    # Gridline colors
    major_grid_color = '#D9D9D9'
    minor_grid_color = '#EDEDED'

    # Bar line color
    bar_line_color = 'white'
    text_color = '#404040'

    # 3D depth
    depth_x = 0.12
    depth_y = max(values) * 0.04 if max(values) > 0 else 5
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(chart_bg)
    ax.set_facecolor(plot_bg)

    # Keep bars touching while maintaining width = 0.45
    x = np.arange(len(bin_labels)) * bar_width
    max_value = max(values)


    # y-axis settings
    y_max = np.ceil(max_value * 1.20 / 10) * 10
    
    if y_max <= 100:
        major_step = 10
    elif y_max <= 300:
        major_step = 20
    elif y_max <= 600:
        major_step = 50
    else:
        major_step = 100

    minor_step = major_step / 6
    major_yticks = np.arange(0, y_max + major_step, major_step)
    minor_yticks = np.arange(0, y_max + minor_step, minor_step)
    plot_left = -0.25
    plot_right = x[-1] + bar_width / 2


    # Primary minor horizontal gridlines
    for y in minor_yticks:
        if not np.any(np.isclose(y, major_yticks)):
            
            ax.plot(
                [plot_left, plot_right],
                [y, y],
                color=minor_grid_color,
                linewidth=0.6,
                zorder=0
            )

            ax.plot(
                [plot_left, plot_left + depth_x],
                [y, y + depth_y],
                color=minor_grid_color,
                linewidth=0.6,
                zorder=0
            )


    # Primary major horizontal gridlines
    for y in major_yticks:
        
        ax.plot(
            [plot_left, plot_right],
            [y, y],
            color=major_grid_color,
            linewidth=0.9,
            zorder=0
        )

        ax.plot(
            [plot_left, plot_left + depth_x],
            [y, y + depth_y],
            color=major_grid_color,
            linewidth=0.9,
            zorder=0
        )


    # Primary major vertical gridlines
    for i in range(len(bin_labels) + 1):
        xpos = -bar_width / 2 + i * bar_width
        ax.plot([xpos, xpos], [0, y_max], color=major_grid_color, linewidth=0.8, zorder=0)


    # Top Perspective Line
    ax.plot(
        [plot_left + depth_x, plot_right + depth_x],
        [y_max + depth_y, y_max + depth_y],
        color=major_grid_color,
        linewidth=0.8,
        zorder=0
    )


    # Step 1 :Back faces first
    for i, value in enumerate(values):
        left = x[i] - bar_width / 2
        right = x[i] + bar_width / 2

        side = Polygon(
            [
                (right, 0),
                (right + depth_x, depth_y),
                (right + depth_x, value + depth_y),
                (right, value)
            ],
            closed=True,
            facecolor=side_color,
            edgecolor=bar_line_color,
            linewidth=1.0,
            zorder=1
        )
        ax.add_patch(side)

        top = Polygon(
            [
                (left, value),
                (right, value),
                (right + depth_x, value + depth_y),
                (left + depth_x, value + depth_y)
            ],
            closed=True,
            facecolor=top_color,
            edgecolor=bar_line_color,
            linewidth=1.0,
            zorder=2
        )
        ax.add_patch(top)


    # Step 2: Draw front faces
    for i, value in enumerate(values):
        left = x[i] - bar_width / 2
        front = Rectangle(
            (left, 0),
            bar_width,
            value,
            facecolor=front_color,
            edgecolor=bar_line_color,
            linewidth=1.0,
            zorder=3
        )
        ax.add_patch(front)


    # Step 3: Data Labels
    for i, value in enumerate(values):
        ax.text(
            x[i] + depth_x / 2,
            value + depth_y + y_max * 0.015,
            f'{int(value)}',
            ha='center',
            va='bottom',
            fontsize=10,
            color=text_color,
            zorder=5
        )


    # Title
    ax.set_title(title, fontsize=20, color='#595959', pad=20)

    # Axis Labels
    ax.set_xlabel(x_label, fontsize=11, color=text_color,labelpad=15)
    ax.set_ylabel(y_label, fontsize=11, color=text_color, labelpad=10)

    # Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=10, color=text_color)
    ax.set_yticks(major_yticks)
    ax.set_yticklabels([str(int(y)) for y in major_yticks], fontsize=10, color=text_color)

    # Limits
    ax.set_xlim(plot_left - 0.15, plot_right + depth_x + 0.05)
    ax.set_ylim(0, y_max + depth_y * 2)

    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.tick_params(axis='both',length=0)
    ax.grid(False)
    plt.tight_layout()
    return fig, ax



# Chart 1 --- Obesity Level Distribution
def obesity_level_chart(ods):
    
    obesity_counts = (
        ods['NObeyesdad']
        .value_counts()
        .reindex(obesity_order)
    )

    fig, ax = excel_style_3d_column_chart(
        categories=obesity_counts.index.tolist(),
        values=obesity_counts.values.tolist(),
        title='Distribution of Obesity Levels',
        x_label='Obesity Level',
        y_label='Frequency'
    )
    return fig


# Chart 2 --- Transportation Method Distribution
def transportation_chart(ods):
    transport_counts = (ods['MTRANS'].value_counts())

    fig, ax = excel_style_3d_column_chart(
        categories=transport_counts.index.tolist(),
        values=transport_counts.values.tolist(),
        title='Distribution of Transportation Methods',
        x_label='Transportation Method',
        y_label='Frequency'
    )
    return fig


# Chart 3 --- Weight Distribution
def weight_chart(ods):
    
    weight_bins = np.arange(30, 181, 10)
    weight_group = pd.cut(ods['Weight'],bins=weight_bins,right=False,include_lowest=True)
    weight_counts = weight_group.value_counts(sort=False)
    weight_labels = [
        f'{int(interval.left)}-{int(interval.right)}'
        for interval in weight_counts.index
    ]
    
    fig, ax = excel_style_3d_histogram_same_width(
        bin_labels=weight_labels,
        values=weight_counts.values.tolist(),
        title='Distribution of Weight',
        x_label='Weight Range (kg)',
        y_label='Frequency',
        figsize=(13, 7),
        bar_width=0.45
    )
    return fig


# Chart 4 --- Height Distribution
def height_chart(ods):

    height_bins = np.arange(1.45, 2.01,0.05)
    height_group = pd.cut(ods['Height'], bins=height_bins, right=False, include_lowest=True)
    height_counts = height_group.value_counts(sort=False)
    height_labels = [
        f'{interval.left:.2f}-{interval.right:.2f}'
        for interval in height_counts.index
    ]

    fig, ax = excel_style_3d_histogram_same_width(
        bin_labels=height_labels,
        values=height_counts.values.tolist(),
        title='Distribution of Height',
        x_label='Height (m)',
        y_label='Frequency',
        figsize=(13, 7),
        bar_width=0.45
    )
    return fig