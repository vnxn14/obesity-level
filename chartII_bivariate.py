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


# Shared Function 1 --- Excel-Style 3D Average Column Chart
def excel_style_3d_avg_column_chart(
    categories,
    values,
    title,
    x_label='',
    y_label='Average Value',
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
    depth_y = max(values) * 0.04 if max(values) > 0 else 0.1
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(chart_bg)
    ax.set_facecolor(plot_bg)
    x = np.arange(len(categories))
    max_value = max(values)


    # y-axis settings
    if max_value <= 1:
        major_step = 0.2
    elif max_value <= 2:
        major_step = 0.5
    elif max_value <= 5:
        major_step = 0.5
    else:
        major_step = 1.0


    y_max = np.ceil((max_value * 1.20) / major_step) * major_step
    minor_step = major_step / 5
    major_yticks = np.arange(0, y_max + major_step, major_step)
    minor_yticks = np.arange(0, y_max + minor_step, minor_step)
    plot_left = -0.55
    plot_right = len(categories) - 0.45


    # Primary Minor Horizontal Gridlines
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


    # Primary Major Horizontal Gridlines
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


    # Primary Major Vertical Gridlines
    for i in range(len(categories) + 1):
        xpos = i - 0.5
        ax.plot([xpos, xpos], [0, y_max], color=major_grid_color, linewidth=0.8, zorder=0)


    # Top Perspective Line
    ax.plot(
        [plot_left + depth_x, plot_right + depth_x],
        [y_max + depth_y, y_max + depth_y],
        color=major_grid_color,
        linewidth=0.8,
        zorder=0
    )


    # Draw the 3D bars for each category and value
    for i, value in enumerate(values):
        left = x[i] - bar_width / 2
        right = x[i] + bar_width / 2


        front = Rectangle(
            (left, 0),
            bar_width,
            value,
            facecolor=front_color,
            edgecolor='none',
            zorder=3
        )
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

        # Data Label
        ax.text(
            x[i] + depth_x / 2,
            value + depth_y + y_max * 0.015,
            f'{value:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            color=text_color,
            zorder=5
        )


    # Title
    ax.set_title(title, fontsize=20, color='#595959', pad=20)

    # Axis labels
    ax.set_xlabel(x_label, fontsize=11, color=text_color, labelpad=15)
    ax.set_ylabel(y_label, fontsize=11, color=text_color, labelpad=10)


    # Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=text_color)

    ax.set_yticks(major_yticks)
    ax.set_yticklabels(
        [
            f'{y:.1f}'.rstrip('0').rstrip('.')
            for y in major_yticks
        ],
        fontsize=10,
        color=text_color
    )


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


# Shared Function 2 --- Excel-Style 3D Percentage Column Chart
def excel_style_3d_percentage_column_chart(
    categories,
    values,
    title,
    x_label='',
    y_label='Percentage (%)',
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
    depth_y = max(values) * 0.04 if max(values) > 0 else 1
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(chart_bg)
    ax.set_facecolor(plot_bg)
    x = np.arange(len(categories))


    # Percentage Scale
    y_max = 100
    major_step = 20
    minor_step = major_step / 5
    major_yticks = np.arange(0, 101, major_step)
    minor_yticks = np.arange(0, 101, minor_step)
    plot_left = -0.55
    plot_right = len(categories) - 0.45


    # Primary Minor Horizontal Gridlines
    for y in minor_yticks:
        if not np.any(np.isclose(y, major_yticks)):
            ax.plot([plot_left, plot_right], [y, y], color=minor_grid_color, linewidth=0.6, zorder=0)
            ax.plot(
                [plot_left, plot_left + depth_x],
                [y, y + depth_y],
                color=minor_grid_color,
                linewidth=0.6,
                zorder=0
            )


    # Primary Major Horizontal Gridlines
    for y in major_yticks:
        ax.plot([plot_left, plot_right], [y, y], color=major_grid_color, linewidth=0.9, zorder=0)
        ax.plot(
            [plot_left, plot_left + depth_x],
            [y, y + depth_y],
            color=major_grid_color,
            linewidth=0.9,
            zorder=0
        )


    # Primary Major Vertical Gridlines
    for i in range(len(categories) + 1):
        xpos = i - 0.5
        ax.plot([xpos, xpos], [0, y_max], color=major_grid_color, linewidth=0.8, zorder=0)


    # Top Perspective Line
    ax.plot(
        [plot_left + depth_x, plot_right + depth_x],
        [y_max + depth_y, y_max + depth_y],
        color=major_grid_color,
        linewidth=0.8,
        zorder=0
    )


    # Draw the 3D bars for each category and value
    for i, value in enumerate(values):
        left = x[i] - bar_width / 2
        right = x[i] + bar_width / 2

        front = Rectangle((left, 0),bar_width,value,facecolor=front_color,edgecolor='none',zorder=3)
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

        # Data Label
        ax.text(
            x[i] + depth_x / 2,
            value + depth_y + y_max * 0.015,
            f'{value:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            color=text_color,
            zorder=5
        )


    # Title
    ax.set_title(title,fontsize=20,color='#595959',pad=20)

    # Axis Labels
    ax.set_xlabel(x_label, fontsize=11, color=text_color, labelpad=15)
    ax.set_ylabel(y_label, fontsize=11, color=text_color, labelpad=10)


    # Ticks
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=text_color)

    ax.set_yticks(major_yticks)
    ax.set_yticklabels(
        [
            f'{int(y)}%'
            for y in major_yticks
        ],
        fontsize=10,
        color=text_color
    )


    # Limits
    ax.set_xlim(-0.7,len(categories) - 0.3 + depth_x)
    ax.set_ylim(0,y_max + depth_y * 2)


    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.tick_params(axis='both',length=0)
    ax.grid(False)
    plt.tight_layout()
    
    return fig, ax



# Chart 1 --- Physical Activity × Obesity Level
def physical_activity_chart(ods):
    physical_activity_avg = (
        ods.groupby('NObeyesdad')['FAF']
        .mean()
        .reindex(obesity_order)
    )

    fig, ax = excel_style_3d_avg_column_chart(
        categories=physical_activity_avg.index.tolist(),
        values=physical_activity_avg.values.tolist(),
        title='Average Physical Activity Frequency Across Obesity Levels',
        x_label='Obesity Level',
        y_label='Average Physical Activity Frequency'
    )
    return fig



# Chart 2 --- Technology Usage Time × Obesity Level
def technology_usage_chart(ods):
    technology_usage_avg = (
        ods.groupby('NObeyesdad')['TUE']
        .mean()
        .reindex(obesity_order)
    )

    fig, ax = excel_style_3d_avg_column_chart(
        categories=technology_usage_avg.index.tolist(),
        values=technology_usage_avg.values.tolist(),
        title='Average Technology Usage Time Across Obesity Levels',
        x_label='Obesity Level',
        y_label='Average Technology Usage Score'
    )
    return fig


# Chart 3 --- Vegetable Consumption × Obesity Level
def vegetable_consumption_chart(ods):
    vegetable_consumption_avg = (
        ods.groupby('NObeyesdad')['FCVC']
        .mean()
        .reindex(obesity_order)
    )

    fig, ax = excel_style_3d_avg_column_chart(
        categories=vegetable_consumption_avg.index.tolist(),
        values=vegetable_consumption_avg.values.tolist(),
        title='Average Vegetable Consumption Across Obesity Levels',
        x_label='Obesity Level',
        y_label='Average Vegetable Consumption Score'
    )
    return fig


# Chart 4 --- Water Consumption × Obesity Level
def water_consumption_chart(ods):
    water_consumption_avg = (
        ods.groupby('NObeyesdad')['CH2O']
        .mean()
        .reindex(obesity_order)
    )
    
    fig, ax = excel_style_3d_avg_column_chart(
        categories=water_consumption_avg.index.tolist(),
        values=water_consumption_avg.values.tolist(),
        title='Average Water Consumption Across Obesity Levels',
        x_label='Obesity Level',
        y_label='Average Water Consumption Score'
    )
    return fig


# Chart 5 --- High-Calorie Food Consumption × Obesity Level
def high_calorie_food_chart(ods):
    favc_binary = ods['FAVC'].map({'yes': 1,'no': 0})
    favc_percentage = (
        pd.DataFrame({'NObeyesdad': ods['NObeyesdad'],'FAVC_binary': favc_binary})
        .groupby('NObeyesdad')['FAVC_binary']
        .mean()
        .mul(100)
        .reindex(obesity_order)
    )

    fig, ax = excel_style_3d_percentage_column_chart(
        categories=favc_percentage.index.tolist(),
        values=favc_percentage.values.tolist(),
        title='High-Calorie Food Consumption Across Obesity Levels',
        x_label='Obesity Level',
        y_label='High-Calorie Food Consumption (%)'
    )
    return fig