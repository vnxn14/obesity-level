import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


labels = [
    'Vegetable\nConsumption',
    'Water\nConsumption',
    'Physical\nActivity',
    'High-Calorie\nFood',
    'Alcohol\nConsumption',
    'Technology\nUsage',
    'Main\nMeals'
]


def prepare_radar_data(ods):
    
    radar_data = ods.copy()
    
    # Convert FAVC into numeric values (no  = 0 ,yes = 1)
    radar_data['FAVC_numeric'] = (
        radar_data['FAVC']
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            'no': 0,
            'yes': 1
        })
    )


    # Convert CALC into numeric values
    radar_data['CALC_numeric'] = (
        radar_data['CALC']
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            'no': 0,
            'sometimes': 1,
            'frequently': 2,
            'always': 3
        })
    )


    # Normalize all variables to 0 - 1
    radar_data['FCVC_norm'] = ((radar_data['FCVC'] - 1) / 2)
    radar_data['CH2O_norm'] = ((radar_data['CH2O'] - 1) / 2)
    radar_data['FAF_norm'] = (radar_data['FAF'] / 3)
    radar_data['FAVC_norm'] = (radar_data['FAVC_numeric'])
    radar_data['CALC_norm'] = (radar_data['CALC_numeric'] / 3)
    radar_data['TUE_norm'] = (radar_data['TUE'] / 2)
    radar_data['NCP_norm'] = ((radar_data['NCP'] - 1) / 3)
    return radar_data


# Format Data Label Values
def format_value(v):
    return f'{v:.2f}'.rstrip('0').rstrip('.')

def draw_excel_style_spider_chart(values, labels, title):
    values = np.array(values)
    n = len(labels)
    angles = np.linspace(
        np.pi / 2,
        np.pi / 2 - 2 * np.pi,
        n,
        endpoint=False
    )


    # Color
    grid_color = '#D9D9D9'
    fill_color = '#5B9BD5'
    edge_color = '#2F5597'
    label_color = '#595959'
    value_color = '#404040'


    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor( '#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.55,1.55)
    ax.set_ylim(-1.30,1.45)
    grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]

    for r in grid_levels:
        points = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        polygon = Polygon(
            points,
            closed=True,
            fill=False,
            edgecolor=grid_color,
            linewidth=0.8
        )
        ax.add_patch(polygon)


    for angle in angles:
        x_end = np.cos(angle)
        y_end = np.sin(angle)
        ax.plot(
            [0, x_end],
            [0, y_end],
            color=grid_color,
            linewidth=0.8
        )


    # Data Polygon
    data_points = np.column_stack([values * np.cos(angles),values * np.sin(angles)])
    data_polygon = Polygon(
        data_points,
        closed=True,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=1.25,
        alpha=0.46
    )
    ax.add_patch(data_polygon)


    # Border and round markers
    outline_points = np.vstack([data_points,data_points[0]])
    ax.plot(
        outline_points[:, 0],
        outline_points[:, 1],
        color=edge_color,
        linewidth=1.25,marker='o',
        markerfacecolor=edge_color,
        markeredgecolor=edge_color,
        markersize=5
    )
    

    #Category Labels
    category_radius = 1.13
    for angle, label in zip(angles,labels):
        x = (category_radius* np.cos(angle))
        y = (category_radius* np.sin(angle))

        # Horizontal alignment
        if np.cos(angle) > 0.15:
            ha = 'left'
        elif np.cos(angle) < -0.15:
            ha = 'right'
        else:
            ha = 'center'

        # Vertical alignment
        if np.sin(angle) > 0.15:
            va = 'bottom'
        elif np.sin(angle) < -0.15:
            va = 'top'
        else:
            va = 'center'

        ax.text(x, y, label, fontsize=10, color=label_color, ha=ha, va=va)



    # Data Value Labels

    value_offset = 0.08

    for angle, value in zip(angles,values):
        x = ((value + value_offset)* np.cos(angle))
        y = ((value + value_offset)* np.sin(angle))

        # Horizontal alignment
        if np.cos(angle) > 0.15:
            ha = 'left'
        elif np.cos(angle) < -0.15:
            ha = 'right'
        else:
            ha = 'center'


        # Vertical alignment
        if np.sin(angle) > 0.15:
            va = 'bottom'
        elif np.sin(angle) < -0.15:
            va = 'top'
        else:
            va = 'center'

        ax.text(x, y, format_value(value), fontsize=10, color=value_color, ha=ha, va=va)

    # Title
    ax.text(0, 1.34, title, fontsize=17, color='#595959', ha='center', va='center')
    plt.tight_layout()

    return fig



# Shared Function _ Calculate Values for 1 Obesity Level
def create_obesity_spider_chart(ods,obesity_level,title):

    # Prepare normalized data
    radar_data = prepare_radar_data(ods)

    # Filter selected obesity level
    selected_data = radar_data[radar_data['NObeyesdad']== obesity_level]

    # Calculate mean normalized values
    values = [
        selected_data['FCVC_norm'].mean(),
        selected_data['CH2O_norm'].mean(),
        selected_data['FAF_norm'].mean(),
        selected_data['FAVC_norm'].mean(),
        selected_data['CALC_norm'].mean(),
        selected_data['TUE_norm'].mean(),
        selected_data['NCP_norm'].mean()
    ]


    # Draw chart
    fig = draw_excel_style_spider_chart(
        values=values,
        labels=labels,
        title=title
    )
    return fig



### CHART 1 --- INSUFFICIENT WEIGHT
def insufficient_weight_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Insufficient_Weight',
        title='Insufficient Weight'
    )


### CHART 2 --- NORMAL WEIGHT
def normal_weight_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Normal_Weight',
        title='Normal Weight'
    )


### CHART 3 --- OBESITY TYPE I
def obesity_type_i_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Obesity_Type_I',
        title='Obesity Type I'
    )


### CHART 4 --- OBESITY TYPE II
def obesity_type_ii_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Obesity_Type_II',
        title='Obesity Type II'
    )


### CHART 5 --- OBESITY TYPE III
def obesity_type_iii_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Obesity_Type_III',
        title='Obesity Type III'
    )


### CHART 6 --- OVERWEIGHT LEVEL I
def overweight_level_i_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Overweight_Level_I',
        title='Overweight Level I'
    )


### CHART 7 --- OVERWEIGHT LEVEL II
def overweight_level_ii_chart(ods):
    return create_obesity_spider_chart(
        ods=ods,
        obesity_level='Overweight_Level_II',
        title='Overweight Level II'
    )