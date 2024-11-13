import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import distance

# Sample data (x, y, cluster)
df = pd.read_csv('data\\OutputRank\\ClusterResults\\ZoneperMinute_perParticipant.csv')


# Function to plot the clusters with boundaries
def plot_clusters_with_boundaries(df):
    clusters = df['assigned_zone'].unique()
    fig, ax = plt.subplots()
    print(clusters[:5])
    print(len(clusters))
    for cluster in clusters[:5]:
        print(cluster)
        cluster_points = df[df['assigned_zone'] == cluster][['position_x', 'position_y']].values

        colors = ["orange", "purple", "blue", "green", "yellow" , "black"]

        # Plot the points
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                   label=f'Cluster {cluster}',
                   color=colors[int(cluster)])

        # Compute and plot the bounding box
        #x_min, y_min = np.min(cluster_points, axis=0)
        #x_max, y_max = np.max(cluster_points, axis=0)
        #bounding_box = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
        #ax.plot(bounding_box[:, 0], bounding_box[:, 1], 'g--', label=f'Bounding Box {cluster}')

        # Compute and plot the convex hull
        if len(cluster_points) > 2:  # Convex hull requires at least 3 points
            hull = ConvexHull(cluster_points)
            hull_points = cluster_points[hull.vertices]
            ax.plot(np.append(hull_points[:, 0], hull_points[0, 0]), np.append(hull_points[:, 1], hull_points[0, 1]),
                    'r-', label=f'Convex Hull {cluster}')

        # Compute and plot the minimum enclosing circle
        center, radius = min_enclosing_circle(cluster_points)
        circle = plt.Circle(center, radius, color='b', fill=False, linestyle='--',
                            label=f'Min Enclosing Circle {cluster}')
        ax.add_artist(circle)

    ax.set_aspect('equal', 'box')
    #plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cluster Boundaries')
    plt.show()


# Helper function to compute the minimum enclosing circle using Ritter's algorithm
def min_enclosing_circle(points):
    # Find the initial bounding box
    center = np.mean(points, axis=0)
    radius = 0
    for point in points:
        radius = max(radius, np.linalg.norm(point - center))
    for point in points:
        if np.linalg.norm(point - center) > radius:
            center = point
            radius = 0
            for pt in points:
                radius = max(radius, np.linalg.norm(pt - center))
    return center, radius


# Plot the clusters with their boundaries
plot_clusters_with_boundaries(df)
