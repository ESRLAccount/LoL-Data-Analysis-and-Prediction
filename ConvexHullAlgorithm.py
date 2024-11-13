
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay


def point_in_bounding_box(point, bounding_box):
    (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max) = bounding_box
    x, y = point
    return x_min <= x <= x_max and y_min <= y <= y_max
def point_in_convex_hull(point, delaunay_hull):
    return delaunay_hull.find_simplex(point) >= 0
def point_in_enclosing_circle(point, circle):
    center, radius = circle
    return np.linalg.norm(point - center) <= radius
def assign_points_to_clusters(new_points, cluster_boundaries):
    assigned_clusters = []

    for point in new_points:
        assigned_cluster = None
        for cluster, boundaries in cluster_boundaries.items():
            if 'convex_hull' in boundaries:
                if point_in_convex_hull(point, boundaries['hull_delaunay']):
                    assigned_cluster = cluster
                    break
            elif 'bounding_box' in boundaries:
                if point_in_bounding_box(point, boundaries['bounding_box']):
                    assigned_cluster = cluster
                    break
            elif 'min_enclosing_circle' in boundaries:
                if point_in_enclosing_circle(point, boundaries['min_enclosing_circle']):
                    assigned_cluster = cluster
                    break
        assigned_clusters.append(assigned_cluster)

    return assigned_clusters
def min_enclosing_circle(points):
    """
    Compute the smallest circle that can enclose all the given points.
    Uses Ritter's algorithm for an approximate solution.
    :param points: numpy array of shape (n, 2) where n is the number of points.
    :return: tuple (center, radius) where center is a numpy array of shape (2,)
             representing the center of the circle, and radius is a float.
    """
    # Start with an arbitrary point as the circle center
    center = points[0]
    radius = 0

    # First pass: find the point farthest from the initial center
    for point in points:
        distance = np.linalg.norm(point - center)
        if distance > radius:
            center = point
            radius = distance

    # Second pass: refine the center and radius
    for point in points:
        distance = np.linalg.norm(point - center)
        if distance > radius:
            # Update center and radius
            radius = (radius + distance) / 2
            center = center + (point - center) * (distance - radius) / distance

    return center, radius


# Sample data (x, y, cluster)
df_existing = pd.read_csv('data\\OutputRank\\ClusterResults\\CL_MapZone.csv')

# Precomputed boundaries for each cluster
cluster_boundaries = {}
print("Part 1")
for cluster in df_existing['cluster'].unique():
    cluster_points = df_existing[df_existing['cluster'] == cluster][['position_x', 'position_y']].values

    # Bounding Box
    x_min, y_min = np.min(cluster_points, axis=0)
    x_max, y_max = np.max(cluster_points, axis=0)
    cluster_boundaries[cluster] = {
        'bounding_box': [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    }

    # Convex Hull
    if len(cluster_points) > 2:  # Convex hull requires at least 3 points
        hull = ConvexHull(cluster_points)
        cluster_boundaries[cluster]['convex_hull'] = cluster_points[hull.vertices]
        cluster_boundaries[cluster]['hull_delaunay'] = Delaunay(cluster_points[hull.vertices])

    # Minimum Enclosing Circle
    center, radius = min_enclosing_circle(cluster_points)
    cluster_boundaries[cluster]['min_enclosing_circle'] = (center, radius)

# Load new points from a CSV file
print("Part 2")
new_points_df = pd.read_csv('data\\OutputRank\\MatchTimeline\\MatchTimeline_masterfile_positionRowdata.csv')

new_points = new_points_df[['position_x', 'position_y']].values
print("Part 3")
# Assign the new points to the clusters
assigned_clusters = assign_points_to_clusters(new_points, cluster_boundaries)
new_points_df['assigned_zone'] = assigned_clusters
print("Part 4")
# Save the results to a new CSV file
new_points_df.to_csv('data\\OutputRank\\ClusterResults\\ZoneperMinute_perParticipant.csv', index=False)
print("Part Final")

print(new_points_df)

