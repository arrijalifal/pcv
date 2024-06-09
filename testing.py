import numpy as np

# Your array of points
points = [[336, 197], [340, 595], [641, 598], [628, 204]]

points_array = np.array(points)
y_values = points_array[:, 1]
miny_index = np.argmin(y_values)
maxy_index = np.argmax(y_values)

lowest_point = points_array[miny_index].tolist()
highest_point = points_array[maxy_index].tolist()

print("Point with the lowest y-value:", lowest_point)
print("Point with the highest y-value:", highest_point)