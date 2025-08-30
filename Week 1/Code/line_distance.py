import operations as ops


# l_1 = t * slope_1 + intercept_1
slope_1 = [0, 1, 2, 3]
intercept_1 = [0, 0, 1, 1]
# l_2 = t_prime * slope_2 + intercept_2
slope_2 = [2, 3, -1, 2]
intercept_2 = [1, 1, 0, 0]

int_dif = ops.sub(intercept_1, intercept_2)

# find orthogonal bases of span(slope_1, slope_2)
ortho_1 = slope_1
ortho_2 = ops.sub(slope_2, ops.project(slope_2, ortho_1))

# find projection of intercept_1 - intercept_2 onto span(slope_1, slope_2) because minimum distance between the lines is
# the min distance between (intercept_1 - intercept_2) and span(slope_1, slope_2)
proj = ops.add(ops.project(int_dif, ortho_1), ops.project(int_dif, ortho_2))

print(f"Distance: {ops.dist(proj, int_dif)}")
