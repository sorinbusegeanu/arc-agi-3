# Board Perception Geometry

v1 board geometry is raw pixel-space geometry only.

Logical cell or grid inference is out of scope for v1.

## Responsibilities

- frame width and height
- normalized bbox and center conventions
- pixel coordinate conventions
- stable position extraction conventions

## Conventions

- pixel origin is top-left
- `bbox` is `(x_min, y_min, x_max, y_max)` in pixel coordinates
- `center` is `(x_center, y_center)` in pixel coordinates
- `position_x` and `position_y` are stable object anchor coordinates in pixel space

