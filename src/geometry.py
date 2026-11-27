"""Continuous geometry; UAV obstacles are clouds and ship obstacles are islands."""
import math
import numpy as np


def wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi


def unit(angle):
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def segment_circle_collision(start, end, center, radius):
    ax, ay = start
    dx, dy = end[0]-ax, end[1]-ay
    denom = dx*dx + dy*dy
    f = max(0., min(1., ((center[0]-ax)*dx+(center[1]-ay)*dy)/denom)) if denom else 0.
    return (ax+f*dx-center[0])**2+(ay+f*dy-center[1])**2 <= radius*radius


def in_bounds(point, bounds, margin=0.):
    return bounds[0]+margin <= point[0] <= bounds[2]-margin and bounds[1]+margin <= point[1] <= bounds[3]-margin


def sector_mask(x, y, position, heading, radius, half_angle):
    dx, dy = x-position[0], y-position[1]
    distance2 = dx*dx + dy*dy
    # Dot product avoids atan2 and wraps, preserving the actual finite sector.
    projection = dx*math.cos(heading)+dy*math.sin(heading)
    return (distance2 <= radius*radius) & (projection >= np.sqrt(distance2)*math.cos(half_angle))


def water_grid(roi, islands, size=64):
    x = roi[0] + (np.arange(size)+.5)*(roi[2]-roi[0])/size
    y = roi[1] + (np.arange(size)+.5)*(roi[3]-roi[1])/size
    xx, yy = np.meshgrid(x, y)
    sea = np.ones((size, size), dtype=bool)
    for ob in islands:
        sea &= (xx-ob['x'])**2+(yy-ob['y'])**2 > ob['radius']**2
    return xx, yy, sea


def line_clear(start, end, obstacles, margin=0.):
    return not any(segment_circle_collision(start, end, (o['x'], o['y']), o['radius']+margin) for o in obstacles)
