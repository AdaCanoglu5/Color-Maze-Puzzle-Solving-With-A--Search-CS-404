import heapq
import time
import os
import psutil
from collections import deque

import pandas as pd 

# Constants for the maze
EMPTY = '0'
WALL = 'X'
COLORED = 'S'
CURRENT = 'C'

# Directions the current cell can move
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

def find_current_and_uncolored(maze):
    """Find the position of the current cell and count uncolored tiles."""
    current_pos = None
    uncolored_count = 0
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == CURRENT:
                current_pos = (i, j)
            elif maze[i][j] == EMPTY:
                uncolored_count += 1
    return current_pos, uncolored_count

def move_and_color(maze, start, direction):
    """Move in the given direction until hitting a wall and color the path, also returns the direction taken."""
    new_maze = [list(row) for row in maze]  # Copy the maze
    x, y = start
    steps = 0
    while 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != WALL:
        new_maze[x][y] = COLORED  # Color the current tile
        x, y = x + direction[0], y + direction[1]
        steps += 1
    # Move one step back (hit the wall or boundary)
    x, y = x - direction[0], y - direction[1]
    steps -= 1  # Correct the number of steps
    new_maze[x][y] = CURRENT  # Mark new current position
    return new_maze, (x, y), steps, direction  # Include direction in the return value

def heuristic(maze):
    """Heuristic function based on the number of uncolored tiles."""
    _, uncolored_count = find_current_and_uncolored(maze)
    return uncolored_count

def a_star_search(initial_maze):
    start_time = time.time()  # Start timing
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB

    start_pos, _ = find_current_and_uncolored(initial_maze)
    # Priority queue: (total_cost, steps, current_pos, maze_configuration, path)
    frontier = [(heuristic(initial_maze), 0, start_pos, initial_maze, [])]
    visited = set()
    expanded_nodes = 0  # Count the number of expanded nodes
    max_memory = 0  # Track the maximum memory used
    g_values = {start_pos: 0}  # Dictionary to store the cost from start to the node

    heuristic_is_monotonic = True  # Assume the heuristic is monotonic until proven otherwise

    while frontier:
        total_cost, steps, current_pos, maze, path = heapq.heappop(frontier)
        expanded_nodes += 1

        if (current_pos, tuple(map(tuple, maze))) in visited:
            continue
        visited.add((current_pos, tuple(map(tuple, maze))))

        if heuristic(maze) == 0:
            total_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
            max_memory = max(max_memory, final_memory - initial_memory)
            return maze, steps, path, expanded_nodes, total_time, max_memory, heuristic_is_monotonic

        for direction in DIRECTIONS:
            new_maze, new_pos, path_cost, move_direction = move_and_color(maze, current_pos, direction)
            if new_pos not in visited:
                new_steps = steps + path_cost
                estimated_cost = new_steps + heuristic(new_maze)
                new_g = g_values[current_pos] + path_cost
                g_values[new_pos] = new_g  # Update g value for the new position

                # Check for heuristic monotonicity
                if new_pos in g_values and g_values[new_pos] + heuristic(new_maze) < g_values[current_pos] + heuristic(maze):
                    heuristic_is_monotonic = False  # Set to False if monotonicity is violated

                new_path = path + [(move_direction, new_maze)]
                heapq.heappush(frontier, (estimated_cost, new_steps, new_pos, new_maze, new_path))

        # Update max memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        max_memory = max(max_memory, current_memory - initial_memory)

    return None, None, None, expanded_nodes, time.time() - start_time, max_memory, heuristic_is_monotonic

def heuristic_h1(maze, weight=1.5):
    """Modified heuristic function that overestimates based on a weight."""
    _, uncolored_count = find_current_and_uncolored(maze)
    return uncolored_count * weight

def a_star_search_h1(initial_maze, heuristic_weight=1.5):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)

    start_pos, _ = find_current_and_uncolored(initial_maze)
    frontier = [(heuristic_h1(initial_maze, heuristic_weight), 0, start_pos, initial_maze, [])]
    visited = {}
    expanded_nodes = 0
    max_memory = 0

    while frontier:
        total_cost, steps, current_pos, maze, path = heapq.heappop(frontier)
        expanded_nodes += 1

        if (current_pos, tuple(map(tuple, maze))) in visited and visited[(current_pos, tuple(map(tuple, maze)))] <= total_cost:
            continue

        visited[(current_pos, tuple(map(tuple, maze)))] = total_cost

        if heuristic_h1(maze, heuristic_weight) == 0:
            total_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024 * 1024)
            max_memory = max(max_memory, final_memory - initial_memory)
            return maze, steps, path, expanded_nodes, total_time, max_memory

        for direction in DIRECTIONS:
            new_maze, new_pos, path_cost, move_direction = move_and_color(maze, current_pos, direction)
            new_steps = steps + path_cost
            estimated_cost = new_steps + heuristic_h1(new_maze, heuristic_weight)

            if (new_pos, tuple(map(tuple, new_maze))) not in visited or visited[(new_pos, tuple(map(tuple, new_maze)))] > estimated_cost:
                new_path = path + [(move_direction, new_maze)]
                heapq.heappush(frontier, (estimated_cost, new_steps, new_pos, new_maze, new_path))

        current_memory = process.memory_info().rss / (1024 * 1024)
        max_memory = max(max_memory, current_memory - initial_memory)

    return None, None, None, expanded_nodes, time.time() - start_time, max_memory

# Example usage:
maze_examples = [
    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #HARD 13 #0
        ["X", "0", "0", "0", "X", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "X", "0", "0", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "X", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "X", "0", "X", "X"],
        ["X", "0", "0", "0", "0", "X", "X", "0", "X", "0", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "C", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 11 #1
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "X", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "X", "0", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "X", "X", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "X", "X", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "C", "X", "0", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 5 #2
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "X", "X", "X", "X", "X", "X", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "X", "X", "X", "X", "0", "X", "X", "X"],
        ["X", "0", "X", "0", "X", "X", "X", "X", "0", "X", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "C", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #EASY custom #3
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "X", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "X", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "X", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "X", "X", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "C", "0", "0", "0", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #EASY 1 #4
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "0", "X", "X", "X", "X", "X", "X", "X", "0", "X", "X"],
        ["X", "0", "0", "X", "X", "X", "X", "X", "X", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "0", "X", "X"],
        ["X", "0", "X", "X", "X", "X", "X", "X", "X", "0", "X", "X"],
        ["X", "C", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 6 #5
        ["X", "X", "0", "X", "X", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "X", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "C", "0", "X", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 10 #6
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "X", "X", "X", "0", "0", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "X", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "X", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "X", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "X", "0", "0", "X", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "X", "X", "0", "X", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "X", "X", "0", "X", "0", "X", "0", "X", "X"],
        ["X", "X", "C", "0", "0", "0", "0", "0", "X", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 12 #7
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "0", "X", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "X", "0", "0", "X", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "X", "0", "0", "X", "0", "0", "X", "X"],
        ["X", "X", "X", "X", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "X", "0", "0", "0", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "0", "0", "C", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 18 #8
        ["X", "0", "0", "0", "0", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "X", "X", "0", "X", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "X", "X", "0", "0", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "X", "0", "0", "0", "X", "X", "X", "X", "X"],
        ["X", "0", "X", "X", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "C", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #HARD 20 #9
        ["X", "0", "0", "0", "0", "0", "X", "0", "0", "0", "X", "X"],
        ["X", "X", "X", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "X", "0", "X", "X"],
        ["X", "0", "X", "0", "X", "X", "0", "0", "0", "0", "X", "X"],
        ["X", "0", "X", "0", "X", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "X", "X", "0", "0", "0", "0", "0", "0", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "C", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 21 #10
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "X", "0", "X", "0", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "X", "0", "X", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "X", "0", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "X", "0", "0", "C", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED #11
        ["X", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "X", "0", "X", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "X", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "X", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "C", "0", "0", "0", "0", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "X", "0", "0", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #EASY 2 #12 
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "C", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "0", "X", "X", "0", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "0", "X", "X", "0", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "0", "X", "X", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "X", "X", "0", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #EASY 3 #13
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "0", "X", "0", "X", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "0", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "0", "0", "0", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "X", "X", "C", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #EASY 3 #14
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "0", "X", "X", "X", "X", "X", "0", "0", "X", "X"],
        ["X", "X", "0", "X", "0", "0", "0", "X", "0", "0", "X", "X"],
        ["X", "X", "0", "X", "0", "X", "0", "X", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "0", "0", "0", "0", "C", "X", "X"],
        ["X", "X", "0", "X", "0", "X", "0", "X", "X", "0", "X", "X"],
        ["X", "X", "0", "X", "0", "X", "0", "X", "X", "0", "X", "X"],
        ["X", "X", "0", "X", "0", "X", "0", "0", "0", "0", "X", "X"],
        ["X", "X", "0", "0", "0", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ],

    [
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"], #MED 40 #15
        ["X", "0", "0", "0", "0", "X", "0", "0", "0", "0", "0", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "X"],
        ["X", "X", "X", "X", "0", "X", "X", "X", "X", "X", "X", "X"],
        ["X", "X", "X", "X", "0", "0", "0", "0", "0", "0", "0", "X"],
        ["X", "X", "X", "X", "0", "X", "X", "X", "X", "X", "0", "X"],
        ["X", "X", "X", "X", "0", "X", "X", "X", "X", "X", "0", "X"],
        ["X", "X", "X", "0", "0", "0", "0", "0", "0", "X", "0", "X"],
        ["X", "0", "0", "0", "0", "0", "0", "0", "0", "X", "0", "X"],
        ["X", "0", "X", "X", "0", "X", "X", "0", "0", "X", "0", "X"],
        ["X", "C", "X", "X", "0", "X", "X", "0", "0", "0", "0", "X"],
        ["X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X", "X"]
    ]
]

def analyze_maze_complexity(maze):
    decision_points = 0
    dead_ends = 0
    rectangles = 0

    start = None
    visited = set()

    for y, row in enumerate(maze):
        for x, value in enumerate(row):
            if value == 'C':
                start = (y, x)
                break
        if start:
            break

    if not start:
        print("No starting point ('C') found in the maze.")
        return 'Unknown', 0

    def explore_rectangle(y, x):
        if (y, x) in visited or maze[y][x] != '0':
            return 0
        visited.add((y, x))
        # Check all four directions
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]):
                explore_rectangle(ny, nx)
        return 1

    total_open_space = 0
    paths_to_dead_ends = 0

    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] == '0':  # Open space
                total_open_space += 1
                open_neighbors = sum(
                    1 for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    if 0 <= y + dy < len(maze) and 0 <= x + dx < len(maze[0]) and maze[y + dy][x + dx] == '0'
                )

                if open_neighbors > 2:
                    decision_points += 1
                elif open_neighbors == 1:
                    dead_ends += 1
                    if start:  # Count path length to this dead-end from start
                        path_length = abs(start[0] - y) + abs(start[1] - x)  # Manhattan distance
                        paths_to_dead_ends += path_length

                if (y, x) not in visited:
                    rectangles += explore_rectangle(y, x)

    complexity_score = (decision_points * 15 + dead_ends * 40) - (rectangles * 30)

    difficulty = 'Difficult' if complexity_score >= 550 else 'Normal' if complexity_score >= 200 else 'Easy'
    
    return difficulty, complexity_score

for i, maze in enumerate(maze_examples):
    difficulty, complexity_score = analyze_maze_complexity(maze)
    print(f'Maze {i}: Estimated Difficulty = {difficulty}, Complexity Score = {complexity_score}')

print("")

if(False): #Creates the table for comparisions / make it False if you dont want to create the table. I wanted to make a single .py file
    # Initialize a list to hold all the results
    results = []

    # Define a function to run both heuristics on a maze and return the metrics
    def compare_heuristics(maze, index):
        # Run A* with the original heuristic
        solution_maze_h, steps_h, _, expanded_nodes_h, time_h, memory_h, _ = a_star_search(maze)
        
        # Run A* with the modified heuristic
        solution_maze_h1, steps_h1, _, expanded_nodes_h1, time_h1, memory_h1 = a_star_search_h1(maze)
        
        # Get maze difficulty and size
        difficulty, _ = analyze_maze_complexity(maze)
        maze_size = len(maze) * len(maze[0])  # assuming rectangular maze
        
        # Collect results
        return {
            'Maze Index': index,
            'Size': maze_size,
            'Difficulty': difficulty,
            'Distance h2': steps_h,
            'Distance h1': steps_h1,
            'Expanded Nodes h2': expanded_nodes_h,
            'Expanded Nodes h1': expanded_nodes_h1,
            'CPU Time h2 (s)': time_h,
            'CPU Time h1 (s)': time_h1,
            'Memory h2 (MB)': memory_h,
            'Memory h1 (MB)': memory_h1
        }

    # Compare heuristics for each maze
    for i, maze in enumerate(maze_examples):
        result = compare_heuristics(maze, i)
        results.append(result)

    # Convert results into a DataFrame for nicer display
    results_df = pd.DataFrame(results)

    # Display the results table
    print(results_df.to_string(index=False))

#Choose the maze you want to be solved with both heuristic functions.
idx = 2

print("")

print("h2 solution: \n")

solution_maze, total_steps, solution_path, expanded_nodes, total_time, max_memory, heuristic_is_monotonic = a_star_search(maze_examples[idx])
if solution_maze:
    print("Initial Maze:")
    for row in maze_examples[idx]:
        print(' '.join(row))
    print("\nSolution Steps:")
    
    for index, (direction, maze_state) in enumerate(solution_path):
        direction_map = {(0, 1): "Right", (1, 0): "Down", (0, -1): "Left", (-1, 0): "Up"}
        print(f"Step {index + 1} ({direction_map.get(direction, 'Unknown')}):")
        for row in maze_state:
            print(' '.join(row))
        print()

    print("Final Maze:")
    for row in solution_maze:
        print(' '.join(row))
    
    print("")

    print("Total steps:", total_steps)
    print("Total expanded nodes:", expanded_nodes)
    print("Total time taken (seconds):", total_time)
    print("Maximum memory used (MB):", max_memory)
    print("heuristic is monotonic:", heuristic_is_monotonic)
else:
    print("No solution found")

print("h1 solution: \n")

solution_maze, total_steps, solution_path, expanded_nodes, total_time, max_memory = a_star_search_h1(maze_examples[idx])
if solution_maze:
    print("Initial Maze:")
    for row in maze_examples[idx]:
        print(' '.join(row))
    print("\nSolution Steps:")
    
    for index, (direction, maze_state) in enumerate(solution_path):
        direction_map = {(0, 1): "Right", (1, 0): "Down", (0, -1): "Left", (-1, 0): "Up"}
        print(f"Step {index + 1} ({direction_map.get(direction, 'Unknown')}):")
        for row in maze_state:
            print(' '.join(row))
        print()

    print("Final Maze:")
    for row in solution_maze:
        print(' '.join(row))
    
    print("")

    print("Total steps:", total_steps)
    print("Total expanded nodes:", expanded_nodes)
    print("Total time taken (seconds):", total_time)
    print("Maximum memory used (MB):", max_memory)
else:
    print("No solution found")
