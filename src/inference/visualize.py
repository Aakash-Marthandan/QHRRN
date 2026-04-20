import jax.numpy as jnp

# ANSI color codes natively mapped directly mimicking ARC visual interfaces globally.
ANSI_COLORS = {
    0: "\033[40m  \033[0m",      # Black (Vacuum)
    1: "\033[44m  \033[0m",      # Blue
    2: "\033[41m  \033[0m",      # Red
    3: "\033[42m  \033[0m",      # Green
    4: "\033[43m  \033[0m",      # Yellow
    5: "\033[100m  \033[0m",     # Gray
    6: "\033[45m  \033[0m",      # Magenta
    7: "\033[46m  \033[0m",      # Cyan
    8: "\033[106m  \033[0m",     # Teal
    9: "\033[101m  \033[0m"       # Dark Red (Maroon proxy)
}

def print_grid(grid: jnp.ndarray, title: str = "Grid"):
    """
    Renders topological integer fields geometrically leveraging terminal escape representations.
    """
    matrix = jnp.array(grid).astype(int)
    print(f"--- {title} ---")
    for row in matrix:
        line_str = "".join([ANSI_COLORS.get(int(val), ANSI_COLORS[0]) for val in row])
        print(line_str)
    print()

def print_side_by_side(input_grid, predicted_grid, actual_grid):
    """
    Aligns topological predictions sequentially isolating metrics functionally.
    """
    print_grid(input_grid, "Test Input")
    print_grid(predicted_grid, "QHRR Prediction")
    print_grid(actual_grid, "Target Truth")
