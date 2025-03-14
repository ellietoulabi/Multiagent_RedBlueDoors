import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_grid(grid_map, step_number=None):
    """Displays a single grid map using Matplotlib."""
    _draw_grid(grid_map, step_number)
    plt.show()

def visualize_trajectories(trajectories, output_filename="trajectory_animation.gif"):
    """Visualizes a sequence of trajectories and saves as an animated GIF."""
    frames = []
    
    for step, grid_map in enumerate(trajectories):
        fig, ax = _draw_grid(grid_map, step)
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(image)
        plt.close(fig)
    imageio.mimsave(output_filename, frames, format="GIF", fps = 1)
    print(f"GIF saved as {output_filename}")

def _draw_grid(grid_map, step_number=None):
    """Draws a grid map using Matplotlib with enhanced visuals."""
    height, width = len(grid_map), len(grid_map[0])
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.set_xticks(np.arange(width)-0.5, minor=True)
    ax.set_yticks(np.arange(height)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_facecolor("white")
    
    if step_number is not None:
        ax.set_title(f"Step {step_number}", fontsize=14, fontweight='bold')
    
    color_mapping = {
        "#": "saddlebrown",  # Walls (brown)
        "R": "red",  # Red door (red)
        "B": "blue",  # Blue door (blue)
        "0": "pink",  # Agent 0 (pink)
        "1": "#C8A2C8",  # Agent 1 (lilac)
        "O": "white"  # Opened door background
    }
    
    for y in range(height):
        for x in range(width):
            char = grid_map[y][x]
            if char in color_mapping:
                if char in ["0", "1"]:
                    # Draw agents as circles
                    agent_circle = patches.Circle((x, height - y - 1), 0.4, color=color_mapping[char])
                    ax.add_patch(agent_circle)
                elif char == "O":
                    # Draw open door as white with a green checkmark
                    ax.add_patch(patches.Rectangle((x-0.5, height - y - 1-0.5), 1, 1, color=color_mapping[char]))
                    ax.text(x, height - y - 1, "✔", ha='center', va='center', fontsize=12, color='green', fontweight='bold')
                else:
                    # Draw grid elements as squares
                    ax.add_patch(patches.Rectangle((x-0.5, height - y - 1-0.5), 1, 1, color=color_mapping[char]))
    
    return fig, ax
