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
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_xticks(np.arange(width) - 0.5, minor=True)
    ax.set_yticks(np.arange(height) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_facecolor("white")

    if step_number is not None:
        ax.set_title(f"Step {step_number}", fontsize=14, fontweight='bold')

    color_mapping = {
        "#": "saddlebrown",  # Walls (brown)
        "S": "yellow",       # Spots (yellow)
        "0": "pink",         # Agent 0
        "1": "#C8A2C8",      # Agent 1
    }

    for y in range(height):
        for x in range(width):
            char = grid_map[y][x]

            # If agent is occupying a spot (marked as "O"), draw green square + agent circle
            if char == "O":
                ax.add_patch(patches.Rectangle((x - 0.5, height - y - 1 - 0.5), 1, 1, color="green"))
                agent_id = grid_map[y][x + 1] if x + 1 < width and grid_map[y][x + 1] in ["0", "1"] else "0"
                agent_color = color_mapping.get(agent_id, "gray")
                agent_circle = patches.Circle((x, height - y - 1), 0.4, color=agent_color)
                ax.add_patch(agent_circle)
            elif char in ["0", "1"]:
                agent_circle = patches.Circle((x, height - y - 1), 0.4, color=color_mapping[char])
                ax.add_patch(agent_circle)
            elif char in color_mapping:
                ax.add_patch(patches.Rectangle((x - 0.5, height - y - 1 - 0.5), 1, 1, color=color_mapping[char]))

    return fig, ax
