import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def create_animated_graph():
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("#1c2128")

    json_file = "generation_data.json"
    if not os.path.exists(json_file):
        print(
            f"Error: '{json_file}' not found. Please run the main Tetris AI script first to generate data."
        )
        return

    with open(json_file, "r") as f:
        data = json.load(f)

    if not data:
        print("JSON file is empty. No data to plot.")
        return

    generations = [item["generation"] for item in data]
    high_scores = [item["high_score"] for item in data]

    def animate(i):

        current_gens = generations[: i + 1]
        current_scores = high_scores[: i + 1]

        ax.clear()
        ax.plot(current_gens, current_scores, color="cyan", linewidth=2)
        ax.set_title(
            "AI Score Progression Over Generations", color="white", fontsize=16
        )
        ax.set_xlabel("Generation", color="white", fontsize=12)
        ax.set_ylabel("High Score", color="white", fontsize=12)

        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        
        if current_scores:
            ax.set_xlim(0, len(generations))
            ax.set_ylim(0, max(high_scores) * 1.1)

    animation.FuncAnimation(
        fig, animate, frames=len(generations), interval=50, repeat=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_animated_graph()
