import matplotlib.pyplot as plt


def visualize_rewards(rewards_file: str):
    with open(rewards_file, "r") as infile:
        data = infile.read().split("\n")
        data = [int(point) for point in data if point]
        plt.plot(data)
        plt.show()
    return
