import matplotlib.pyplot as plt
import numpy as np

# This will allow it to work both in Jupyter and in a normal script
def plot(scores, mean_scores):
    plt.clf()  # Clear the current figure
    
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plot both scores and mean scores
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')

    plt.ylim(ymin=0)
    
    # Add the current score at the end of the plot
    plt.text(len(scores)-1, scores[-1], str(scores[-1]), fontsize=12)
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), fontsize=12)
    
    plt.legend()  # Add a legend to distinguish between scores

    # Show the plot without blocking the game loop
    plt.draw()
    plt.pause(0.1)  # Small pause to allow the plot to update

# Call plt.ion() to make sure the plot is interactive in non-blocking mode
plt.ion()

