#########################################################################################################
#  Description: Collection of functions for various visualization needs
#########################################################################################################
import logging
import logging.config

# Python libraries
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import math

#########################################################################################################
# Global variables
__author__ = "DataCentric1"
__pass__ = 1
__fail__ = 0

#########################################################################################################
# Setup logging
logging.config.fileConfig('logging.conf')

logger = logging.getLogger("info")

#########################################################################################################


class Plots:
    def __init__(self):
        # Create a list of frequently used colors
        self.color_list = list(["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"])

    # Create plot from given x and multiple y datasets
    def basic_2d_plot(self, x, y=(), legends=(), title="", xaxis_label="", yaxis_label=""):
        plt.figure()
        plt.title(title)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)

        plt.grid()

        # plt.fill_between([0.0, 1.0], [0.0, 1.0])
        # plt.fill_between(threshold, precision, alpha=0.1, color="r")
        # plt.fill_between(threshold, recall, alpha=0.1, color="g")
        # plt.fill_between(threshold, fbeta_score, alpha=0.1, color="b")

        index = 0
        color_index = 0

        for yval in y:
            # Recycle if we reach end of unique colors that can be used
            # Hopefully we don't have so many unique "y" datasets that we need to do this
            if color_index > len(self.color_list):
                color_index = 0

            color = colors.cnames[self.color_list[color_index]]

            plt.plot(x, yval, 'o-', color=color, label=legends[index])

            color_index += 1
            index += 1

        plt.legend(loc="best")

        # plt.show()
        plt.savefig("temp_pyplot_images_dont_commit/{0:s}.png".format(title))

    @staticmethod
    # Function to test out code or sub-parts of any visualization routine
    def scratchpad():
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ratio = 1.0 / 3.0
        count = math.ceil(math.sqrt(len(colors.cnames)))
        x_count = count * ratio
        y_count = count / ratio
        x = 0
        y = 0
        w = 1 / x_count
        h = 4 / y_count

        for c in colors.cnames:
            pos = (x / x_count, y / y_count)
            ax.add_patch(patches.Rectangle(pos, w, h, color=c))
            ax.annotate(c, xy=pos)
            if y >= y_count-1:
                x += 1
                y = 0
            else:
                y += 1

        plt.show()
#########################################################################################################

if __name__ == "__main__":
    vis = Plots()
    vis.scratchpad()
