import matplotlib.pyplot as plt
import numpy as np


def barplot_annotate_brackets(
    num1,
    num2,
    data,
    height,
    down=[0, 0],
    dh=0.05,
    barh=0.05,
    fontsize=None,
    maxasterix=3,
    linewidth=1,
    axis=None,
    downdash=True,
):
    """
    The barplot_annotate_brackets function adds p-values to the barplot.

    Args:
        num1: Specify the number of the left bar to put the bracket over
        num2: Specify which right bar the bracket should be placed over
        data: string to write or number for generating asterixes
        height: heights of all bars (like plt.bar() input)
        dh=.05: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        barh=.05: bar height in axes coordinates (0 to 1)
        fontsize=None: font size
        maxasterix=4: maximum number of asterixes to write (for very small p-values)

    Returns:
        A string

    """
    # Check if data is string
    # If not, use the significance level to create the text
    if isinstance(data, str):
        text = data
    else:
        # Define the significance levels and corresponding asterisks
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ""
        p = 0.05

        # Add asterisks for each level of significance
        while data < p:
            text += "*"
            p /= 10.0

            # Stop adding asterisks if the maximum number is reached
            if maxasterix and len(text) == maxasterix:
                break

        # If no significance, label as 'n. s.' (not significant)
        if len(text) == 0:
            text = "n. s."

    # Define the coordinates for the left and right ends of the bar
    lx, ly = num1, height[0]
    rx, ry = num2, height[1]

    # Get the current y-axis limits
    ax_y0, ax_y1 = plt.gca().get_ylim()

    # Scale the height and bar height according to the y-axis limits
    dh *= ax_y1 - ax_y0
    barh *= ax_y1 - ax_y0

    # Calculate the y-coordinate for the bar
    y = max(ly, ry) + dh

    # Define the coordinates for the bar
    barx = [lx, lx, rx, rx]
    bary = [y - down[0], y + barh, y + barh, y - down[1]]
    mid_downdash = ((lx + rx) / 2, y + barh)
    mid_no_downdash = ((lx + rx) / 2, y)

    # Prepare the kwargs for the text
    kwargs = dict(ha="center", va="bottom")
    if fontsize is not None:
        kwargs["fontsize"] = fontsize

    # Draw the bar
    if axis is not None:
        if downdash:
            axis.plot(barx, bary, c="black", lw=linewidth)
            # Display the text in the middle of the bar
            axis.text(*mid_downdash, text, **kwargs)
        if downdash is False:
            bary = [y, y + barh, y + barh, y]
            axis.plot(barx, bary, c="black", lw=linewidth)
            # Display the text in the middle of the bar
            axis.text(*mid_no_downdash, text, **kwargs)

    else:
        if downdash:
            plt.plot(barx, bary, c="black", lw=linewidth)
            # Display the text in the middle of the bar
            plt.text(*mid_downdash, text, **kwargs)

        if downdash is False:
            bary = [y, y, y, y]
            plt.plot(barx, bary, c="black", lw=linewidth)
            # Display the text in the middle of the bar
            plt.text(*mid_no_downdash, text, **kwargs)
        # plt.plot(barx, bary, c='black', lw=linewidth)
        # Display the text in the middle of the bar
        # plt.text(*mid, text, **kwargs)

# todo fix that if needs two values for height
# todo cleanup
# todo consolidate cases


if __name__ == "__main__":
    data = np.array([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]])
    plt.bar(x=[1, 2, 3, 4, 5, 6], height=data.mean(axis=0))  # yerr=data.std(axis=0))
    barplot_annotate_brackets(
        1, 2, data="***", height=[1, 5], down=[1, 2], linewidth=12, downdash=False
    )

    plt.show()
