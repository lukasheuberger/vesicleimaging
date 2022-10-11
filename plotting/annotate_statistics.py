import matplotlib.pyplot as plt


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=4):
    """
    The barplot_annotate_brackets function adds p-values to the barplot.


    Args:
        num1: Specify the number of the left bar to put the bracket over
        num2: Specify which right bar the bracket should be placed over
        data: string to write or number for generating asterixes
        center: centers of all bars (like plt.bar() input)
        height: heights of all bars (like plt.bar() input)
        yerr=None: yerrs of all bars (like plt.bar() input)
        dh=.05: height offset over bar / bar + yerr in axes coordinates (0 to 1)
        barh=.05: bar height in axes coordinates (0 to 1)
        fs=None: font size
        maxasterix=4: maximum number of asterixes to write (for very small p-values)

    Returns:
        A string

    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y + barh, y + barh, y]
    mid = ((lx + rx) / 2, y + barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
