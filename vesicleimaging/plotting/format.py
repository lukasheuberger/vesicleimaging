def formatLH(figsizex=2, figsizey=2, frame=False, fontsize=12):
    """
      :param: figsizex, integer specifying how many figures should be next to each other in x-direction
      :param: figsizey, integer specifying how many figures should be next to each other in y-direction

    """
    import matplotlib as mpl
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['figure.frameon'] = frame
    mpl.rcParams['font.sans-serif'] = 'Gill Sans'
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['figure.figsize'] = 5.25 / figsizex, 4.75 / figsizey
    mpl.rcParams['axes.labelpad'] = 10
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['legend.fontsize'] = 10
    mpl.rcParams['patch.antialiased'] = True
    mpl.rcParams['axes.labelsize'] = 'x-large'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['axes.spines.right'] = frame
    mpl.rcParams['axes.spines.top'] = frame
    mpl.rcParams["errorbar.capsize"] = 5

def plotProps():
    # todo find out how to implement this in a standard template

    PROPS = {
    "boxprops": {"facecolor": "none", "edgecolor": "black", "linewidth":1},
    "medianprops": {"color": "royalblue", "linewidth":2},
    "whiskerprops": {"color": "black", "markersize":2},
    "capprops": {"color": "black"},
    }
    return PROPS

    # -> USE: **vim.plotProps())

    # examples:

    # sns.catplot(kind='box', data=results[(results['FDG'] == 'mPolymersome')&(results['bGal']=='polymersomes')],
    #             x='pore', y='Mean', col='illumination', sharey=True, width = 0.3, **PROPS)
    #
    # sns.catplot(
    #     kind="box",
    #     data=mPolybPoly_data,
    #     x="pore",
    #     y="Mean",
    #     sharey=True,
    #     width=0.3,
    #     **PROPS
    # )
    # sns.stripplot(data=mPolybPoly_data, x="pore", y="Mean", size=6, color="blue", alpha=0.3)