def formatLH(figsizex=2, figsizey=2, frame=False):
    """
      :param: figsizex, integer specifying how many figures should be next to each other in x-direction
      :param: figsizey, integer specifying how many figures should be next to each other in y-direction

    """
    import matplotlib as mpl
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['figure.frameon'] = False
    mpl.rcParams['font.sans-serif'] = 'Gill Sans'
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['figure.figsize'] = 5.25 / figsizex, 4.75 / figsizey
    mpl.rcParams['axes.labelpad'] = 10
    mpl.rcParams['figure.autolayout'] = True
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['patch.antialiased'] = True
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['axes.titlesize'] = 'x-large'
    mpl.rcParams['axes.spines.right'] = frame
    mpl.rcParams['axes.spines.top'] = frame
    mpl.rcParams["errorbar.capsize"] = 5

def plotProps():
    # todo find out how to implement this in a standard template
    boxprops = dict(linewidth=1)
    medianprops = dict(linewidth=2, color="royalblue")
    flierprops = dict(markersize=2)
    whiskerprops = dict(color="black")
    
    PROPS = {
        "boxprops": {"facecolor": "none", "edgecolor": "black"},
        "medianprops": {"color": "royalblue"},
        "whiskerprops": {"color": "black"},
        "capprops": {"color": "black"},
    }

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