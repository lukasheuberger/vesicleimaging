def formatLH(figsizex = 2, figsizey = 2, frame = False):
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