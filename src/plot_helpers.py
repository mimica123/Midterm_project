# Just to keep some common plotting function tidy and out of the way
# I'm not sure how including matplotlib here will perform in a
# notebook. I imagine it will work because of how seaborn behaves,
# but it still feels incredibly off. So all of these methods will
# require ax to be passed explicitly.

def two_axes_plot(ax, x, y1, y2,
                  xlabel='', ylabel1='', ylabel2='', colors=None, legend=None):
    '''
    Plot a set of lines on the same plot, splitting them onto two axes.
    The legend will need to be added later.
    
        Parameters:
            ax: The matplotlib axis to plot on. This will be twinned when
                plotting the second set of lines
                
            x (np.array): An array of values to use for the x axis
            
            y1 (np.array): Values to use for the main axis
            y2 (np.array): Values to use for the alternate axis
                If the y arrays are 2d then each column will be treated as one line
                
            xlabel (str): x axis label
            ylabel1 (str): Main y axis label
            ylabel2 (str): Alternate y axis label
            
            colors (list): What colors to use for the lines. If this argument
                is given it should have a length of 2. The first item will
                dictate the colors on the main axis, and the second will dictate
                the colors on the alternate axis.
                
            legend (list): A list of line labels to use in the legend. If left
                out the legend won't be shown.
    '''
    if len(y1.shape) == 2:
        l1 = []
        for i in range(y1.shape[1]):
            l1 += ax.plot(x, y1[:,i], color=(None if colors is None else colors[0][i]))
    else:
        l1 = ax.plot(x, y1, color=(None if colors is None else colors[0]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1)

    ax2 = ax.twinx()
    if len(y2.shape) == 2:
        l2 = []
        for i in range(y2.shape[1]):
            l2 += ax2.plot(x, y2[:,i], color=(None if colors is None else colors[1][i]))
    else:
        l2 = ax2.plot(x, y2, color=(None if colors is None else colors[1]))
    ax2.set_ylabel(ylabel2)
    
    if legend is not None:
        lines = l1 + l2
        ax.legend(lines, legend)
    