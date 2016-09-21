import matplotlib.pyplot as plt


def days_to_seconds(days):
    seconds_in_a_trading_day = (9*60 + 30)*60
    return days*seconds_in_a_trading_day


def miliseconds_to_days(ms):
    minutes_in_a_trading_day = 9*60 + 30
    miliseconds_in_a_minute = 60*1000
    return ms/(minutes_in_a_trading_day*miliseconds_in_a_minute)

def preformatted_plot(x_label, y_label):
    fig = plt.figure(figsize=(10.24, 7.68))
    ax = fig.add_subplot(111)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


    return ax

def save_static_plot(ax, filename):
    if len(ax.lines) > 1:
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
