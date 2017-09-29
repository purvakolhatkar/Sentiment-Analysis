import plotly.graph_objs as go

import plotly.plotly as py

py.sign_in('pkol', 'vgY4QDkrCppeARGwNz0K')


def plotSentiment(sentimentList, asin):
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k']
    i = 0
    features = []
    data=[]
    for feature, pos, neg in sentimentList:
        trace = go.Scatter(
            x=[pos],
            y=[neg],

            mode='markers+text',
            name=feature,
            text=[feature],
            textposition='bottom'
        )
        data.append(trace)
        features.append(feature)
        if i < 7:
            i = i + 1
        else:
            i = 0
    layout = go.Layout(
        title="Plot of sentiment for product: " + asin,
           xaxis=dict(
            title='Positive',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title='Negative',
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='text-chart-basic')

