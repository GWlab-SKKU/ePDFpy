import dash
import dash_core_components as dcc
import dash_html_components as html

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import numpy as np


import mrcfile
import file
from pathlib import Path
import glob
import numpy as np
import image_process
import matplotlib.pyplot as plt
import cv2

mrc_search_path = '/mnt/experiment/TEM diffraction/'
mrc_file_paths = [str(i) for i in Path(mrc_search_path).rglob("*.mrc")]
random_mrc_files = np.random.choice(mrc_file_paths,10)
mrc_img = mrcfile.open(random_mrc_files[0])
raw_data = mrc_img.data
img = np.array(raw_data)
center = image_process.get_center(img, (120,130),10)
azavg = image_process.get_azimuthal_average(mrc_img.data,center)[0]
new_azavg = azavg * np.arange(0,len(azavg))
new_two = azavg * np.power(np.arange(0,len(azavg)),2)

fig = make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(
    go.Scatter(x=np.arange(0,len(azavg)),y=azavg, name="yaxis data"),
    secondary_y=False,
)
fig.add_trace(
    go.Scatter(x=np.arange(0,len(azavg)),y=new_azavg),
    secondary_y=True,
)
fig.update_yaxes(
    title_text="<b>primary</b> yaxis title",
    secondary_y=False)
fig.update_yaxes(
    title_text="<b>secondary</b> yaxis title",
    secondary_y=True)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div(style={}, children=[
    dcc.Graph(id='fig',figure=fig)])


if __name__ == '__main__':
    app.run_server(debug=True, port=3339, host='0.0.0.0')