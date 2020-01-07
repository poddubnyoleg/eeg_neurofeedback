#!/usr/bin/env python3

# -*- coding: utf-8 -*-
from bokeh.models.sources import ColumnDataSource
from bokeh.models import LinearColorMapper
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column, widgetbox
from bokeh.models.widgets import TextInput, Button
from bokeh.models import LinearAxis, Range1d
from bokeh.models.glyphs import Text
import numpy as np
import time


class BokehVisuals:
    def create_testing_charts(self):
        # create charts for testing headset
        self.figs = [
            figure(plot_width=200, plot_height=125, toolbar_location=None)
            for i in range(self.channels_number)
        ]
        for f in self.figs:
            f.extra_y_ranges = {"foo": Range1d(start=-100, end=100)}

        [f.add_layout(LinearAxis(y_range_name="foo"), "right") for f in self.figs]

        self.cds = ColumnDataSource(
            data={
                i: []
                for i in [
                    item
                    for sublist in [
                        ["y" + str(i), "f" + str(i)]
                        for i in range(self.channels_number)
                    ]
                    for item in sublist
                ]
                + ["x"]
            }
        )

        [
            (
                self.figs[i].line("x", "y" + str(i), source=self.cds, line_alpha=0.3),
                self.figs[i].line(
                    "x",
                    "f" + str(i),
                    source=self.cds,
                    line_color="red",
                    y_range_name="foo",
                    line_width=1,
                ),
            )
            for i in range(self.channels_number)
        ]

        for f in self.figs:
            f.axis.visible = False
            f.xgrid.grid_line_color = None
            f.ygrid.grid_line_color = None

    def create_features_charts(self):
        # create charts for visualising features
        cds_feats_data = {"x": [], "y": []}
        for i in range(self.channels_number):
            cds_feats_data["value_" + str(i)] = []

        self.cds_feats = ColumnDataSource(data=cds_feats_data)

        colors = [
            "#75968f",
            "#a5bab7",
            "#c9d9d3",
            "#e2e2e2",
            "#dfccce",
            "#ddb7b1",
            "#cc7878",
            "#933b41",
            "#550b1d",
        ]
        mapper = LinearColorMapper(palette=colors, low=10, high=70)

        self.figs_feats = [
            figure(plot_width=400, plot_height=120, toolbar_location=None)
            for i in range(self.channels_number)
        ]

        for f in self.figs_feats:
            f.axis.visible = False
            f.xgrid.grid_line_color = None
            f.ygrid.grid_line_color = None

        [
            self.figs_feats[i].rect(
                x="x",
                y="y",
                width=1,
                height=1,
                source=self.cds_feats,
                line_color=None,
                fill_color={"field": "value_" + str(i), "transform": mapper},
            )
            for i in range(self.channels_number)
        ]

    def create_forecast_status_charts(self):
        self.forecast_status_cds = ColumnDataSource(
            data={"x": [], "stage": [], "prediction": []}
        )
        self.forecast_status_figure = figure(
            plot_width=400, plot_height=100, toolbar_location=None
        )
        self.forecast_status_figure.line(
            "x",
            "stage",
            source=self.forecast_status_cds,
            line_color="red",
            line_alpha=0.5,
        )
        self.forecast_status_figure.line(
            "x", "prediction", source=self.forecast_status_cds, line_color="black"
        )
        self.forecast_status_figure.axis.visible = False
        self.forecast_status_figure.xgrid.grid_line_color = None
        self.forecast_status_figure.ygrid.grid_line_color = None

        # figure for status
        self.status_text_cds = ColumnDataSource(
            data={"x": [0], "y": [0], "text": ["Calibration"]}
        )
        self.status_text_figure = figure(
            plot_width=200, plot_height=100, toolbar_location=None
        )
        self.status_text_figure.axis.visible = False
        self.status_text_figure.xgrid.grid_line_color = None
        self.status_text_figure.ygrid.grid_line_color = None
        status_text = Text(x="x", y="y", text="text", text_color="black")
        self.status_text_figure.add_glyph(self.status_text_cds, status_text)

        self.update_button = Button(label="Start session")

    def __init__(self, helmet):

        self.channels_number = helmet.channels_number

        self.create_testing_charts()
        self.create_features_charts()
        self.create_forecast_status_charts()

        self.doc = curdoc()

        # todo add port selector for real data

    def button_link(self, button_f):
        # todo add stop button
        self.update_button.on_click(button_f)

    def add_layout(self):

        rows = []
        for i in range(int(self.channels_number) / 2):
            rows.append(
                row(self.figs[i * 2 : i * 2 + 2] + self.figs_feats[i * 2 : i * 2 + 2])
            )
        # todo add option for uneven channel count

        rows.append(
            row(
                widgetbox([self.update_button], width=200),
                self.status_text_figure,
                self.forecast_status_figure,
            )
        )

        layout = column(*rows)
        self.doc.add_root(layout)

    def update_tuning(self, new_data, new_filtered_data, filtered_data):
        lsd = len(filtered_data)
        update_data = {"x": np.arange(lsd - len(new_data[:, 0]), lsd, 1)}

        for i in range(self.channels_number):
            update_data["f" + str(i)] = new_filtered_data[:, i]
            update_data["y" + str(i)] = new_data[:, i]

        self.cds.stream(update_data, 1000)

    def update_protocol(self, new_features_data, current_state, current_prediction):
        update_data = {
            "x": new_features_data.index.get_level_values(0).values,
            "y": new_features_data.index.get_level_values(1).values,
        }

        for i in range(self.channels_number):
            update_data["value_" + str(i)] = new_features_data[i].values

        self.cds_feats.stream(
            update_data, len(new_features_data.columns) * self.channels_number * 100
        )

        if current_state[0] == "target":
            stage = 1
        else:
            stage = 0

        self.forecast_status_cds.stream(
            {
                "x": [int(time.time()) - 1],
                "stage": [stage],
                "prediction": [current_prediction],
            },
            1000,
        )

        if current_state[1] == "calibration":
            self.status_text_cds.stream(
                {"x": [0], "y": [0], "text": ["calibration"]}, 1
            )
        else:
            self.status_text_cds.stream({"x": [0], "y": [0], "text": ["feedback"]}, 1)
