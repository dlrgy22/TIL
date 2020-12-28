import React, { Component } from 'react';
import '../App.css'
import {
    LineSeries,
    XYPlot,
    HorizontalGridLines,
    VerticalGridLines,
    XAxis,
    YAxis,
}from "react-vis"

class GraphForm extends Component {
    
    render() {
        const { data } = this.props
        return (
            <div>
                <XYPlot xType={"ordinal"} width={1000} height={800}>
                    <LineSeries data={data} strokeStyle={"solid"} stroke={"#0000ff"} />
                    <HorizontalGridLines/>
                    <VerticalGridLines/>
                    <XAxis/>
                    <YAxis/>
                </XYPlot>
            </div>
        );
    }
}

export default GraphForm;