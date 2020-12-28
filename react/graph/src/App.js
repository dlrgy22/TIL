import React, { Component } from 'react';
import GraphForm from './components/GraphForm';
class App extends Component {
  state = {
    fire_data : [
      {
        x : "2020.12.25 10:25",
        y : 10
      },
      {
        x : "2020.12.25 10:35",
        y : 23
      },
      {
        x : "2020.12.25 10:45",
        y : 9
      },
      {
        x : "2020.12.25 10:55",
        y : 3
      }
    ],

    move_data : [
      {
        x : "2020.12.25 10:25",
        y : 1
      },
      {
        x : "2020.12.25 10:35",
        y : 2
      },
      {
        x : "2020.12.25 10:45",
        y : 3
      },
      {
        x : "2020.12.25 10:55",
        y : 0
      }
    ],
    Type : null
  }
  
  // makeGraph = (value) => {
  //   if (value === 'fire') {
  //     return <GraphForm data={this.state.fire_data}/>
  //   }
  //   else if (value === 'move') {
  //     return <GraphForm data={this.state.move_data}/>
  //   }
  // }

  changeType = (value) => {
    this.setState({
      Type : value
    })
  }
  render() {

    return (
      <div>
        <button onClick={() => {this.changeType('fire')}}> fire </button>
        <button onClick={() => {this.changeType('move')}}> move </button>
        {
          this.state.Type === 'fire'
            ? <GraphForm data={this.state.fire_data}/>
            : this.state.Type === 'move'
              ? <GraphForm data={this.state.move_data}/>
              : null
        }
      </div>
    );
  }
}

export default App;