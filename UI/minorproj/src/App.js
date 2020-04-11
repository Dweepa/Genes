import React, { useState, useEffect } from 'react';
import { Button } from 'reactstrap';
import "bootstrap/dist/css/bootstrap.min.css";
import './App.css';
import ReactDOM, {findDOMNode} from "react-dom";
import FadeIn from 'react-fade-in'


function App() {
  const [KNN, setKNN] = useState(0);
  const [ANN, setANN] = useState(0);
  const [RF, setRF] = useState(0);
  const [XT, setXT] = useState(0);
  const [Smiles, setSmiles] = useState(0);
  const [Actual, setActual] = useState(0);
  const [Input, setInput] = useState(0);
  const [isVisible, setVisible] = useState(false);
  const [Sclicked, setSClicked] = useState(false);
  const [Lclicked, setLClicked] = useState(false);
  function getLINCS(event)
  {

    console.log("LINCS");
    setLClicked(true);
    setSClicked(false);
    setVisible(false);
  }

  function getSMILES(event)
  {
    console.log("SMILES");
    setLClicked(false);
    setSClicked(true);
    setVisible(false);
  }

  function getClass(event) {
    setVisible(false);
    event.preventDefault();
    if(Sclicked)
    {
    fetch('http://localhost:5000/getClassification?smile='+Input).then(res => res.json()).then(data => {
        setSmiles(data['smiles']);
        setKNN(data["KNN"]);
        setANN(data["ANN"]);
        setRF(data["RF"]);
        setXT(data["XT"]);
        setActual(data['Actual']);
        setVisible(true);

        return (<FadeIn>
        <div class="col-sm rounded border border-info text-info" style={{'letter-spacing':'2px','text-align':'center','font-size':'80%','margin-left':'1%','padding':'20px', "background-color":"#282c34"}}>KNN predicts {KNN}</div>
        </FadeIn>)
      });

    }
    else if(Lclicked)
    {
      fetch('/LINCS/'+Input).then(res => res.json()).then(data => {
          setSmiles(data['genes']);
          setKNN(data["KNN"]);
          setANN(data["ANN"]);
          setRF(data["RF"]);
          setXT(data["XT"]);
          setActual(data['Actual']);
          setVisible(true);

          return (<FadeIn>
          <div class="col-sm rounded border border-info text-info" style={{'letter-spacing':'2px','text-align':'center','font-size':'80%','margin-left':'1%','padding':'20px', "background-color":"#282c34"}}>KNN predicts {KNN}</div>
          </FadeIn>)
        });
    }

  }
  return (
    <div className="App">
      <div className="App-header">
        <div style={{'padding-top':'100px'}}>
            {Sclicked==Lclicked && !Sclicked &&
        <div>
      <Button outline style={{"border-color":"pink","color":"pink", "margin":"50px", "margin-top":"200px"}} onClick={getLINCS}> LINCS </Button>
      <Button outline style={{"border-color":"pink","color":"pink", "margin":"50px", "margin-top":"200px"}} onClick={getSMILES}> SMILES </Button>
    </div>}
            {Lclicked &&
              <div>
              <FadeIn>
            <Button outline style={{"border-color":"teal","color":"teal", "margin-bottom":"20px"}} onClick={getSMILES}> Switch to SMILES </Button>
            <p style={{"color":"pink"}}>LINCS</p>
            <form onSubmit={getClass}>
            <input type="text" class="rounded border border-primary" name="smiles" style={{'letter-spacing':'2px','text-align':'center','color':'white','width':'750px','font-size':'80%','margin':'10px','padding':'20px', "background-color":"#282c34"}}onChange={event => setInput(event.target.value)}/>
            <br/>
              <Button type="submit" outline color="warning" style={{'margin':'10px','padding':'10px'}}> Submit </Button>
            </form>
            </FadeIn>
            </div>}

            {Sclicked &&
              <div>
              <FadeIn>
            <Button outline style={{"border-color":"teal","color":"teal", "margin-bottom":"20px"}} onClick={getLINCS}> Switch to LINCS </Button>
            <p style={{"color":"pink"}}>SMILES</p>
            <form onSubmit={getClass}>
            <input type="text" class="rounded border border-primary" name="smiles" style={{'letter-spacing':'2px','text-align':'center','color':'white','width':'750px','font-size':'80%','margin':'10px','padding':'20px', "background-color":"#282c34"}}onChange={event => setInput(event.target.value)}/>
            <br/>
              <Button type="submit" outline color="warning" style={{'margin':'10px','padding':'10px'}}> Submit </Button>
            </form>
            </FadeIn>
            </div>
            }

            { Sclicked!=Lclicked &&
              <div> <FadeIn>
              <div>
                    {isVisible &&
                      <FadeIn>
                      {Smiles}
                      </FadeIn>}

                  <div class="container">

                  {isVisible &&
                    <FadeIn>

                    <div class="row" style={{'margin-top':'20px'}}>
                    <div class="col-sm rounded border border-info text-info" style={{'letter-spacing':'2px','text-align':'center','font-size':'80%','margin-left':'1%','padding':'20px', "background-color":"#282c34"}}>XT predicts {XT}</div>
                    <div class="col-sm rounded border border-danger text-danger" style={{'letter-spacing':'2px','text-align':'center','font-size':'80%','margin-left':'2%','padding':'20px', "background-color":"#282c34"}}>RF predicts {RF}</div>
                    <div class="col-sm rounded border border-success text-success" style={{'letter-spacing':'2px','text-align':'center','font-size':'80%','margin-left':'2%','padding':'20px', "background-color":"#282c34"}}>KNN predicts {KNN}</div>
                    </div>

                    <div class="row" style={{'margin-top':'20px'}}>
                    <div class="col-sm rounded border border-info text-info" style={{'letter-spacing':'2px','text-align':'center','font-size':'80%','margin-left':'1%','padding':'20px', "background-color":"#282c34"}}>ANN predicts {ANN}</div>
                    </div>

                    
                    </FadeIn> }

                  </div>
              </div>
          </FadeIn></div>}
          </div>
      </div>
    </div>
  );
}

export default App;
