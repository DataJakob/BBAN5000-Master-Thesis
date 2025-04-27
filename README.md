<h1>🧠 BBAN5000 Master Thesis — Portfolio Performance on RL Optimised Portfolios</h1>
<br>
<b>Contributors:</b>
<p> Jakob Lindstrøm & Marcus Hjertaas </p>
<b>Course: </b><p>BBAN5000 - Master's thesis in Business Analytics</p>
<b> Keywords: </b2>
<p> Reinforcement Learning, Portfolio Optimization, Portfolio Performance, Attribution Analysis, Markowitz Portfolio Theory </p>

<h2> Table of Content</h2>
<ul>
  <li>Overview</li>
  <li>Project Structure</li>
  <li> Instructions</li>
  <li> Features</li>
  <li> License</li>
</ul>

<h2> 📚 Overview </h2>
<p>
  This project analyzes the performance of optimized investment portfolios using attribution methods. It evaluates allocation and selection effects, generates visualizations, calculates financial metrics (P/L,     Sharpe, Sortino, Sterling ratios), and integrates ESG scoring analysis.
</p>

<h2>
  📁 Project Structure
</h2>

<p>
.
<br>
├── Data/
<br>
  └── [Input data and prediction weights] 
<br>
├── Expmnt/
<br>
│   └── [EDA and data fit notebooks]
<br>
├── Result/
<br>
│   └── [Tables and visualizations]
<br>
├── src/
<br>
│   └── Analysis
<br>
|       └── [.py scripts for analysis ]
<br>
│   └── Optimisation
<br>
|    └── [.py scripts for optimisation ]
<br>
│   └── DataRetrieval.py
<br>
├── .gitignore
<br>
├── LICENSE
<br>
├── main.py
<br>
├── README.md
<br>
├── requirements.txt


<h2> 
  🛠 Instructions
</h2>
Install python version 3.11.8
<br><br>

Check that it is present:
```bash
py -3.11 --version
```

Create enviroment:
```bash
py -3.11 -m venv MyVenv
```

Activate the enviroment:
```bash
MyVenv\Scripts\activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Run main file:
```bash
python main.py
```


<h2> 
  ✨ Features
</h2>
<ul>
  <li> RL/MPT/equal-weight portfolio optimisation</li>
  <li> ESG-aware and ESG-naive optimisation strategies</li>
  <li> Return, Sharpe, Sortino and Sterling optimisation strategies  </li>
  <li> Performance, ESG and attribution analysis</li>
</ul>

<h2>
  📜 License
</h2>
<p> 
  This project is part of an academic thesis and is not intended for commercial use. If you use parts of this code, cite the contributors appropriately.
</p>
