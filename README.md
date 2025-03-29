<h1>BBAN5000-Master-Thesis</h1>
<br>

<h2> Table of Content</h2>
<ul>
  <li>Instructions</li>
  <li> Component To-Do-List</li>
</ul>


<h2> Instructions </h2>
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


<h2> Pipeline:</h2>
<ol>
  <li> Setting initial variables</li>
  <li> Data retrieval (yf): Open, Close, Volume, Rolling Mean, Rolling Volatility</li>
  <li> Markowitz portfolio optimization, dynamic trading</li>
  <li> RL optimization, dynamic trading</li>
  <li> Comparison using OGA (Menchero, 2005)</li>
  <li> Convey result/comparison, visualizations and tables</li>
  <li></li>
</ol>