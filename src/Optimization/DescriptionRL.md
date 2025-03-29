<h1> Description MPT </h1>
No description needed. 
<br>

<h1>  Descrioption RL </h1>
<h2> Environment </h2>
Actions are generated from the model. The actions (weights) are passed into the environment. These weights are then multiplied with returns at time t+1. At the right edge case (last iteration), the weights are generated, but they are not being multiplied with returns for t+1 (as they do not exists).  
<br>
<br>
Reward is therefore being calculated correctly, and the bot can be used for trading, even when it do not have data for returns in the future.
<br>

