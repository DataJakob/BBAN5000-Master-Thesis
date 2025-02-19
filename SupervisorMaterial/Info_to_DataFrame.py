import pandas as pd


# This function applies a model (policy) and generates a dataframe with all the necessary information that 
# is returned in the info-output after the reset- or step-function of the environment.

# Version 2024_01_03

def Info_to_DataFrame(env, model):
    
    # Make the dataframe for the output
    output = pd.DataFrame()    
        
    state, info = env.reset()
    terminated = False
    
    
    while not terminated:
   
        action, _state = model.predict(state, deterministic = True)
        
        # Add action to dictionary
        info['Action'] = action
    
        df_dict = pd.DataFrame([info])
        output = pd.concat([output, df_dict], ignore_index = True)
        
        state, reward, terminated, truncated, info = env.step(action)
    

    # Adding all info from the last iteration
    df_dict = pd.DataFrame([info])
    output = pd.concat([output, df_dict], ignore_index=True)       

    return output