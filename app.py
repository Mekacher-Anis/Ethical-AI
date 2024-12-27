#installs

#imports
import streamlit as st
import pandas as pd
import numpy as np
from annotated_text import annotated_text

CATEGORIES = [
    #Threshold tags for influential tokens 
    "","Weak Influence","Medium Influence","Strong Influence"
]
THRESHOLDS = [
    #needs to be edited to match actual thresholds
    0, 0.5 , 1
]

TESTINPUT = {
  "Inappropiateness":
  [
    {"token": "Du", "einfluss_wrt": 0.2},
    {"token": "Hurensohn", "einfluss_wrt": 0.7}
  ],
  "Toxic Emotions": 
  [
    {"token": "Du","einfluss_wrt": 0},
    {"token": "Hurensohn", "einfluss_wrt": 1}
  ]
}


#    Here the model can be called to generate a responce on the input.
#    The model predictes if a text is Innaprpriate or not and uses Captum to determin which tokens are influencing the decision.
#    The Influence is formatet in a dictionary: {"Dimension of Innapropriateness": [{"token": "actul_token", "einfluss_wrt": float}]}

#    @input      the raw text input 
#    @return     the input that was given, a Bool True if input was innapropreate, the formated tuples using build_tuples for influential tokens marked for anoteted_text

def predict_on_input(input):
    #call model here
    

    inappropriate = True
    
    #captum answer as a dictionary formated like TESTINPUT

    captum_out = TESTINPUT


    return input, inappropriate, build_tuples(captum_out)



#    Building a formatet string with annotated tokens using annotated_text from streamlit. 
#    The annotations show how high the influenz on the decition of the models was per token based on thresholds.

#    @tup input list of tuples with ("token", float)



def build_string(tup):
    copy = tup
    updated_list = []
    for token, value in copy:
        if value <= 0:
            updated_list.append(token)
        else:
            replacement = next((rep for t, rep in zip(THRESHOLDS, CATEGORIES) if value <= t), CATEGORIES[-1])
        
            updated_list.append((token, replacement))
        
    
    #put the formated string out
    annotated_text(updated_list)    

    return

#
#    Transfoming the captum dictionary in a form usuable by annotated_text a dictionary of lists of tuples.
#   The entries of the dictionary are sorted by categorie. Each list is to be used with annotated text seperately.

#   @dic input dictionary formated like TESTINPUT

def build_tuples(dic):
    #build a dictionary of touples
    tuples = {}
    for category, entries in dic.items():
        tuples[category] = [(entry['token'], entry['einfluss_wrt']) for entry in entries]
    return tuples  

def main():
    st.title("Ethical-Artificial-Inteligence")
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Check an argument for Inappropriatness")
        
        with st.form(key='myform'):
            raw_text = st.text_area("Input an argument here.")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            #process the text
            output, result, category_tuples = predict_on_input(raw_text)
            if result:
                st.error("Inappropriate")
            else:
                st.success("Appropriate")
            
            #test method taking the testinput and iterating through the dictionary
            
            for category,entry in category_tuples.items():
                st.text(f"{category}: ")
                build_string(entry)
                



    else:
        st.subheader("About")
        
if __name__ == '__main__':
    main()