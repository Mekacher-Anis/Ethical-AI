#installs

#imports
import streamlit as st
import pandas as pd
import numpy as np
import ast
from annotated_text import annotated_text, parameters

parameters.PALETTE = [
    "#D3212C",
    "#ED944D",
    "#069C56",
]

CATEGORIES = [
    #Threshold tags for influential tokens 
    "Strong Influence","Medium Influence","Weak Influence"
]
THRESHOLDS = [
    #needs to be edited to match actual thresholds
    0.5, 0.1 , 0.00
]




#load csv of the captum testsets output
testset_csv = "tables/testset_together.csv"
#testset_csv = "tables/testset_prediction_text_result.csv"
data = pd.read_csv(testset_csv)



#convert the strings of the csv to its original data form
def convert_to_list_of_tuples(entry):
    try:
        return ast.literal_eval(entry)
    except (ValueError, SyntaxError) as e:
        print(f"Error converting entry: {entry} -> {e}")
        return entry
    
converted_data = data.applymap(convert_to_list_of_tuples)

def get_text(captum_out):
    entries = convert_to_list_of_tuples(captum_out['Inappropriateness'][1])
    text_new = [entry[0].replace('▁', ' ') for entry in entries if not (entry[0] == '[CLS]' or entry[0] == '[SEP]')]

    return "".join(text_new)

#    Here the model can be called to generate a responce on the input.
#    The model predictes if a text is Innaprpriate or not and uses Captum to determin which tokens are influencing the decision.
#    The Influence is formatet in a dictionary: {"Dimension of Innapropriateness": ( "Value_if_Dimension_is_classified", [( "token", "influenz_value")]) }

#    @input      the raw text input 
#    @return     the input that was given, a Bool True if input was innapropreate, the formated tuples using build_tuples for influential tokens marked for anoteted_text

def predict_on_input(random=True,row=None):
    #call model here
    

    
    
    if random:
        captum_out = converted_data.sample(n=1).to_dict(orient="records")[0]
    else:
        #print(type(converted_data.iloc[[int(row)]]))
        #print(type(converted_data.sample(n=1)))
        captum_out = converted_data.iloc[[int(row)]].to_dict(orient="records")[0]

    raw_text = get_text(captum_out)
    if isinstance(captum_out['Inappropriateness'][0], str):
        inappropriate = True if captum_out['Inappropriateness'][0].startswith('1') else False
    else:
        inappropriate = captum_out['Inappropriateness'][0] == 1
    

    return raw_text, inappropriate, build_tuples(captum_out)

def assign_color_to_influences(data):
    result = []

    influence_colors = {
        "Strong Influence": "#D3212C",  # Red
        "Medium Influence": "#ED944D",  # Yellow
        "Weak Influence": "#069C56"    # Green
    }

    for item in data:
        if isinstance(item, tuple) and len(item) == 2:
            text, influence = item
            color = influence_colors.get(influence, "#808080")  # Default to gray if influence not recognized
            result.append((text, "", color))
        else:
            # Keep strings as they are
            result.append(item)

    return result


#    Building a formatet string with annotated tokens using annotated_text from streamlit. 
#    The annotations show how high the influenz on the decition of the models was per token based on thresholds.

#    @tup input list of tuples with ("token", float)

def merge_same_influence(list):
    updated_list = []
    last = None
    for string in list:
        if type(string) == type(("1",1)):
            if string[1] == last:
                
                curr, influence = updated_list.pop()
                updated_list.append((curr + string[0], influence))
            else:
                updated_list.append(string)
                last = string[1]
        else:
            updated_list.append(string)
            last = None
    return updated_list

def build_string(tup):
    copy = tup
    updated_list = []

    for token, value in copy:
        if value <= 0:
            updated_list.append(token)
        else:
            replacement = None
            
            for thresh, rep in zip(THRESHOLDS, CATEGORIES): 
                if value >= thresh:
                    replacement = rep
                    
                    break
            if replacement:
                updated_list.append((token, replacement)) 
            else:
                updated_list.append(token)
     
    updated_list = merge_same_influence(updated_list)
    updated_list = assign_color_to_influences(updated_list)

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
    for category, (value, entries) in dic.items():
        
        entries= convert_to_list_of_tuples(entries)
        
        if isinstance(value, str) and value.startswith('1') or isinstance(value, int) and value == 1:
            
            tuples[category] = [ (entry[0].replace('▁', ' '), entry[1])  for entry in entries if not (entry[0] == '[CLS]' or entry[0] == '[SEP]')]
            
    return tuples  

def main():
    st.title("Ethical-Artificial-Inteligence")
    menu = ["Captum","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Captum":
        st.subheader("Check an argument for Inappropriatness")
        random = st.checkbox("Random Argument", value=False)
        raw_text = "0"
        if not random:    
            with st.form(key='myform'):
                raw_text = st.text_area("Input an index for the desired argument entry")
                submit_text = st.form_submit_button(label='Submit')
        else:
            submit_text = st.button("Get a random Text")
            

        if submit_text:

            #legend
            annotated_text([("Strong Influence","","#D3212C"),("Medium Influence","","#ED944D"),("Weak Influence","","#069C56")])
            #process the text
            #print(raw_text)
            output, result, category_tuples = predict_on_input(random, raw_text)

            
            st.text_area( "This is the chosen text", output, height=200, disabled=True) #uneditable textbox

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