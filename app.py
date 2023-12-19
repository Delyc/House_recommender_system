import pandas as pd
import numpy as np
import streamlit as st
from streamlit_card import card

from house_recommender_system import HouseRecommenderSystem

reload = None

def main():
    st.title('House recommender system: Content Based Recommender System Demo')

    st.session_state["card_clicked"] = False

    # Read the CSV file containing house data
    data_directory = "./austinHousingData.csv"

    if 'df_original' not in st.session_state:
        st.session_state['df_original'] = pd.read_csv(data_directory, low_memory=False)

    print("csv loaded")
    # print(st.session_state['df_original'].head())

    if 'hrs' not in st.session_state:
        hrs = HouseRecommenderSystem(st.session_state['df_original'])
        hrs.fit()
        st.session_state['hrs'] = hrs
    else:
        hrs = st.session_state['hrs']


    print("hrs loaded")
    # print(hrs.df_original.head())

    if 'house_cards_data' not in st.session_state:
        st.session_state['house_cards_data'] = None

    house_cards_data = st.session_state['house_cards_data']

    if 'reset' not in st.session_state:
        st.session_state['reset'] = True
    
    reset = st.session_state['reset']

    if st.button("reset"):
        st.session_state['house_cards_data']  = hrs.df_original.sample(n=10)
        st.session_state['reset'] =  True
        st.session_state["sim_scores"] = []

    if(not hrs.df_original.empty):
        # print("hrs is not empty")
        if house_cards_data is None or house_cards_data.empty and reset:
            # print("house cards reset")
            st.session_state['house_cards_data'] = house_cards_data = hrs.df_original.sample(n=10)
    else:
        st.error("There was an error loading the dataset. Please reload the page")
    
    global reload

    if "widgets" not in st.session_state:
        st.session_state["widgets"] = []
    else:
        for key in st.session_state["widgets"]:
            del key
        st.session_state["widgets"].clear()

    button_index = 0

    for index, house_data in st.session_state['house_cards_data'].iterrows():

        # if st.session_state['house_cards_data'].iloc[0]["zpid"] == house_data["zpid"]:
        #     print("first item: ", house_data["zpid"])
        #     print("reset: ", st.session_state['reset'])
        
        st.session_state["widgets"].append(st.button(' \n '.join([house_data["streetAddress"],
                                                                  f"description: {house_data['description']}",
                f"garage spaces: {house_data['garageSpaces']}",
                f"hasCooling: {house_data['hasCooling']}",
                f"hasHeating: {house_data['hasHeating']}",
                f"homeType: {house_data['homeType']}",
                "etc",
                ],)
                ,key=index)
                )

        if "sim_scores" not in st.session_state:
            pass
        else:
            if st.session_state["sim_scores"]:
                if(button_index == 0):
                    pass
                else:
                    st.write(f"similarity score: {st.session_state['sim_scores'][button_index-1]}")

        if st.session_state["widgets"][button_index]:
            st.session_state['reset'] = False
            
            # print(house_data)

            st.session_state["card_clicked"] = True

            st.session_state['house_cards_data'] = pd.DataFrame(house_data.to_dict(), index=[0])
            # print("card clicked: ", st.session_state['house_cards_data'])

            # print("recommendations returned: ", hrs.recommend(index))

            recommendations, sim_scores = hrs.recommend(index)

            st.session_state["sim_scores"] = sim_scores
        
            st.session_state['house_cards_data']= pd.concat([st.session_state['house_cards_data'], recommendations], ignore_index=True)
            # print("new display: ", st.session_state['house_cards_data'])

            st.rerun()
        button_index+=1
                    



if __name__ == '__main__':
    main()