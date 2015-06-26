from pricemyrental import PriceMyRental, load_data_and_models, run_pmr

'''
Test the PriceMyRental pipeline
'''
beds = 1
baths = 2
neighborhood = 'mission district'
address = '350 Alabama St, San Francisco CA, 94110'
parking = 4 #Attached garage
description = 'This beautiful Contemporary 1017 square foot condo home has 1 bedrooms and \
                2.0 bathrooms in the vibrant and hip Inner Mission is a light-filled oasis \
                with views to the west and Twin Peaks. The main level features an open \
                floor plan with soaring ceilings and a wall of windows. The kitchen is \
                adjacent to the living and dining area and includes stainless steel \
                appliances, granite counters, and ceramic tile backsplash. A walk-out \
                balcony and a powder room with w/d hook-ups complete this level. Upstairs\
                 is the mezzanine bedroom with a full bathroom and custom closet system. \
                 One car parking and storage room complete this home in the middle of an \
                 active neighborhood. Easy access to 101/280, public transportation and shuttle stops.'
price = 3000



if __name__=='__main__':        
    rfr, search_df, nhood_medians, vectorizer, nmf, kd = load_data_and_models()

    print run_pmr(beds, baths, address, neighborhood, parking, 
            washer_dryer, description, price, rfr, search_df, 
            nhood_medians, vectorizer, nmf, kd)