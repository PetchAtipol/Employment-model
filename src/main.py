from analysis import Gradient_Boosting_Regression, Random_Forest
from preprocess import preprocess

while True:
    input_mode = input("\n Which mode that you want to run. \n Press '1' for Analysis\n Press '2' for Preprocess \n Press 'e' for Exit\n")
    # print(type(input_mode))
    if input_mode == "1":
        while True:
            model = input("\n Which model do you want.\n Press '1' for Gradient_Boosting_Regression \n Press '2' for Random_Forest \n Press 'e' for Exit to main program\n")
            if model == "1":
                Gradient_Boosting_Regression()
            elif model == "2":
                Random_Forest()
            elif model == "e":
               break 
    elif input_mode == "2":
        preprocess()
        print("\n Your data stored in data/processed \n")
    elif input_mode == "e":
        break
    else:
        print("\n Wrong input")