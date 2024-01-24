
# Life-Expectancy-Predictive-Model

Life expectancy is a statistical measurement used to estimate an individual's lifespan. 

We were tasked with providing an estimate prediction of life expectancies, on a national level.

The data was provided by WHO, and it contains records between 2000 and 2015 across 179 countries. 

Instructed to construct two models: One that uses the least information necessary to make a prediction, as well as a more elaborate one that can be used if countries decide to share more sensitive data.

The objective was to produce a function that takes in relevant population statistics (features) and makes a prediction on the average life expectancy. The function would only be valid if the RMSE was less than 4.5.

Life Expectancy Predictive Model Data taken from https://www.kaggle.com/datasets/lashagoch/life-expectancy-who-updated. The code should be ran in the following order: Group_6_Life_Expectancy_Cleaning_and_EDA.ipynb Group_6_Life_Expectancy_Feature_Engineering.ipynb Group_6_Life_Expectancy_Modelling Group_6_Final_Function.ipynb

A function with only non-sensitive data is available: Group_6_Non-Sensitive_Data.ipynb


## Authors

- [@KacperRawicki](https://github.com/KacperRawicki)

- [@MingHuanGit](https://github.com/MingHuanGit)

## Appendix

### Metadata
| Field                       | Description                                                                                               |
|-----------------------------|-----------------------------------------------------------------------------------------------------------|
| Region                      | 179 countries are distributed in 9 regions. E.g. Africa, Asia, Oceania etc                                |
| Year                        | Years observed from 2000 to 2015                                                                          |
| Infant_deaths               | Represents infant deaths per 1000 population                                                              |
| Under_five_deaths           | Represents deaths of children under five years old per 1000 population                                    |
| Adult_mortality             | Represents deaths of adults per 1000 population                                                           |
| Alcohol_consumption         | Represents alcohol consumption that is recorded in liters of pure alcohol per capita with 15+ years old   |
| Hepatitis_B                 | Represents % of coverage of Hepatitis B (HepB3) immunization among 1-year-olds.                           |
| Measles                     | Represents % of coverage of Measles containing vaccine first dose (MCV1) immunization among 1-year-olds   |
| BMI                         | Measure of nutritional status in adults  (kg/m**2)                                                        |
| Polio                       | Represents % of coverage of Polio (Pol3) immunization among 1-year-olds.                                  |
| Diphtheria                  | Represents % of coverage of Diphtheria tetanus toxoid and pertussis (DTP3) immunization among 1-year-olds |
| Incidents_HIV               | Incidents of HIV per 1000 population aged 15-49                                                           |
| GDP_per_capita              | GDP per capita in current USD($)                                                                          |
| Population_mln              | Total population in millions                                                                              |
| Thinness_ten_nineteen_years | Prevalence of thinness among adolescents aged 10-19 years. BMI < -2 stdev below the median                |
| Thinness_five_nine_years    | Prevalence of thinness among children aged 5-9 years. BMI < -2 stdev below the median.                    |
| Schooling                   | Average years that people aged 25+ spent in formal education                                              |
| Economy_status_Developed    | Developed country                                                                                         |
| Economy_status_Developing   | Developing country                                                                                        |
| Life_expectancy             | Average life expectancy of both genders in different years from 2010 to 2015                              |

## 1) Exploratory-Data-Analysis

### Notable Visualisations

![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/0c6ac0ad-f3ad-4373-bad4-2452b1ae717c)


![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/3ed6e8f3-210c-4a90-8b3a-50570c9f5bfc)


![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/ab2bd82d-63e8-4f2d-8682-3a813d460627)


![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/2b184fa8-566a-4e9f-8ee5-462f417aedd9)


![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/d33fb3a8-e6ef-4876-bff2-3bdb91e75f13)




## 2) Feature-Engineering

### Keypoints


Created a function to use One Hot Encoding and transforming the features to the suitable scaler
``` 
def feature_eng(df): 
    
    df = df.copy() 

    ''' One hot encoding '''
    df = pd.get_dummies(df, columns = ['Region'], prefix = 'Region')

    ''' MinMax scaling '''
    minmax = MinMaxScaler() # Initialise scaler

    Fit and transform scaler   
    df[['BMI', 'Schooling', 'Alcohol_consumption']] = minmax.fit_transform(df[['BMI', 'Schooling', 'Alcohol_consumption']])

    ''' Robust scaling '''
    
 Define list of columns to be robust scaled

    robust_list = ['Infant_deaths',
                   'Under_five_deaths',
                   'Adult_mortality',
                   'Hepatitis_B',
                   'Measles',
                   'Polio',
                   'Diphtheria',
                   'Incidents_HIV',
                   'GDP_per_capita',
                   'Population_mln',
                   'Thinness_ten_nineteen_years',
                   'Thinness_five_nine_years',]

    rob = RobustScaler() # Initialise scaler

    Fit and transform scaler
    
    df[robust_list] = rob.fit_transform(df[robust_list])

    ''' Add constant '''
    df = sm.add_constant(df)

    return df # Return df


```

## 3) First Model

Using all the prelimary selected features to assess where improvements can be made.

This unsuprisingly led to multicollinearity being strong as the condition number was 2.07e+16. There was also the issue of many features having p-values higher than the accepted level (0.05). 

The next step was to elimate the features which were deemed not statistically relevant and made the model unreliable. 

![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/0912c74b-ee67-4146-ba57-140d5d955e82)


## 4) Second Model

Used VIF to have a closer look at the features and the relationship between the independant variables. Anything higher than 5 is deemed to have strong correlation so a function was created to drop those features.

```
def calculate_vif(X, thresh = 5.0):
    variables = list(range(X.shape[1])) 
    dropped = True
    while dropped:
        dropped = False

        # List comprehension to gather all the VIF values of the different variables
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif)) # Get the index of the highest VIF value
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc] # Delete the highest VIF value on condition that it's higher than the threshold
            dropped = True # If we deleted anything, we set the 'dropped' value to True to stay in the while loop

            print('Remaining variables:')
            print(X.columns[variables]) # Print the variables that are still in our set
    return X.iloc[:, variables] # Return our X cut down to the remaining variables

```

Then created a function for the p values 

```
# Define function that drops columns by their p-value in a stepwise manner
def stepwise_selection(X, y, threshold_in = 0.01, threshold_out = 0.05, verbose = True):
    # The function is checking for p-values (whether features are statistically significant) - lower is better

    included = [] # This is going to be the list of features we keep

    while True:
        changed = False
        ''' Forward step '''
        excluded = list(set(X.columns) - set(included)) # Get list of excluded columns
        new_pval = pd.Series(index = excluded, dtype = 'float64') # Create empty series
        for new_column in excluded: # Iterate through each excluded column
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit() # Fit model using included columns and new_column
            new_pval[new_column] = model.pvalues[new_column] # Put p-value of each column into series
        best_pval = new_pval.min() # Get the best p-value
        # Add the feature with the lowest (best) p-value under the threshold to our 'included' list
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin() # 'Lowest' p-value
            included.append(best_feature) # Append feature to 'included' list
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval)) # Specifying the verbose text


        ''' Backward step: removing features if new features added to the list make them statistically insignificant '''
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit() # Fit model using all included columns
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:] # Get all p-values
        worst_pval = pvalues.max() # Null if pvalues is empty
        # If the p-value exceeds the upper threshold, the feature will be dropped from the 'included' list
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval)) # Specifying the verbose text
        if not changed:
            break
    return included

```

Resulting features from functions after fine tuning model

['const', 'Infant_deaths', 'Incidents_HIV', 'GDP_per_capita', 'Region_Asia', 'Region_Central America and Caribbean', 'Region_South America', 'Region_European Union', 'Region_Middle East', 'Region_North America', 'Measles', 'Hepatitis_B']

![image](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/c36517b7-15b5-4268-90d0-b3b62fb2fa84)

All p-values are < 0.05
Condition number - 51.6 (No multicollinearity)
RMSE - 2.43

We were happy with these results and applied our model to the test data. The RMSE was 2.72 from the test data which is similar to the training data. This suggests that the model is robust and reliable.


## 5) Demo

A demonstration of how the predictive model works by implementing user input to calculate the life expectancy of their nation. It provides an option to use the ethical model which doesn't ask questions that are deemed sensitive to answer, with the disclaimer that it may lead to less accurate results. 

[screen-capture.webm](https://github.com/KSR-16/Life-Expectancy-Model/assets/135542281/0e62345d-1ebd-40f6-b5db-c60f9e777f21)








