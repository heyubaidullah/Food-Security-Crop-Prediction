This tool aims to predict crop yield values by leveraging various environmental and agricultural features, employing machine learning techniques, and using popular data science libraries.
Food security is a major concern even today in many parts of the world. 
With the onset of technology, and the various resources, I am trying to predict the food production from the crops by analyzing the historic data of a region.
The key libraries I am using are NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn, and XGBoost.
The datasets are sourced from a google sheet repository where the data is provided by the ministries. After the usual steps of data loading, cleaning and preprocessing, I tried feature engineering.
This project implements feature engineering by creating new features like logarithmic transformations of pesticide values, combining temperature and rainfall information, and creating categorical labels for the target variable (crop yield).
Currently, I am using linear regression as the model to predict the output but I am also trying out other models based on their accuracy.
