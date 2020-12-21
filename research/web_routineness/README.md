# Web Routineness and Limits of Predictability: Investigating Demographic and Behavioral Differences Using Web Tracking Data 

[Juhi Kulshrestha](http://www.juhikulshrestha.com/), [Marcos Oliveira](https://marcosoliveira.info/), [Orkut Karacalik](https://orkutkaracalik.info/), [Denis Bonnay](http://lumiere.ens.fr/~dbonnay/), [Claudia Wagner](http://claudiawagner.info/)

**Abstract** Understanding human activities and movements on the Web is not only important for computational social scientists but can also offer valuable guidance for the design of online systems for recommendations, caching, advertising, and personalization. In this work, we demonstrate that people tend to follow routines on the Web, and these repetitive patterns of web visits increase their browsing behavior's achievable predictability.  We present an information-theoretic framework for measuring the uncertainty and theoretical limits of predictability of human mobility on the Web. We systematically assess the impact of different design decisions on the measurement. We apply the framework to a web tracking dataset of German internet users. Our empirical results highlight that individual's routines on the Web make their browsing behavior predictable to 85\% on average, though the value varies across individuals. We observe that these differences in the users' predictabilities can be explained to some extent by their demographic and behavioral attributes.

## Jupyter Notebooks:

#### Pre-processing: 
* [[Pre-process 1]](release/00_1_Adding_Category.ipynb) Adding category information from subdomain.
* [[Pre-process 2]](release/00_2_Filtering_Users.ipynb) Filtering users out based on the length of their trajectories.

#### Analyses
* [[Analysis 1]](release/01_Basic_Statistics.ipynb) Basic statistics about the data set.
* [[Analysis 2]](release/02_Framework.ipynb) The predictability measurement framework.
* [[Analysis 3]](release/03_Predictability.ipynb) Comparing predictability of different types trajectories and confidence intervals.
* [[Analysis 4]](release/04_Predictability_and_Demographics.ipynb) Examining the relationship between predictability and demographics. 
* [[Analysis 5]](release/05_Predictability_and_Browsing_Behavior.ipynb) Examining the relationship between predictability and browsing behavior. 
